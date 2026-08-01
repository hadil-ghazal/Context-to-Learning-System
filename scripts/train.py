# AI Assistance Attribution:
# This file was generated with assistance from OpenAI ChatGPT:
# https://chatgpt.com/share/6a6d47a0-1a54-83e8-91a4-2f8c8b673fe5

# Hadil Ghazal fine tuned on 7/31/26:
# - removed bug "#overwrite_output_dir=True," in return TrainingArguments( which was causing the model to fail

"""Fine-tune an instruction-tuned language model using LoRA and PEFT."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Final

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_DATASET_PATH: Final[Path] = Path(
    "data/raw/context_learning_dataset.jsonl"
)
DEFAULT_OUTPUT_DIRECTORY: Final[Path] = Path(
    "models/context-learning-lora"
)
DEFAULT_MODEL_NAME: Final[str] = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_RANDOM_SEED: Final[int] = 42
DEFAULT_MAX_SEQUENCE_LENGTH: Final[int] = 768

LOGGER = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for LoRA fine-tuning."""

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a small instruction model using LoRA on the synthetic "
            "context-to-learning dataset."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=(
            "Path to the raw JSONL dataset. "
            f"Default: {DEFAULT_DATASET_PATH}"
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=(
            "Hugging Face model identifier. "
            f"Default: {DEFAULT_MODEL_NAME}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Directory where the trained LoRA adapter will be saved. "
            f"Default: {DEFAULT_OUTPUT_DIRECTORY}"
        ),
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
        help=(
            "Maximum tokenized prompt-response length. "
            f"Default: {DEFAULT_MAX_SEQUENCE_LENGTH}"
        ),
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=3.0,
        help="Number of training epochs. Default: 3.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=2,
        help="Training batch size per device. Default: 2.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps. Default: 8.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Initial learning rate. Default: 2e-4.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.05,
        help="Fraction of examples reserved for validation. Default: 0.05.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Number of optimizer steps between log entries. Default: 10.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Number of optimizer steps between checkpoints. Default: 100.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed. Default: {DEFAULT_RANDOM_SEED}.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure application logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Validate paths and numeric training arguments."""

    if not arguments.dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {arguments.dataset_path}"
        )

    if arguments.max_sequence_length <= 0:
        raise ValueError("Maximum sequence length must be greater than zero.")

    if arguments.num_train_epochs <= 0:
        raise ValueError("Number of training epochs must be greater than zero.")

    if arguments.per_device_train_batch_size <= 0:
        raise ValueError("Training batch size must be greater than zero.")

    if arguments.gradient_accumulation_steps <= 0:
        raise ValueError(
            "Gradient accumulation steps must be greater than zero."
        )

    if arguments.learning_rate <= 0:
        raise ValueError("Learning rate must be greater than zero.")

    if not 0.0 < arguments.validation_split < 1.0:
        raise ValueError("Validation split must be between zero and one.")


def load_raw_dataset(dataset_path: Path) -> Dataset:
    """Load the raw JSONL dataset with the Hugging Face datasets library."""

    LOGGER.info("Loading raw dataset from %s", dataset_path)

    dataset = load_dataset(
        "json",
        data_files={"train": str(dataset_path)},
        split="train",
    )

    if len(dataset) < 2:
        raise ValueError(
            "The dataset must contain at least two examples for training and "
            "validation."
        )

    return dataset


def normalize_response(response: Any) -> dict[str, str]:
    """Validate and normalize a generated example's response object."""

    if not isinstance(response, dict):
        raise ValueError("Each dataset response must be a JSON object.")

    required_response_fields = (
        "educational_concept",
        "grade_appropriate_explanation",
        "curriculum_aligned_learning_connection",
        "practice_question",
        "answer",
    )

    normalized_response: dict[str, str] = {}

    for field_name in required_response_fields:
        field_value = response.get(field_name)

        if field_value is None:
            raise ValueError(
                f"Response is missing required field: {field_name}"
            )

        normalized_response[field_name] = str(field_value).strip()

    return normalized_response


def create_user_prompt(example: dict[str, Any]) -> str:
    """Convert a raw dataset example into an instruction-style user prompt."""

    required_fields = ("grade", "subject", "state", "current_content")

    for field_name in required_fields:
        if field_name not in example:
            raise ValueError(
                f"Dataset example is missing required field: {field_name}"
            )

    grade = str(example["grade"]).strip()
    subject = str(example["subject"]).strip()
    state = str(example["state"]).strip()
    current_content = str(example["current_content"]).strip()

    return (
        "Transform the child's current content into a concise, age-appropriate, "
        "curriculum-aligned learning opportunity.\n\n"
        f"Grade: {grade}\n"
        f"Subject: {subject}\n"
        f"State: {state}\n"
        f"Current content: {current_content}\n\n"
        "Include the relevant educational concept, a grade-appropriate "
        "explanation, a curriculum-aligned learning connection, one practice "
        "question, and its answer."
    )


def create_assistant_response(example: dict[str, Any]) -> str:
    """Convert a structured response object into assistant training text."""

    response = normalize_response(example.get("response"))

    return (
        f"Educational concept: {response['educational_concept']}\n\n"
        "Grade-appropriate explanation: "
        f"{response['grade_appropriate_explanation']}\n\n"
        "Curriculum-aligned learning connection: "
        f"{response['curriculum_aligned_learning_connection']}\n\n"
        f"Practice question: {response['practice_question']}\n\n"
        f"Answer: {response['answer']}"
    )


def convert_to_prompt_response(example: dict[str, Any]) -> dict[str, str]:
    """Convert one raw JSONL example into a prompt-response pair."""

    return {
        "prompt": create_user_prompt(example),
        "response_text": create_assistant_response(example),
    }


def create_prompt_response_dataset(raw_dataset: Dataset) -> Dataset:
    """Convert all raw examples into prompt-response pairs."""

    LOGGER.info("Converting raw examples into prompt-response pairs")

    return raw_dataset.map(
        convert_to_prompt_response,
        remove_columns=raw_dataset.column_names,
        desc="Creating prompt-response pairs",
    )


def load_tokenizer(
    model_name: str,
) -> PreTrainedTokenizerBase:
    """Load and configure the tokenizer for causal language modeling."""

    LOGGER.info("Loading tokenizer: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=False,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither a pad token nor an EOS token."
            )
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_prompt_response(
    example: dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
) -> dict[str, list[int]]:
    """
    Tokenize one prompt-response pair and mask prompt tokens in the labels.

    Masking prompt tokens with -100 ensures that loss is calculated only on
    the assistant response rather than on the user instruction.
    """

    messages_without_response = [
        {
            "role": "system",
            "content": (
                "You create accurate, age-appropriate, curriculum-aligned "
                "learning opportunities from the content a child is viewing."
            ),
        },
        {
            "role": "user",
            "content": example["prompt"],
        },
    ]

    complete_messages = [
        *messages_without_response,
        {
            "role": "assistant",
            "content": example["response_text"],
        },
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages_without_response,
        tokenize=False,
        add_generation_prompt=True,
    )
    complete_text = tokenizer.apply_chat_template(
        complete_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_encoding = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_sequence_length,
    )
    complete_encoding = tokenizer(
        complete_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_sequence_length,
    )

    input_ids = complete_encoding["input_ids"]
    attention_mask = complete_encoding["attention_mask"]
    labels = list(input_ids)

    prompt_token_count = min(
        len(prompt_encoding["input_ids"]),
        len(labels),
    )

    labels[:prompt_token_count] = [-100] * prompt_token_count

    if all(label == -100 for label in labels):
        raise ValueError(
            "A tokenized example contains no response tokens. Increase "
            "--max-sequence-length or shorten the source examples."
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_dataset(
    prompt_response_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
) -> Dataset:
    """Tokenize every prompt-response pair for causal language modeling."""

    LOGGER.info(
        "Tokenizing dataset with maximum sequence length %d",
        max_sequence_length,
    )

    return prompt_response_dataset.map(
        tokenize_prompt_response,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_sequence_length": max_sequence_length,
        },
        remove_columns=prompt_response_dataset.column_names,
        desc="Tokenizing prompt-response pairs",
    )


def split_dataset(
    tokenized_dataset: Dataset,
    validation_split: float,
    random_seed: int,
) -> tuple[Dataset, Dataset]:
    """Split the tokenized dataset into training and validation datasets."""

    split_datasets = tokenized_dataset.train_test_split(
        test_size=validation_split,
        seed=random_seed,
        shuffle=True,
    )

    training_dataset = split_datasets["train"]
    validation_dataset = split_datasets["test"]

    LOGGER.info(
        "Dataset split complete: %d training examples, %d validation examples",
        len(training_dataset),
        len(validation_dataset),
    )

    return training_dataset, validation_dataset


def determine_model_dtype() -> torch.dtype:
    """Choose an appropriate model dtype for the available hardware."""

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    return torch.float32


def load_base_model(model_name: str) -> PreTrainedModel:
    """Load the base causal language model."""

    model_dtype = determine_model_dtype()

    LOGGER.info(
        "Loading base model %s with dtype %s",
        model_name,
        model_dtype,
    )

    model_loading_arguments: dict[str, Any] = {
        "pretrained_model_name_or_path": model_name,
        "torch_dtype": model_dtype,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }

    if torch.cuda.is_available():
        model_loading_arguments["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        **model_loading_arguments,
    )

    model.config.use_cache = False
    return model


def configure_lora_model(
    base_model: PreTrainedModel,
) -> PeftModel:
    """Apply a LoRA adapter configuration to the base model."""

    lora_configuration = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    lora_model = get_peft_model(base_model, lora_configuration)
    lora_model.print_trainable_parameters()

    return lora_model


def create_training_arguments(
    arguments: argparse.Namespace,
) -> TrainingArguments:
    """Create hackathon-friendly Hugging Face training arguments."""

    use_bfloat16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    use_float16 = torch.cuda.is_available() and not use_bfloat16

    return TrainingArguments(
        output_dir=str(arguments.output_dir / "checkpoints"),
        #overwrite_output_dir=True,
        num_train_epochs=arguments.num_train_epochs,
        per_device_train_batch_size=(
            arguments.per_device_train_batch_size
        ),
        per_device_eval_batch_size=(
            arguments.per_device_train_batch_size
        ),
        gradient_accumulation_steps=(
            arguments.gradient_accumulation_steps
        ),
        learning_rate=arguments.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=arguments.logging_steps,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=arguments.save_steps,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        fp16=use_float16,
        bf16=use_bfloat16,
        optim="adamw_torch",
        seed=arguments.seed,
        data_seed=arguments.seed,
    )


def create_trainer(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    training_dataset: Dataset,
    validation_dataset: Dataset,
    training_arguments: TrainingArguments,
) -> Trainer:
    """Create a Hugging Face Trainer for supervised fine-tuning."""

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    return Trainer(
        model=model,
        args=training_arguments,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )


def save_trained_adapter(
    trainer: Trainer,
    tokenizer: PreTrainedTokenizerBase,
    output_directory: Path,
) -> None:
    """Save the trained LoRA adapter and tokenizer."""

    output_directory.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Saving trained LoRA adapter to %s", output_directory)

    trainer.save_model(str(output_directory))
    tokenizer.save_pretrained(str(output_directory))

    training_metadata = {
        "base_model": DEFAULT_MODEL_NAME,
        "adapter_type": "LoRA",
        "task_type": "causal_language_modeling",
        "dataset_path": str(DEFAULT_DATASET_PATH),
    }

    metadata_path = output_directory / "training_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(
            training_metadata,
            metadata_file,
            indent=2,
            ensure_ascii=False,
        )
        metadata_file.write("\n")


def main() -> None:
    """Run the complete LoRA fine-tuning workflow."""

    configure_logging()
    arguments = parse_arguments()
    validate_arguments(arguments)
    set_seed(arguments.seed)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    raw_dataset = load_raw_dataset(arguments.dataset_path)
    prompt_response_dataset = create_prompt_response_dataset(raw_dataset)

    tokenizer = load_tokenizer(arguments.model_name)
    tokenized_dataset = tokenize_dataset(
        prompt_response_dataset=prompt_response_dataset,
        tokenizer=tokenizer,
        max_sequence_length=arguments.max_sequence_length,
    )

    training_dataset, validation_dataset = split_dataset(
        tokenized_dataset=tokenized_dataset,
        validation_split=arguments.validation_split,
        random_seed=arguments.seed,
    )

    base_model = load_base_model(arguments.model_name)
    lora_model = configure_lora_model(base_model)

    training_arguments = create_training_arguments(arguments)
    trainer = create_trainer(
        model=lora_model,
        tokenizer=tokenizer,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        training_arguments=training_arguments,
    )

    LOGGER.info("Starting LoRA fine-tuning")
    trainer.train()

    save_trained_adapter(
        trainer=trainer,
        tokenizer=tokenizer,
        output_directory=arguments.output_dir,
    )

    LOGGER.info(
        "Training complete. LoRA adapter saved to %s",
        arguments.output_dir,
    )


if __name__ == "__main__":
    main()