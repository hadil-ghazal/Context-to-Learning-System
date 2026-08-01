# AI Assistance Attribution:
# This file was generated with assistance from OpenAI ChatGPT:
# https://chatgpt.com/share/6a6d47a0-1a54-83e8-91a4-2f8c8b673fe5

"""Evaluate the base and LoRA fine-tuned Context-to-Learning models."""

from __future__ import annotations

import argparse
import csv
import gc
import logging
from pathlib import Path
from typing import Final, Sequence

import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)


DEFAULT_BASE_MODEL_NAME: Final[str] = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ADAPTER_PATH: Final[Path] = Path("models/context-learning-lora")
DEFAULT_OUTPUT_PATH: Final[Path] = Path(
    "data/outputs/model_comparison.csv"
)
DEFAULT_MAX_NEW_TOKENS: Final[int] = 256
DEFAULT_RANDOM_SEED: Final[int] = 42

SYSTEM_PROMPT: Final[str] = (
    "You create accurate, age-appropriate, curriculum-aligned learning "
    "opportunities from the content a child is viewing."
)

CSV_COLUMNS: Final[tuple[str, str, str]] = (
    "Prompt",
    "Base Model Response",
    "Fine-Tuned Model Response",
)

EVALUATION_EXAMPLES: Final[tuple[dict[str, str | int], ...]] = (
    {
        "grade": 4,
        "subject": "Mathematics",
        "state": "California",
        "current_content": (
            "A Fortnite player compares two landing locations by counting "
            "loot chests, estimating travel distance, and deciding which "
            "location is safer before the storm circle closes."
        ),
    },
    {
        "grade": 5,
        "subject": "Mathematics",
        "state": "Texas",
        "current_content": (
            "A Minecraft tutorial shows how to build a castle wall using "
            "eight equal rows with twelve blocks in each row."
        ),
    },
    {
        "grade": 6,
        "subject": "Science",
        "state": "New York",
        "current_content": (
            "A cooking video shows bread dough before and after yeast causes "
            "it to rise and explains why the dough becomes larger."
        ),
    },
    {
        "grade": 7,
        "subject": "Mathematics",
        "state": "Florida",
        "current_content": (
            "A basketball analysis video compares a player's successful "
            "shots with the total number of shots attempted from different "
            "areas of the court."
        ),
    },
    {
        "grade": 5,
        "subject": "Science",
        "state": "North Carolina",
        "current_content": (
            "A science experiment tests how far a balloon-powered car travels "
            "when different amounts of air are added to the balloon."
        ),
    },
    {
        "grade": 8,
        "subject": "Science",
        "state": "California",
        "current_content": (
            "An animal documentary explains how an octopus uses camouflage, "
            "flexible arms, and problem-solving behaviors to survive in the "
            "ocean."
        ),
    },
    {
        "grade": 9,
        "subject": "English Language Arts",
        "state": "Texas",
        "current_content": (
            "A news explainer presents two viewpoints about a proposal to "
            "replace an empty lot with a public community park."
        ),
    },
    {
        "grade": 7,
        "subject": "Science",
        "state": "Florida",
        "current_content": (
            "A robotics tutorial shows a sensor detecting an obstacle and "
            "sending a signal that makes the robot turn in a new direction."
        ),
    },
    {
        "grade": 6,
        "subject": "Social Studies",
        "state": "New York",
        "current_content": (
            "A travel vlog follows a train through several cities and shows "
            "maps, travel times, rivers, mountains, and major stations."
        ),
    },
    {
        "grade": 10,
        "subject": "English Language Arts",
        "state": "North Carolina",
        "current_content": (
            "A laptop review recommends one device over another using "
            "technical specifications, personal opinions, test results, and "
            "a sponsored promotion from one manufacturer."
        ),
    },
)

LOGGER = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for model evaluation."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare the original base model with the Context-to-Learning "
            "LoRA fine-tuned model."
        )
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=DEFAULT_ADAPTER_PATH,
        help=(
            "Path to the trained LoRA adapter. "
            f"Default: {DEFAULT_ADAPTER_PATH}"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Path for the model comparison CSV. "
            f"Default: {DEFAULT_OUTPUT_PATH}"
        ),
    )
    parser.add_argument(
        "--base-model-name",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face base model identifier. When omitted, the "
            "base model is read from the LoRA adapter configuration."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=(
            "Maximum number of response tokens generated per prompt. "
            f"Default: {DEFAULT_MAX_NEW_TOKENS}"
        ),
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
    """Validate evaluation paths and numeric arguments."""

    if not arguments.adapter_path.exists():
        raise FileNotFoundError(
            f"LoRA adapter path does not exist: {arguments.adapter_path}"
        )

    if not arguments.adapter_path.is_dir():
        raise NotADirectoryError(
            f"LoRA adapter path is not a directory: "
            f"{arguments.adapter_path}"
        )

    required_adapter_files = (
        arguments.adapter_path / "adapter_config.json",
        arguments.adapter_path / "adapter_model.safetensors",
    )
    missing_adapter_files = [
        file_path
        for file_path in required_adapter_files
        if not file_path.is_file()
    ]

    if missing_adapter_files:
        formatted_paths = ", ".join(
            str(file_path) for file_path in missing_adapter_files
        )
        raise FileNotFoundError(
            f"Required LoRA adapter file or files are missing: "
            f"{formatted_paths}"
        )

    if arguments.max_new_tokens <= 0:
        raise ValueError("Maximum new tokens must be greater than zero.")


def resolve_base_model_name(
    adapter_path: Path,
    model_name_override: str | None,
) -> str:
    """Resolve the training base model from the adapter configuration."""

    if model_name_override:
        return model_name_override

    try:
        adapter_configuration = PeftConfig.from_pretrained(
            str(adapter_path)
        )
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Unable to read LoRA adapter configuration from "
            f"{adapter_path}."
        ) from error

    configured_model_name = (
        adapter_configuration.base_model_name_or_path
    )

    if configured_model_name:
        return configured_model_name

    LOGGER.warning(
        "The adapter configuration does not contain a base model name. "
        "Using the training default: %s",
        DEFAULT_BASE_MODEL_NAME,
    )
    return DEFAULT_BASE_MODEL_NAME


def create_training_prompt(
    grade: str | int,
    subject: str,
    state: str,
    current_content: str,
) -> str:
    """Create an evaluation prompt using the training prompt structure."""

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


def build_evaluation_prompts(
    examples: Sequence[dict[str, str | int]],
) -> list[str]:
    """Convert representative examples into training-style prompts."""

    required_fields = {
        "grade",
        "subject",
        "state",
        "current_content",
    }
    prompts: list[str] = []

    for example_index, example in enumerate(examples, start=1):
        missing_fields = required_fields.difference(example)

        if missing_fields:
            raise ValueError(
                f"Evaluation example {example_index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        prompts.append(
            create_training_prompt(
                grade=example["grade"],
                subject=str(example["subject"]),
                state=str(example["state"]),
                current_content=str(example["current_content"]),
            )
        )

    return prompts


def determine_device() -> torch.device:
    """Select the best available inference device."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def determine_model_dtype(device: torch.device) -> torch.dtype:
    """Choose a model dtype compatible with the inference device."""

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    return torch.float32


def load_tokenizer(
    adapter_path: Path,
    base_model_name: str,
) -> PreTrainedTokenizerBase:
    """Load the training tokenizer from the adapter or base model."""

    tokenizer_source: str | Path = base_model_name

    if (adapter_path / "tokenizer_config.json").is_file():
        tokenizer_source = adapter_path

    LOGGER.info("Loading tokenizer from %s", tokenizer_source)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_source),
            use_fast=True,
            trust_remote_code=False,
        )
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Unable to load tokenizer from {tokenizer_source}."
        ) from error

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "The tokenizer has neither a pad token nor an EOS token."
            )
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    return tokenizer


def load_base_model(
    base_model_name: str,
    device: torch.device,
) -> PreTrainedModel:
    """Load the original base causal language model."""

    model_dtype = determine_model_dtype(device)

    LOGGER.info(
        "Loading base model %s on %s using %s",
        base_model_name,
        device,
        model_dtype,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=model_dtype,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            f"Unable to load base model: {base_model_name}"
        ) from error

    try:
        model.to(device)
    except RuntimeError as error:
        raise RuntimeError(
            f"Unable to move the base model to device {device}."
        ) from error

    model.eval()
    return model


def load_fine_tuned_model(
    base_model_name: str,
    adapter_path: Path,
    device: torch.device,
) -> PeftModel:
    """Load the base model with the trained LoRA adapter attached."""

    base_model = load_base_model(
        base_model_name=base_model_name,
        device=device,
    )

    LOGGER.info("Loading LoRA adapter from %s", adapter_path)

    try:
        fine_tuned_model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            is_trainable=False,
        )
    except (OSError, ValueError, RuntimeError) as error:
        release_model(base_model)
        raise RuntimeError(
            f"Unable to load LoRA adapter from {adapter_path}."
        ) from error

    fine_tuned_model.eval()
    return fine_tuned_model


def format_prompt_for_model(
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    """Apply the same chat structure used during training."""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return (
        f"System: {SYSTEM_PROMPT}\n\n"
        f"User: {prompt}\n\n"
        "Assistant:"
    )


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    """Generate one deterministic response for a training-style prompt."""

    formatted_prompt = format_prompt_for_model(
        prompt=prompt,
        tokenizer=tokenizer,
    )

    try:
        encoded_prompt = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("Failed to tokenize an evaluation prompt.") from error

    encoded_prompt = {
        key: tensor.to(device)
        for key, tensor in encoded_prompt.items()
    }
    prompt_token_count = encoded_prompt["input_ids"].shape[1]

    try:
        with torch.inference_mode():
            generated_token_ids = model.generate(
                **encoded_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    except RuntimeError as error:
        raise RuntimeError(
            "Model inference failed. The selected device may not have enough "
            "available memory."
        ) from error

    response_token_ids = generated_token_ids[
        0,
        prompt_token_count:,
    ]
    response_text = tokenizer.decode(
        response_token_ids,
        skip_special_tokens=True,
    ).strip()

    if not response_text:
        return "[No response generated]"

    return response_text


def run_inference_for_prompts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    device: torch.device,
    max_new_tokens: int,
    model_label: str,
) -> list[str]:
    """Run inference for all prompts using one model configuration."""

    responses: list[str] = []

    for prompt_index, prompt in enumerate(prompts, start=1):
        LOGGER.info(
            "Generating %s response %d of %d",
            model_label,
            prompt_index,
            len(prompts),
        )

        try:
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=max_new_tokens,
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"Failed to generate {model_label} response for evaluation "
                f"prompt {prompt_index}."
            ) from error

        responses.append(response)

    return responses


def release_model(model: PreTrainedModel) -> None:
    """Release model memory before loading another model."""

    try:
        model.to("cpu")
    except RuntimeError:
        LOGGER.warning("Unable to move model to CPU during cleanup.")

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def create_comparison_rows(
    prompts: Sequence[str],
    base_model_responses: Sequence[str],
    fine_tuned_model_responses: Sequence[str],
) -> list[dict[str, str]]:
    """Combine prompts and model responses into CSV-ready rows."""

    if not (
        len(prompts)
        == len(base_model_responses)
        == len(fine_tuned_model_responses)
    ):
        raise ValueError(
            "The prompt and response collections must have equal lengths."
        )

    return [
        {
            "Prompt": prompt,
            "Base Model Response": base_response,
            "Fine-Tuned Model Response": fine_tuned_response,
        }
        for prompt, base_response, fine_tuned_response in zip(
            prompts,
            base_model_responses,
            fine_tuned_model_responses,
        )
    ]


def save_comparison_results(
    comparison_rows: Sequence[dict[str, str]],
    output_path: Path,
) -> None:
    """Save model comparison results to the required CSV path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output_path = output_path.with_suffix(
        f"{output_path.suffix}.tmp"
    )

    try:
        with temporary_output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=CSV_COLUMNS,
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(comparison_rows)

        temporary_output_path.replace(output_path)
    except (OSError, csv.Error, ValueError) as error:
        temporary_output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Unable to save comparison CSV to {output_path}."
        ) from error


def extract_prompt_summary(prompt: str) -> str:
    """Extract the identifying prompt fields for terminal output."""

    summary_lines = [
        line
        for line in prompt.splitlines()
        if line.startswith(
            ("Grade:", "Subject:", "State:", "Current content:")
        )
    ]
    return "\n".join(summary_lines)


def print_comparison_summary(
    comparison_rows: Sequence[dict[str, str]],
    base_model_name: str,
    adapter_path: Path,
    output_path: Path,
) -> None:
    """Print each before-and-after comparison to the terminal."""

    major_separator = "=" * 100
    minor_separator = "-" * 100

    print(f"\n{major_separator}")
    print("CONTEXT-TO-LEARNING MODEL COMPARISON")
    print(major_separator)
    print(f"Base model: {base_model_name}")
    print(f"LoRA adapter: {adapter_path}")
    print(f"Evaluation prompts: {len(comparison_rows)}")
    print(f"CSV output: {output_path}")
    print(major_separator)

    for comparison_index, row in enumerate(
        comparison_rows,
        start=1,
    ):
        print(f"\nCOMPARISON {comparison_index}")
        print(minor_separator)
        print(extract_prompt_summary(row["Prompt"]))
        print("\nBASE MODEL RESPONSE")
        print(minor_separator)
        print(row["Base Model Response"])
        print("\nFINE-TUNED MODEL RESPONSE")
        print(minor_separator)
        print(row["Fine-Tuned Model Response"])
        print(major_separator)

    print(
        f"\nEvaluation complete. Full comparison results were saved to "
        f"{output_path}."
    )


def main() -> None:
    """Run the complete base-versus-LoRA evaluation workflow."""

    configure_logging()
    arguments = parse_arguments()
    validate_arguments(arguments)
    set_seed(arguments.seed)

    base_model_name = resolve_base_model_name(
        adapter_path=arguments.adapter_path,
        model_name_override=arguments.base_model_name,
    )
    device = determine_device()
    evaluation_prompts = build_evaluation_prompts(
        EVALUATION_EXAMPLES
    )
    tokenizer = load_tokenizer(
        adapter_path=arguments.adapter_path,
        base_model_name=base_model_name,
    )

    LOGGER.info("Using inference device: %s", device)

    base_model = load_base_model(
        base_model_name=base_model_name,
        device=device,
    )
    base_model_responses = run_inference_for_prompts(
        model=base_model,
        tokenizer=tokenizer,
        prompts=evaluation_prompts,
        device=device,
        max_new_tokens=arguments.max_new_tokens,
        model_label="base model",
    )
    release_model(base_model)

    fine_tuned_model = load_fine_tuned_model(
        base_model_name=base_model_name,
        adapter_path=arguments.adapter_path,
        device=device,
    )
    fine_tuned_model_responses = run_inference_for_prompts(
        model=fine_tuned_model,
        tokenizer=tokenizer,
        prompts=evaluation_prompts,
        device=device,
        max_new_tokens=arguments.max_new_tokens,
        model_label="fine-tuned model",
    )
    release_model(fine_tuned_model)

    comparison_rows = create_comparison_rows(
        prompts=evaluation_prompts,
        base_model_responses=base_model_responses,
        fine_tuned_model_responses=fine_tuned_model_responses,
    )
    save_comparison_results(
        comparison_rows=comparison_rows,
        output_path=arguments.output_path,
    )
    print_comparison_summary(
        comparison_rows=comparison_rows,
        base_model_name=base_model_name,
        adapter_path=arguments.adapter_path,
        output_path=arguments.output_path,
    )


if __name__ == "__main__":
    main()