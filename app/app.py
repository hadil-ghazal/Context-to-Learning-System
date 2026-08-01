# AI Assistance Attribution:
# This file was generated with assistance from OpenAI ChatGPT:
# https://chatgpt.com/share/6a6d47a0-1a54-83e8-91a4-2f8c8b673fe5

# COde enhancements made by Hadil Ghazal 7/31/26 to:

# Added preset exmaples to use in the demo
# Replaced entire render_input_form() function for ex
# Removed the empty input check in main()

"""Streamlit application for the fine-tuned Context-to-Learning system."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import streamlit as st
import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


ADAPTER_PATH: Final[Path] = Path("models/context-learning-lora")
DEFAULT_BASE_MODEL_NAME: Final[str] = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS: Final[int] = 384

SYSTEM_PROMPT: Final[str] = (
    "You create accurate, age-appropriate, curriculum-aligned learning "
    "opportunities from the content a child is viewing."
)

SUBJECT_OPTIONS: Final[tuple[str, ...]] = (
    "Mathematics",
    "Science",
    "English Language Arts",
    "Social Studies",
)

STATE_OPTIONS: Final[tuple[str, ...]] = (
    "California",
    "Florida",
    "New York",
    "North Carolina",
    "Texas",
)


PRESET_EXAMPLES: Final[dict[str, dict[str, str | int]]] = {
    "Minecraft House — Area and Multiplication": {
        "grade": 5,
        "subject": "Mathematics",
        "state": "California",
        "current_content": (
            "A Minecraft tutorial demonstrates building a rectangular house "
            "with repeated block patterns and limited materials."
        ),
    },
    "Cooking Video — Fractions and Proportions": {
        "grade": 5,
        "subject": "Mathematics",
        "state": "California",
        "current_content": (
            "A cooking video demonstrates doubling a pancake recipe while "
            "measuring flour, milk, eggs, and cooking time."
        ),
    },
    "Basketball — Percentages and Data": {
        "grade": 6,
        "subject": "Mathematics",
        "state": "California",
        "current_content": (
            "A basketball analysis video compares shooting percentages from "
            "different areas of the court."
        ),
    },
    "Balloon-Powered Car — Forces and Motion": {
        "grade": 5,
        "subject": "Science",
        "state": "California",
        "current_content": (
            "A science channel demonstrates a balloon-powered car and compares "
            "how far it travels under different conditions."
        ),
    },
    "Octopus Documentary — Animal Adaptations": {
        "grade": 6,
        "subject": "Science",
        "state": "California",
        "current_content": (
            "A video explains how octopuses use camouflage, flexible bodies, "
            "and problem-solving behaviors to survive."
        ),
    },
    "Travel Vlog — Geography": {
        "grade": 5,
        "subject": "Social Studies",
        "state": "California",
        "current_content": (
            "A travel vlog follows a train trip through several cities and "
            "shows maps, travel times, landmarks, and local transportation."
        ),
    },
}


RESPONSE_SECTION_LABELS: Final[dict[str, str]] = {
    "educational_concept": "Educational Concept",
    "grade_appropriate_explanation": (
        "Grade-Level Appropriate Explanation"
    ),
    "curriculum_aligned_learning_connection": (
        "Curriculum-Aligned Learning Connection"
    ),
    "practice_question": "Practice Question",
    "answer": "Answer",
}

SECTION_HEADING_PATTERNS: Final[
    dict[str, tuple[str, ...]]
] = {
    "educational_concept": (
        "educational concept",
        "relevant educational concept",
        "concept",
    ),
    "grade_appropriate_explanation": (
        "grade-appropriate explanation",
        "grade appropriate explanation",
        "grade-level appropriate explanation",
        "grade level appropriate explanation",
        "explanation",
    ),
    "curriculum_aligned_learning_connection": (
        "curriculum-aligned learning connection",
        "curriculum aligned learning connection",
        "learning connection",
        "curriculum connection",
    ),
    "practice_question": (
        "practice question",
        "question",
    ),
    "answer": (
        "answer",
    ),
}


def configure_page() -> None:
    """Configure the Streamlit page and application header."""

    st.set_page_config(
        page_title="Context-to-Learning System",
        page_icon="📚",
        layout="centered",
    )

    st.title("Context-to-Learning System")
    st.write(
        "Turn the content a student is currently viewing into an "
        "age-appropriate, curriculum-aligned learning opportunity."
    )


def validate_adapter_directory(adapter_path: Path) -> None:
    """Validate that the required LoRA adapter files are available."""

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"The trained model directory was not found at "
            f"'{adapter_path}'."
        )

    if not adapter_path.is_dir():
        raise NotADirectoryError(
            f"The trained model path is not a directory: "
            f"'{adapter_path}'."
        )

    adapter_configuration_path = adapter_path / "adapter_config.json"

    if not adapter_configuration_path.is_file():
        raise FileNotFoundError(
            "The LoRA adapter configuration is missing at "
            f"'{adapter_configuration_path}'."
        )

    supported_adapter_weight_files = (
        adapter_path / "adapter_model.safetensors",
        adapter_path / "adapter_model.bin",
    )

    if not any(
        model_file.is_file()
        for model_file in supported_adapter_weight_files
    ):
        raise FileNotFoundError(
            "The LoRA adapter weights are missing from "
            f"'{adapter_path}'."
        )


def resolve_base_model_name(adapter_path: Path) -> str:
    """Read the training base model name from the LoRA configuration."""

    try:
        adapter_configuration = PeftConfig.from_pretrained(
            str(adapter_path)
        )
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "The LoRA adapter configuration could not be loaded."
        ) from error

    configured_model_name = (
        adapter_configuration.base_model_name_or_path
    )

    if configured_model_name:
        return configured_model_name

    return DEFAULT_BASE_MODEL_NAME


def determine_inference_device() -> torch.device:
    """Select the best available device for model inference."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def determine_model_dtype(device: torch.device) -> torch.dtype:
    """Choose a model data type compatible with the inference device."""

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16

        return torch.float16

    return torch.float32


def load_application_tokenizer(
    adapter_path: Path,
    base_model_name: str,
) -> PreTrainedTokenizerBase:
    """Load the tokenizer saved during training or the base tokenizer."""

    tokenizer_source: str | Path = base_model_name

    if (adapter_path / "tokenizer_config.json").is_file():
        tokenizer_source = adapter_path

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_source),
            use_fast=True,
            trust_remote_code=False,
        )
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"The tokenizer could not be loaded from "
            f"'{tokenizer_source}'."
        ) from error

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "The tokenizer does not define a padding token or an "
                "end-of-sequence token."
            )

        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    return tokenizer


@st.cache_resource(show_spinner=False)
def load_fine_tuned_model_and_tokenizer() -> tuple[
    PeftModel,
    PreTrainedTokenizerBase,
    torch.device,
]:
    """Load and cache the tokenizer and LoRA fine-tuned model."""

    validate_adapter_directory(ADAPTER_PATH)

    base_model_name = resolve_base_model_name(ADAPTER_PATH)
    inference_device = determine_inference_device()
    model_dtype = determine_model_dtype(inference_device)

    tokenizer = load_application_tokenizer(
        adapter_path=ADAPTER_PATH,
        base_model_name=base_model_name,
    )

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=model_dtype,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            f"The base model '{base_model_name}' could not be loaded."
        ) from error

    try:
        base_model.to(inference_device)

        fine_tuned_model = PeftModel.from_pretrained(
            base_model,
            str(ADAPTER_PATH),
            is_trainable=False,
        )
        fine_tuned_model.to(inference_device)
        fine_tuned_model.eval()
    except (OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            "The trained LoRA adapter could not be attached to the "
            "base model."
        ) from error

    return fine_tuned_model, tokenizer, inference_device


def build_training_prompt(
    grade: int,
    subject: str,
    state: str,
    current_content: str,
) -> str:
    """Build the same user prompt structure used during training."""

    return (
        "Transform the child's current content into a concise, "
        "age-appropriate, curriculum-aligned learning opportunity.\n\n"
        f"Grade: {grade}\n"
        f"Subject: {subject}\n"
        f"State: {state}\n"
        f"Current content: {current_content.strip()}\n\n"
        "Include the relevant educational concept, a grade-appropriate "
        "explanation, a curriculum-aligned learning connection, one practice "
        "question, and its answer."
    )


def format_prompt_for_model(
    training_prompt: str,
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    """Apply the same chat-message structure used during training."""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": training_prompt,
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
        f"User: {training_prompt}\n\n"
        "Assistant:"
    )


def generate_learning_opportunity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    training_prompt: str,
) -> str:
    """Generate a learning opportunity with the fine-tuned model."""

    formatted_prompt = format_prompt_for_model(
        training_prompt=training_prompt,
        tokenizer=tokenizer,
    )

    try:
        encoded_prompt = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "The learning request could not be tokenized."
        ) from error

    encoded_prompt = {
        input_name: input_tensor.to(device)
        for input_name, input_tensor in encoded_prompt.items()
    }
    prompt_token_count = encoded_prompt["input_ids"].shape[1]

    try:
        with torch.inference_mode():
            generated_token_ids = model.generate(
                **encoded_prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    except RuntimeError as error:
        raise RuntimeError(
            "The model could not generate a response. The application "
            "may not have enough available memory."
        ) from error

    response_token_ids = generated_token_ids[
        0,
        prompt_token_count:,
    ]
    generated_response = tokenizer.decode(
        response_token_ids,
        skip_special_tokens=True,
    ).strip()

    if not generated_response:
        raise RuntimeError(
            "The model completed inference but did not generate a response."
        )

    return generated_response


def normalize_heading(heading: str) -> str:
    """Normalize a response heading for safe section matching."""

    normalized_heading = heading.strip().lower()
    normalized_heading = normalized_heading.replace("_", " ")
    normalized_heading = normalized_heading.replace("*", "")
    normalized_heading = normalized_heading.replace("#", "")
    normalized_heading = re.sub(r"\s+", " ", normalized_heading)
    return normalized_heading.strip(" :-")


def identify_section_key(heading: str) -> str | None:
    """Map a generated heading to a known response section."""

    normalized_heading = normalize_heading(heading)

    for section_key, heading_patterns in (
        SECTION_HEADING_PATTERNS.items()
    ):
        if normalized_heading in heading_patterns:
            return section_key

    return None


def parse_generated_response(
    generated_response: str,
) -> dict[str, str] | None:
    """Parse labeled response sections without raising parsing errors."""

    parsed_sections: dict[str, str] = {}
    active_section_key: str | None = None
    active_section_lines: list[str] = []

    def save_active_section() -> None:
        """Save the currently accumulated response section."""

        if active_section_key is None:
            return

        section_content = "\n".join(active_section_lines).strip()

        if section_content:
            parsed_sections[active_section_key] = section_content

    for response_line in generated_response.splitlines():
        stripped_line = response_line.strip()

        if not stripped_line:
            if active_section_key and active_section_lines:
                active_section_lines.append("")
            continue

        heading_match = re.match(
            r"^\s*(?:[-*]\s*)?(?:\*\*)?"
            r"([^:\n]+?)(?:\*\*)?\s*:\s*(.*)$",
            stripped_line,
        )

        if heading_match:
            possible_heading = heading_match.group(1)
            inline_content = heading_match.group(2).strip()
            detected_section_key = identify_section_key(
                possible_heading
            )

            if detected_section_key:
                save_active_section()
                active_section_key = detected_section_key
                active_section_lines = []

                if inline_content:
                    active_section_lines.append(inline_content)

                continue

        markdown_heading_match = re.match(
            r"^\s*#{1,6}\s+(.+?)\s*$",
            stripped_line,
        )

        if markdown_heading_match:
            detected_section_key = identify_section_key(
                markdown_heading_match.group(1)
            )

            if detected_section_key:
                save_active_section()
                active_section_key = detected_section_key
                active_section_lines = []
                continue

        if active_section_key:
            active_section_lines.append(stripped_line)

    save_active_section()

    required_section_keys = set(RESPONSE_SECTION_LABELS)
    parsed_section_keys = set(parsed_sections)

    if not required_section_keys.issubset(parsed_section_keys):
        return None

    return parsed_sections


def display_structured_response(
    parsed_response: dict[str, str],
) -> None:
    """Display each parsed learning-opportunity section."""

    for section_key, display_label in (
        RESPONSE_SECTION_LABELS.items()
    ):
        st.subheader(display_label)
        st.write(parsed_response[section_key])


def display_generated_response(generated_response: str) -> None:
    """Display a parsed response or safely fall back to the full output."""

    parsed_response = parse_generated_response(generated_response)

    st.divider()
    st.header("Generated Learning Opportunity")

    if parsed_response is None:
        st.info(
            "The response could not be separated into every expected "
            "section, so the complete model response is shown below."
        )
        st.write(generated_response)
        return

    display_structured_response(parsed_response)


def render_input_form() -> tuple[int, str, str, str, bool]:
    """renderpreset content options and retrun the selected example."""

    with st.form("learning_opportunity_form"):
        selected_example_name = st.selectbox(
            "Choose Example ofcurrent student content",
            options=list(PRESET_EXAMPLES),
        )

        selected_example = PRESET_EXAMPLES[selected_example_name]

        st.info(str(selected_example["current_content"]))

        st.caption(
            f"Grade {selected_example['grade']} · "
            f"{selected_example['subject']} · "
            f"{selected_example['state']}"
        )

        submitted = st.form_submit_button(
            "Generate Learning opportunity",
            type="primary",
            use_container_width=True,
        )

    return (
        int(selected_example["grade"]),
        str(selected_example["subject"]),
        str(selected_example["state"]),
        str(selected_example["current_content"]),
        submitted,
    )


def main() -> None:
    """Run the Context-to-Learning Streamlit application."""

    configure_page()

    grade, subject, state, current_content, submitted = (
        render_input_form()
    )

    if not submitted:
        return

    #if not current_content.strip():
        #st.warning(
            #"Please enter a summary of the content the student is currently "
            #"consuming."
        #)
        #return

    try:
        with st.spinner("Loading the fine-tuned learning model..."):
            model, tokenizer, device = (
                load_fine_tuned_model_and_tokenizer()
            )
    except (
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        ValueError,
    ) as error:
        st.error(
            "The fine-tuned model could not be loaded. Confirm that the "
            "trained LoRA adapter exists in "
            "`models/context-learning-lora`."
        )
        st.caption(str(error))
        return

    training_prompt = build_training_prompt(
        grade=grade,
        subject=subject,
        state=state,
        current_content=current_content,
    )

    try:
        with st.spinner("Creating the learning opportunity..."):
            generated_response = generate_learning_opportunity(
                model=model,
                tokenizer=tokenizer,
                device=device,
                training_prompt=training_prompt,
            )
    except RuntimeError as error:
        st.error(
            "The model was loaded, but the learning opportunity could not "
            "be generated. Please try again with a shorter content summary."
        )
        st.caption(str(error))
        return

    display_generated_response(generated_response)


if __name__ == "__main__":
    main()