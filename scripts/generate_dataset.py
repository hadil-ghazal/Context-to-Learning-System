# AI Assistance Attribution:
# This file was generated with assistance from OpenAI ChatGPT:
# https://chatgpt.com/share/6a6d47a0-1a54-83e8-91a4-2f8c8b673fe5

"""Generate a synthetic context-to-learning supervised fine-tuning dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Final, Sequence


DEFAULT_OUTPUT_PATH: Final[Path] = Path(
    "data/raw/context_learning_dataset.jsonl"
)
DEFAULT_DATASET_SIZE: Final[int] = 1_000
DEFAULT_RANDOM_SEED: Final[int] = 42

GRADE_LEVELS: Final[tuple[int, ...]] = tuple(range(1, 13))

STATE_STANDARDS: Final[dict[str, str]] = {
    "California": "California Common Core State Standards",
    "Florida": "Florida B.E.S.T. Standards",
    "New York": "New York State Next Generation Learning Standards",
    "North Carolina": "North Carolina Standard Course of Study",
    "Texas": "Texas Essential Knowledge and Skills",
}

SUBJECT_CONCEPTS: Final[dict[str, tuple[dict[str, Any], ...]]] = {
    "Mathematics": (
        {
            "concept": "addition and subtraction",
            "grades": range(1, 4),
            "explanation": (
                "Addition combines quantities, while subtraction finds how "
                "many remain or how much separates two quantities."
            ),
            "connection": (
                "Use the quantities shown in the content to model a real-world "
                "addition or subtraction problem."
            ),
            "question_template": (
                "{first} items are shown, and {second} more are added. "
                "How many items are there altogether?"
            ),
            "answer_function": lambda first, second: first + second,
            "number_range": (2, 20),
        },
        {
            "concept": "multiplication and equal groups",
            "grades": range(3, 6),
            "explanation": (
                "Multiplication represents equal-sized groups and provides a "
                "faster way to calculate repeated addition."
            ),
            "connection": (
                "Organize repeated objects, players, ingredients, or resources "
                "from the content into equal groups."
            ),
            "question_template": (
                "There are {first} groups with {second} items in each group. "
                "How many items are there in total?"
            ),
            "answer_function": lambda first, second: first * second,
            "number_range": (2, 12),
        },
        {
            "concept": "fractions and proportional reasoning",
            "grades": range(4, 8),
            "explanation": (
                "A fraction represents part of a whole, and proportional "
                "reasoning compares quantities that change at the same rate."
            ),
            "connection": (
                "Interpret portions, completion levels, recipe amounts, maps, "
                "or statistics shown in the content as fractions or ratios."
            ),
            "question_template": (
                "A player completed {first} out of {second} challenges. "
                "What fraction of the challenges was completed?"
            ),
            "answer_function": lambda first, second: f"{first}/{second}",
            "number_range": (2, 10),
            "ordered_numbers": True,
        },
        {
            "concept": "linear relationships",
            "grades": range(7, 10),
            "explanation": (
                "A linear relationship has a constant rate of change, meaning "
                "one quantity changes by the same amount for each unit change "
                "in another quantity."
            ),
            "connection": (
                "Model changing scores, distance, costs, resources, or time "
                "from the content with a linear equation."
            ),
            "question_template": (
                "A score begins at {first} points and increases by {second} "
                "points per round. What is the score after 5 rounds?"
            ),
            "answer_function": lambda first, second: first + (5 * second),
            "number_range": (2, 25),
        },
        {
            "concept": "probability",
            "grades": range(6, 13),
            "explanation": (
                "Probability measures how likely an event is by comparing the "
                "number of favorable outcomes with the total possible outcomes."
            ),
            "connection": (
                "Analyze random drops, game outcomes, sports attempts, weather "
                "events, or experimental results shown in the content."
            ),
            "question_template": (
                "There are {second} equally likely outcomes, and {first} are "
                "favorable. What is the probability of a favorable outcome?"
            ),
            "answer_function": lambda first, second: f"{first}/{second}",
            "number_range": (2, 12),
            "ordered_numbers": True,
        },
    ),
    "Science": (
        {
            "concept": "forces and motion",
            "grades": range(2, 9),
            "explanation": (
                "A force is a push or pull that can change an object's speed, "
                "direction, or position."
            ),
            "connection": (
                "Examine how characters, vehicles, balls, tools, or animals "
                "move and identify the forces affecting them."
            ),
            "question_template": (
                "A moving object speeds up after being pushed. What effect did "
                "the push have on the object's motion?"
            ),
            "fixed_answer": "The push increased the object's speed.",
        },
        {
            "concept": "ecosystem interactions",
            "grades": range(3, 9),
            "explanation": (
                "Organisms in an ecosystem depend on living and nonliving "
                "parts of their environment for energy, shelter, and survival."
            ),
            "connection": (
                "Identify producers, consumers, habitats, food sources, or "
                "environmental changes visible in the content."
            ),
            "question_template": (
                "What is one resource an animal in the scene may need from its "
                "habitat to survive?"
            ),
            "fixed_answer": (
                "A valid answer is food, water, shelter, space, or oxygen."
            ),
        },
        {
            "concept": "energy transfer",
            "grades": range(4, 11),
            "explanation": (
                "Energy can move between objects or change form, such as from "
                "chemical energy into heat, light, sound, or motion."
            ),
            "connection": (
                "Trace energy through cooking, machines, exercise, sunlight, "
                "electric devices, or moving objects in the content."
            ),
            "question_template": (
                "When a stove heats a pan, what form of energy is transferred "
                "to the pan?"
            ),
            "fixed_answer": "Thermal energy is transferred to the pan.",
        },
        {
            "concept": "adaptation and natural selection",
            "grades": range(6, 13),
            "explanation": (
                "Adaptations are inherited traits that can improve an "
                "organism's ability to survive and reproduce in an environment."
            ),
            "connection": (
                "Use an organism's body structures or behaviors in the content "
                "to explain how it is suited to its environment."
            ),
            "question_template": (
                "How could camouflage improve an animal's chance of survival?"
            ),
            "fixed_answer": (
                "Camouflage can help the animal avoid predators or approach "
                "prey without being detected."
            ),
        },
    ),
    "English Language Arts": (
        {
            "concept": "main idea and supporting details",
            "grades": range(1, 7),
            "explanation": (
                "The main idea is the most important point, while supporting "
                "details explain, prove, or develop that point."
            ),
            "connection": (
                "Summarize the central message of the content and identify a "
                "specific detail that supports it."
            ),
            "question_template": (
                "What is one detail from the content that would help support "
                "its main idea?"
            ),
            "fixed_answer": (
                "A correct answer should name a specific event, fact, example, "
                "or description from the content."
            ),
        },
        {
            "concept": "sequence and cause and effect",
            "grades": range(2, 8),
            "explanation": (
                "Sequence describes the order of events, while cause and "
                "effect explains why something happened and what followed."
            ),
            "connection": (
                "Track the steps, decisions, and results shown in gameplay, "
                "instructions, demonstrations, stories, or experiments."
            ),
            "question_template": (
                "What happened as a result of one action described in the "
                "content?"
            ),
            "fixed_answer": (
                "A correct answer should identify an action and its resulting "
                "event using evidence from the content."
            ),
        },
        {
            "concept": "author's purpose and point of view",
            "grades": range(5, 11),
            "explanation": (
                "Author's purpose is the reason content was created, and point "
                "of view reflects the creator's perspective or position."
            ),
            "connection": (
                "Determine whether the creator is informing, explaining, "
                "entertaining, reviewing, or persuading the audience."
            ),
            "question_template": (
                "What is the creator's likely purpose, and what evidence "
                "supports your answer?"
            ),
            "fixed_answer": (
                "The answer should identify a purpose and cite a feature such "
                "as facts, instructions, opinions, humor, or persuasive claims."
            ),
        },
        {
            "concept": "argument and evidence",
            "grades": range(6, 13),
            "explanation": (
                "A strong argument presents a clear claim supported by "
                "relevant evidence and logical reasoning."
            ),
            "connection": (
                "Evaluate recommendations, rankings, commentary, or opinions "
                "expressed in the content."
            ),
            "question_template": (
                "What evidence would make one claim in the content more "
                "convincing?"
            ),
            "fixed_answer": (
                "Relevant data, examples, expert testimony, or direct "
                "observations would strengthen the claim."
            ),
        },
    ),
    "Social Studies": (
        {
            "concept": "geography and human-environment interaction",
            "grades": range(2, 9),
            "explanation": (
                "Geography examines places and how people interact with their "
                "physical and human environments."
            ),
            "connection": (
                "Analyze maps, landscapes, transportation, settlements, "
                "resources, or environmental choices visible in the content."
            ),
            "question_template": (
                "How might one geographic feature shown or described affect "
                "where people live or travel?"
            ),
            "fixed_answer": (
                "A geographic feature can influence access to water, food, "
                "transportation, safety, resources, or suitable land."
            ),
        },
        {
            "concept": "economic decision-making",
            "grades": range(3, 11),
            "explanation": (
                "Economic decisions involve choosing how to use limited "
                "resources to satisfy needs and wants."
            ),
            "connection": (
                "Examine choices involving virtual currency, supplies, trades, "
                "budgets, prices, labor, or resource scarcity."
            ),
            "question_template": (
                "Why does scarcity require people or players to make choices?"
            ),
            "fixed_answer": (
                "Scarcity means resources are limited, so not every want can "
                "be satisfied at the same time."
            ),
        },
        {
            "concept": "civic participation and rules",
            "grades": range(3, 10),
            "explanation": (
                "Communities use rules, responsibilities, and decision-making "
                "processes to promote safety, fairness, and cooperation."
            ),
            "connection": (
                "Connect rules, teamwork, moderation, leadership, or conflict "
                "resolution in the content to civic life."
            ),
            "question_template": (
                "What is one reason a community or online game needs shared "
                "rules?"
            ),
            "fixed_answer": (
                "Shared rules can promote fairness, safety, cooperation, and "
                "predictable consequences."
            ),
        },
    ),
}

CONTENT_TEMPLATES: Final[tuple[dict[str, Any], ...]] = (
    {
        "category": "Fortnite gameplay",
        "subjects": (
            "Mathematics",
            "Science",
            "English Language Arts",
            "Social Studies",
        ),
        "summaries": (
            "A Fortnite creator compares two landing locations by counting "
            "loot chests, estimating travel distance, and discussing risk.",
            "A player explains how storm-circle timing affects movement across "
            "the map while tracking health, materials, and remaining players.",
            "A squad reviews a match replay and explains how communication and "
            "resource-sharing influenced the final outcome.",
        ),
    },
    {
        "category": "Minecraft gameplay",
        "subjects": (
            "Mathematics",
            "Science",
            "English Language Arts",
            "Social Studies",
        ),
        "summaries": (
            "A Minecraft tutorial demonstrates building a rectangular house "
            "with repeated block patterns and limited materials.",
            "A survival-mode video shows a player farming crops, caring for "
            "animals, and organizing food and tools.",
            "A redstone tutorial explains how switches, circuits, and powered "
            "components operate an automatic door.",
        ),
    },
    {
        "category": "cooking video",
        "subjects": ("Mathematics", "Science", "English Language Arts"),
        "summaries": (
            "A cooking video demonstrates doubling a pancake recipe while "
            "measuring flour, milk, eggs, and cooking time.",
            "A baker explains how yeast causes bread dough to rise and shows "
            "the dough before and after proofing.",
            "A short recipe video presents the steps for preparing vegetable "
            "soup and explains when each ingredient is added.",
        ),
    },
    {
        "category": "sports clip",
        "subjects": (
            "Mathematics",
            "Science",
            "English Language Arts",
            "Social Studies",
        ),
        "summaries": (
            "A basketball analysis video compares shooting percentages from "
            "different areas of the court.",
            "A soccer replay shows a player accelerating, changing direction, "
            "and passing the ball before a goal.",
            "A commentator explains how teamwork and strategic substitutions "
            "changed the momentum of a close game.",
        ),
    },
    {
        "category": "animal documentary",
        "subjects": (
            "Science",
            "English Language Arts",
            "Social Studies",
        ),
        "summaries": (
            "A wildlife documentary follows polar bears searching for food as "
            "seasonal sea ice changes.",
            "A video explains how octopuses use camouflage, flexible bodies, "
            "and problem-solving behaviors to survive.",
            "A documentary shows a bee colony collecting nectar, protecting "
            "the hive, and supporting plant pollination.",
        ),
    },
    {
        "category": "science video",
        "subjects": ("Mathematics", "Science", "English Language Arts"),
        "summaries": (
            "A science channel demonstrates a balloon-powered car and compares "
            "how far it travels under different conditions.",
            "A video explains the water cycle using diagrams of evaporation, "
            "condensation, precipitation, and collection.",
            "An experiment video tests which materials conduct heat most "
            "effectively and records the observations in a table.",
        ),
    },
    {
        "category": "technology tutorial",
        "subjects": (
            "Mathematics",
            "Science",
            "English Language Arts",
            "Social Studies",
        ),
        "summaries": (
            "A beginner coding tutorial uses a loop to move a game character "
            "through a repeated sequence of actions.",
            "A video explains how an algorithm recommends videos based on "
            "watch history, clicks, and viewing time.",
            "A robotics tutorial shows sensors detecting obstacles and sending "
            "signals that change a robot's direction.",
        ),
    },
    {
        "category": "travel vlog",
        "subjects": (
            "Mathematics",
            "English Language Arts",
            "Social Studies",
        ),
        "summaries": (
            "A travel vlog follows a train trip through several cities and "
            "shows maps, travel times, landmarks, and local transportation.",
            "A creator visits a coastal community and discusses tourism, local "
            "jobs, weather, and environmental conservation.",
            "A family compares food, architecture, language, and daily routines "
            "while visiting another country.",
        ),
    },
    {
        "category": "DIY tutorial",
        "subjects": ("Mathematics", "Science", "English Language Arts"),
        "summaries": (
            "A craft tutorial demonstrates measuring, cutting, and folding "
            "paper to construct a small geometric model.",
            "A home-repair video explains how levers and screws help assemble "
            "a piece of furniture.",
            "A creator gives step-by-step instructions for building a birdhouse "
            "from a limited set of wooden pieces.",
        ),
    },
    {
        "category": "music video",
        "subjects": ("Mathematics", "Science", "English Language Arts"),
        "summaries": (
            "A music lesson explains rhythm by counting repeated beats and "
            "combining whole, half, and quarter notes.",
            "A producer demonstrates how changing vibration frequency affects "
            "the pitch of a sound.",
            "A songwriter explains how imagery and repetition help communicate "
            "a song's central theme.",
        ),
    },
    {
        "category": "news explainer",
        "subjects": (
            "Mathematics",
            "Science",
            "English Language Arts",
            "Social Studies",
        ),
        "summaries": (
            "A student-friendly news explainer uses a chart to describe changes "
            "in local rainfall and water use.",
            "A short current-events video presents two viewpoints about a "
            "proposed community park.",
            "A news segment explains how a city budget divides funding among "
            "schools, roads, libraries, and emergency services.",
        ),
    },
    {
        "category": "online product review",
        "subjects": (
            "Mathematics",
            "Science",
            "English Language Arts",
            "Social Studies",
        ),
        "summaries": (
            "A reviewer compares two headphones using price, battery life, "
            "comfort, sound quality, and personal opinion.",
            "A creator tests several reusable water bottles and ranks them "
            "using insulation measurements and durability observations.",
            "A video recommends one laptop over another while mixing technical "
            "specifications, personal preferences, and sponsored claims.",
        ),
    },
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for dataset generation."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic context-to-learning JSONL dataset for "
            "supervised fine-tuning."
        )
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_DATASET_SIZE,
        help=f"Number of examples to generate. Default: {DEFAULT_DATASET_SIZE}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSONL path. Default: {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for reproducibility. Default: {DEFAULT_RANDOM_SEED}.",
    )
    return parser.parse_args()


def validate_dataset_size(dataset_size: int) -> None:
    """Validate that the requested dataset size is positive."""

    if dataset_size <= 0:
        raise ValueError("Dataset size must be greater than zero.")


def select_grade(rng: random.Random) -> int:
    """Select a grade level using a balanced random distribution."""

    return rng.choice(GRADE_LEVELS)


def select_content_template(
    rng: random.Random,
    subject: str,
) -> dict[str, Any]:
    """Select a content template compatible with the chosen subject."""

    compatible_templates = [
        template
        for template in CONTENT_TEMPLATES
        if subject in template["subjects"]
    ]
    return rng.choice(compatible_templates)


def select_subject_and_concept(
    rng: random.Random,
    grade: int,
) -> tuple[str, dict[str, Any]]:
    """Select a subject and grade-appropriate educational concept."""

    eligible_subjects: dict[str, list[dict[str, Any]]] = {}

    for subject, concepts in SUBJECT_CONCEPTS.items():
        eligible_concepts = [
            concept for concept in concepts if grade in concept["grades"]
        ]
        if eligible_concepts:
            eligible_subjects[subject] = eligible_concepts

    subject = rng.choice(tuple(eligible_subjects))
    concept = rng.choice(eligible_subjects[subject])
    return subject, concept


def generate_practice_question(
    rng: random.Random,
    concept: dict[str, Any],
) -> tuple[str, str]:
    """Generate a practice question and answer for an educational concept."""

    question_template = concept["question_template"]

    if "fixed_answer" in concept:
        return question_template, concept["fixed_answer"]

    minimum_value, maximum_value = concept["number_range"]
    first_number = rng.randint(minimum_value, maximum_value)
    second_number = rng.randint(minimum_value, maximum_value)

    if concept.get("ordered_numbers") and first_number > second_number:
        first_number, second_number = second_number, first_number

    question = question_template.format(
        first=first_number,
        second=second_number,
    )
    answer = str(
        concept["answer_function"](first_number, second_number)
    )
    return question, answer


def adapt_explanation_for_grade(explanation: str, grade: int) -> str:
    """Adjust explanation framing to match the learner's grade band."""

    if grade <= 2:
        prefix = "Think of it this way: "
    elif grade <= 5:
        prefix = "In this example, "
    elif grade <= 8:
        prefix = "The key idea is that "
    else:
        prefix = "A useful way to analyze this is that "

    return f"{prefix}{explanation[0].lower()}{explanation[1:]}"


def create_response(
    rng: random.Random,
    grade: int,
    state: str,
    concept: dict[str, Any],
) -> dict[str, str]:
    """Create a structured educational response for one dataset example."""

    practice_question, answer = generate_practice_question(rng, concept)
    standards_name = STATE_STANDARDS[state]

    return {
        "educational_concept": concept["concept"],
        "grade_appropriate_explanation": adapt_explanation_for_grade(
            concept["explanation"],
            grade,
        ),
        "curriculum_aligned_learning_connection": (
            f"For grade {grade}, this connects to the {standards_name} by "
            f"asking the learner to apply {concept['concept']} to an authentic "
            f"context. {concept['connection']}"
        ),
        "practice_question": practice_question,
        "answer": answer,
    }


def generate_example(
    rng: random.Random,
    example_index: int,
) -> dict[str, Any]:
    """Generate one synthetic context-to-learning training example."""

    grade = select_grade(rng)
    subject, concept = select_subject_and_concept(rng, grade)
    state = rng.choice(tuple(STATE_STANDARDS))
    content_template = select_content_template(rng, subject)
    current_content = rng.choice(content_template["summaries"])

    response = create_response(
        rng=rng,
        grade=grade,
        state=state,
        concept=concept,
    )

    return {
        "grade": grade,
        "subject": subject,
        "state": state,
        "current_content": current_content,
        "response": response,
        "metadata": {
            "example_id": f"context-learning-{example_index:06d}",
            "content_category": content_template["category"],
        },
    }


def generate_dataset(
    dataset_size: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Generate a reproducible synthetic dataset of the requested size."""

    validate_dataset_size(dataset_size)
    rng = random.Random(random_seed)

    return [
        generate_example(rng, example_index)
        for example_index in range(1, dataset_size + 1)
    ]


def validate_example(example: dict[str, Any]) -> None:
    """Validate the required schema for one generated example."""

    required_fields = {
        "grade",
        "subject",
        "state",
        "current_content",
        "response",
    }
    missing_fields = required_fields.difference(example)

    if missing_fields:
        raise ValueError(
            f"Generated example is missing fields: {sorted(missing_fields)}"
        )

    required_response_fields = {
        "educational_concept",
        "grade_appropriate_explanation",
        "curriculum_aligned_learning_connection",
        "practice_question",
        "answer",
    }
    missing_response_fields = required_response_fields.difference(
        example["response"]
    )

    if missing_response_fields:
        raise ValueError(
            "Generated response is missing fields: "
            f"{sorted(missing_response_fields)}"
        )


def write_jsonl(
    examples: Sequence[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write validated dataset examples to a UTF-8 JSONL file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")

    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for example in examples:
                validate_example(example)
                json.dump(
                    example,
                    output_file,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                output_file.write("\n")

        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    """Generate and save the synthetic context-to-learning dataset."""

    arguments = parse_arguments()
    dataset = generate_dataset(
        dataset_size=arguments.size,
        random_seed=arguments.seed,
    )
    write_jsonl(dataset, arguments.output)

    print(
        f"Generated {len(dataset):,} examples and saved them to "
        f"{arguments.output}."
    )


if __name__ == "__main__":
    main()