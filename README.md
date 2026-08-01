# Context-to-Learning-System
Context aware LLM for transforming everyday content into personalized learning opportunities

# Context-to-Learning-System

## Overview

Context-to-Learning-System is a deep learning project that finetunes an LLM to transform the content a student is currently consuming into an age appropriate, curriculum aligned learning opportunit y

Rather than requiring students to switch between entertainment and educational platforms, this project demonstrates how AI can bridge the two by generating educational explanations, curriculum connections, and practice questions from real world content

---

## Motivation

Students spend significant time consuming digital media with lhtings like videos, games, recipes, sports, documentaries, and online articles. These moments represent opportunities to reinforce classroom learning through contextualized instruction.

This project explores whether an LLM can learn a new capability through supervised fine tuning:

> Convert a summary of a student's current content into a curriculum aligned learning opportunity appropriate for the student's grade level
---

## Project Pipeline

1. Generate a synthetic supervised fine-tuning dataset.
2. Finetune the base LLM using LoRA.
3. Evaluate the fineutuned model against the original base model
4. Deploy the finetuneed model through a Streamlit application.

---

## Dataset

The project generates a synthetic JSONL dataset containing diverse educational scenarios.

Each example includes:

- Grade
- Subject
- State
- Current Content
- Target Response

Each target response contains:

- Educational Concept
- Grade-Appropriate Explanation
- Curriculum-Aligned Learning Connection
- Practice Question
- Answer

---

## Model

- Base Model: Qwen2.5-0.5B-Instruct
- Fine-Tuning Method: LoRA (PEFT)
- Framework: Hugging Face Transformers

---

## Evaluation

The evaluation script compares the original base model against the fine-tuned model using identical prompts and saves the results to:

```
data/outputs/model_comparison.csv
```

This demonstrates the new capability learned through fine tuning

---

## Application

The Streamlit application allows a user to select an example of content a student is consuming and generates a curriculum aligned learning opportunity using the fine tuned model

Outputs include:

- Educational Concept
- Grade-Appropriate Explanation
- Curriculum Aligned Learning Connection
- Practice Question
- Answer

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- PEFTLoRA)
- Datasets
- Streamlit

---

## Results

The fine tuned model was evaluated against the original base model using the identical prompts

Evaluation outputs are stored in and can be viewed here:

```text
data/outputs/model_comparison.csv
```

Across a;; the various representative educational scenarios, the fine tuned model consistently produced responses that were more structured and aligned with the project's objective by generating:

- An educational concept
- A grade / age aligned explanation
- the curriculum aligned learning connection
- practice questions
- An answer

on the other hand  the original base model produced more general instructional responses and did not consistently follow the desired output structure

## Future Work

Potential future enhancements include:

- Browser extension for automatic webpage summarization
- Realtime context detection from videos and webpages
- Support for additional curriculum standards
- Personalized learning based on student performance
- Gamification through rewards and progress tracking

---

## Disclaimer

The browser context used by the application is simulated through sampel examples rather than captured directly from a live browser. the idea is to build it as a popup rather than a free standing app