"""Core logic for Local Travel Guide Generator.

Milestone 1: Prepare small dataset of travel interests and corresponding local recommendations for a fictional city.
Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM.
Milestone 3: Develop a Streamlit UI to input interests and destination, generating a mini-guide.
Milestone 4: Evaluate the relevance and appeal of the generated recommendations.
Milestone 5: Implement adapter merging for inference.
"""
from __future__ import annotations

import re
from typing import Any

from app.dataset import (
    TRAVEL_DATASET_RAW,
    TravelRecord,
    get_dataset_stats,
    get_training_pairs,
    load_dataset,
)


# ============================================================================
# Milestone 1: Travel Dataset & Fine-Tuning Formatting
# ============================================================================

def format_prompt(destination: str, interests: list[str] | str) -> str:
    """Format input prompt for travel guide generation."""
    if isinstance(interests, list):
        interests_str = ", ".join(interests)
    else:
        interests_str = str(interests).strip()
    return f"Destination: {destination.strip()} | Interests: {interests_str} -> Guide:"


def format_completion(title: str, highlights: list[str], recommendation: str, local_tip: str) -> str:
    """Format expected guide completion."""
    hl = "; ".join(highlights) if highlights else "Historic Quarter"
    return f" {title}. Recommended stops: {hl}. {recommendation} Insider Tip: {local_tip}"


def create_training_pairs(records: list[TravelRecord] | None = None) -> list[dict[str, str]]:
    """Create structured prompt-completion pairs suitable for causal LM fine-tuning."""
    data = records if records is not None else load_dataset()
    pairs = []
    for r in data:
        pair = r.to_training_pair()
        pairs.append(pair)
    return pairs


def milestone_1() -> dict[str, Any]:
    """Milestone 1: Prepare a small dataset of travel interests and corresponding local recommendations for a fictional city.

    Structures text data for fine-tuning with clear input-output pairs.
    """
    records = load_dataset()
    stats = get_dataset_stats()
    training_pairs = create_training_pairs(records)

    # Validate that every sample has valid prompt, completion, and full_text
    solaria_records = [r for r in records if r.destination == "Solaria"]
    sample = training_pairs[0]

    return {
        "status": "completed",
        "records_count": len(records),
        "fictional_cities": stats["destinations"],
        "primary_city": "Solaria",
        "solaria_records_count": len(solaria_records),
        "unique_interests_count": stats["unique_interests_count"],
        "unique_interests": stats["unique_interests"],
        "training_pairs": training_pairs,
        "sample_prompt": sample["prompt"],
        "sample_completion": sample["completion"],
        "sample_full_text": sample["full_text"],
    }


# --- Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).
# Hint: Use the 'LoRA adapter injection' and 'training loop' techniques from the drill to adapt the LLM.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Develop a Streamlit UI to input interests and destination, generating a mini-guide.
# Hint: Integrate your fine-tuned model with a Streamlit interface for user input and guide generation.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Evaluate the relevance and appeal of the generated recommendations.
# Hint: Manually review the generated recommendations to assess their relevance and appeal.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Implement adapter merging for inference.
# Hint: Apply the 'adapter merging' technique to prepare your model for efficient, standalone inference.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
