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

import torch
import torch.nn as nn

from app.dataset import (
    TRAVEL_DATASET_RAW,
    TravelRecord,
    get_dataset_stats,
    get_training_pairs,
    load_dataset,
)
from app.model import (
    LoRALinear,
    MiniCausalLM,
    TravelTokenizer,
    inject_lora,
    merge_lora_weights,
    train_lora_model,
    unmerge_lora_weights,
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


# ============================================================================
# Milestone 2: LoRA Adapter Injection & Training Loop
# ============================================================================

def build_travel_model(
    vocab_size: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
) -> MiniCausalLM:
    """Instantiate a base causal language model for fine-tuning."""
    return MiniCausalLM(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )


def milestone_2(
    rank: int = 4,
    alpha: float = 8.0,
    epochs: int = 3,
    lr: float = 1e-3,
    hidden_dim: int = 64,
) -> dict[str, Any]:
    """Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM.

    Applies LoRA adapter injection to target linear layers, freezes base model weights,
    and runs the training loop over the travel guide dataset.
    """
    records = load_dataset()
    tokenizer = TravelTokenizer(records)

    # 1. Instantiate pre-trained base model
    model = build_travel_model(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
    )

    # 2. Inject LoRA adapters
    injection_stats = inject_lora(model, rank=rank, alpha=alpha)

    # Verify that base weights are frozen
    base_weights_frozen = True
    adapter_weights_trainable = True

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if module.base_layer.weight.requires_grad:
                base_weights_frozen = False
            if not module.lora_A.requires_grad or not module.lora_B.requires_grad:
                adapter_weights_trainable = False

    # 3. Fine-tune LoRA adapters using training loop
    training_texts = [r.to_training_pair()["full_text"] for r in records]
    training_result = train_lora_model(
        model=model,
        training_texts=training_texts,
        tokenizer=tokenizer,
        epochs=epochs,
        lr=lr,
        max_len=64,
    )

    return {
        "status": "completed",
        "model": model,
        "tokenizer": tokenizer,
        "rank": rank,
        "alpha": alpha,
        "trainable_parameters": injection_stats["trainable_parameters"],
        "total_parameters": injection_stats["total_parameters"],
        "trainable_percentage": injection_stats["trainable_percentage"],
        "injected_layers_count": injection_stats["injected_layers_count"],
        "base_weights_frozen": base_weights_frozen,
        "adapter_weights_trainable": adapter_weights_trainable,
        "epochs": training_result["epochs"],
        "initial_loss": training_result["initial_loss"],
        "final_loss": training_result["final_loss"],
        "loss_reduced": training_result["loss_reduced"],
        "loss_history": training_result["loss_history"],
    }


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
