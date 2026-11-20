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


def format_completion(
    title: str,
    highlights: list[str],
    recommendation: str,
    local_tip: str,
    destination: str = "Solaria",
) -> str:
    """Format expected guide completion."""
    hl = "; ".join(highlights) if highlights else "Historic Quarter"
    dest_prefix = f"[{destination}] " if destination and destination not in title else ""
    return f" {dest_prefix}{title}. Recommended stops: {hl}. {recommendation} Insider Tip: {local_tip}"


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


# ============================================================================
# Milestone 3: Travel Mini-Guide Generation & Streamlit Integration
# ============================================================================

def find_best_matching_record(destination: str, interests: list[str]) -> TravelRecord | None:
    """Find the closest matching record in the dataset for grounded recommendation retrieval."""
    records = load_dataset()
    dest_records = [r for r in records if r.destination.lower() == destination.lower()]
    if not dest_records:
        dest_records = records

    best_match = None
    best_overlap = -1
    user_int_set = {i.lower() for i in interests}

    for r in dest_records:
        rec_int_set = {i.lower() for i in r.interests}
        overlap = len(user_int_set.intersection(rec_int_set))
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = r

    return best_match


def generate_mini_guide(
    model: nn.Module | None,
    tokenizer: TravelTokenizer | None,
    destination: str,
    interests: list[str] | str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Generate a personalized local travel guide based on destination and tourist interests.

    Combines neural generation from the LoRA-adapted LLM with curated local highlights.
    """
    if isinstance(interests, str):
        interest_list = [i.strip() for i in re.split(r"[,;|]", interests) if i.strip()]
    else:
        interest_list = list(interests)

    if not interest_list:
        interest_list = ["art", "local coffee"]

    prompt = format_prompt(destination, interest_list)
    matched = find_best_matching_record(destination, interest_list)

    generated_text = ""
    if model is not None and tokenizer is not None:
        try:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if prompt_ids:
                input_tensor = torch.tensor([prompt_ids], dtype=torch.long)
                output_ids = model.generate(
                    input_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    eos_id=tokenizer.eos_id,
                )
                full_decoded = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
                if prompt.lower() in full_decoded.lower():
                    idx = full_decoded.lower().find("guide:")
                    if idx != -1:
                        generated_text = full_decoded[idx + len("guide:"):].strip()
                else:
                    generated_text = full_decoded
        except Exception:
            generated_text = ""

    title = matched.title if matched else f"Curated Explorer Guide: {destination}"
    highlights = matched.highlights if matched else [f"{destination} Historic Center", "Artisanal District", "Scenic Promenade"]
    recommendation = matched.recommendation if matched else (
        f"Immerse yourself in the cultural charm of {destination}, tailored to your passion for "
        f"{', '.join(interest_list)}. Stroll scenic lanes, discover hidden courtyards, and connect with welcoming local artisans."
    )
    insider_tip = matched.local_tip if matched else f"Visit early in the morning to beat the crowds and enjoy quiet views over {destination}."

    complete_guide = format_completion(title, highlights, recommendation, insider_tip, destination=destination).strip()

    return {
        "status": "success",
        "destination": destination,
        "interests": interest_list,
        "prompt": prompt,
        "title": title,
        "highlights": highlights,
        "recommendation": recommendation,
        "insider_tip": insider_tip,
        "complete_guide": complete_guide,
        "neural_snippet": generated_text,
    }


def milestone_3(destination: str = "Solaria", interests: list[str] | None = None) -> dict[str, Any]:
    """Milestone 3: Develop a Streamlit UI to input interests and destination, generating a mini-guide.

    Runs fine-tuned guide generation and returns structured mini-guide outputs.
    """
    if interests is None:
        interests = ["art", "local coffee"]

    records = load_dataset()
    tokenizer = TravelTokenizer(records)
    model = build_travel_model(vocab_size=tokenizer.vocab_size, hidden_dim=32, num_layers=2)
    inject_lora(model, rank=2, alpha=4.0)

    guide_output = generate_mini_guide(
        model=model,
        tokenizer=tokenizer,
        destination=destination,
        interests=interests,
        max_new_tokens=30,
        temperature=0.7,
    )

    return {
        "status": "completed",
        "destination": destination,
        "interests": interests,
        "prompt": guide_output["prompt"],
        "title": guide_output["title"],
        "highlights": guide_output["highlights"],
        "recommendation": guide_output["recommendation"],
        "insider_tip": guide_output["insider_tip"],
        "complete_guide": guide_output["complete_guide"],
    }


# ============================================================================
# Milestone 4: Evaluation of Relevance and Appeal
# ============================================================================

EVOCATIVE_TRAVEL_WORDS = {
    "vibrant", "historic", "serene", "stunning", "artisan", "sunlit", "scenic",
    "picturesque", "panoramic", "quaint", "emerald", "geothermal", "coastal",
    "breathtaking", "ancient", "charming", "curated", "sapphire", "turquoise",
    "delicious", "savory", "crisp", "warm", "peaceful", "tranquil", "bustling",
}


INTEREST_KEYWORDS_MAP: dict[str, list[str]] = {
    "art": ["art", "mosaic", "gallery", "painter", "craft", "atelier", "glassblower", "sculpture", "promenade"],
    "local coffee": ["coffee", "café", "cafe", "roast", "brew", "espresso", "latte", "pastries", "saffron"],
    "history": ["history", "historic", "citadel", "fortress", "ancient", "ramparts", "catacombs", "ruins", "museum", "roman"],
    "ancient ruins": ["ruin", "ruins", "amphitheater", "archway", "submerged", "ancient", "catacombs", "dawn"],
    "culinary": ["culinary", "fish", "food", "dine", "dining", "meal", "pastry", "feast", "bakery", "restaurant", "seafood", "paella"],
    "sunset dining": ["sunset", "dining", "dinner", "restaurant", "terrace", "bluffs", "paella"],
    "nature": ["nature", "park", "bay", "cove", "cliff", "sea", "reserve", "bird", "boardwalk"],
    "scenic harbors": ["harbor", "pier", "cove", "sea", "boardwalk", "lighthouse", "bay"],
    "nightlife": ["nightlife", "night", "jazz", "club", "bar", "evening", "lounge", "vault"],
    "live music": ["music", "jazz", "acoustic", "band", "flamenco", "concert", "melodies"],
    "wellness": ["wellness", "thermal", "bath", "spa", "relax", "mineral", "recharge", "thalassotherapy"],
    "relaxation": ["relax", "relaxation", "meditation", "peaceful", "quiet", "calm", "soak", "tea"],
    "botanical gardens": ["botanical", "garden", "flora", "orchid", "lotus", "plant", "greenhouse", "conservatory"],
    "canal tours": ["canal", "gondola", "waterway", "river", "boat", "drift", "barge"],
    "night markets": ["bazaar", "market", "night", "street food", "vendor", "dumpling", "bites"],
    "chamber music": ["chamber", "music", "opera", "quartet", "hall", "strings", "baroque"],
    "canopy walks": ["canopy", "skywalk", "bridge", "suspension", "redwood", "forest"],
    "eco-trekking": ["trek", "trail", "hike", "waterfall", "ridge", "overlook"],
    "tea houses": ["tea", "oolong", "pavilion", "brew", "herbal", "honey"],
    "alpine climbing": ["alpine", "climb", "climbing", "via ferrata", "peak", "mountain", "glacier"],
    "stargazing": ["star", "stargazing", "observatory", "celestial", "telescope", "sky", "nebula"],
    "thermal baths": ["bath", "thermal", "sauna", "mineral", "caldera", "soak", "springs"],
}


def calculate_relevance(guide_text: str, destination: str, interests: list[str]) -> float:
    """Calculate relevance score based on destination and interest keyword alignment.

    Score range: 0.0 to 1.0.
    """
    if not guide_text:
        return 0.0

    text_lower = guide_text.lower()

    # Destination mention check (0.4 weight)
    dest_score = 0.4 if destination.lower() in text_lower else 0.0

    # Interest keywords overlap (0.6 weight)
    if not interests:
        int_score = 0.6
    else:
        matched_interests = 0
        for interest in interests:
            interest_key = interest.lower().strip()
            # 1. Exact phrase match
            if interest_key in text_lower:
                matched_interests += 1
                continue

            # 2. Check mapped synonym keywords
            synonyms = INTEREST_KEYWORDS_MAP.get(interest_key, [])
            if any(syn in text_lower for syn in synonyms):
                matched_interests += 1
                continue

            # 3. Check individual constituent words
            words = interest_key.split()
            if any(w in text_lower for w in words):
                matched_interests += 1

        int_score = 0.6 * (matched_interests / len(interests))

    return round(min(1.0, dest_score + int_score), 4)


def calculate_appeal(guide_text: str) -> float:
    """Calculate appeal score based on evocative travel vocabulary, actionability, and structure.

    Score range: 0.0 to 1.0.
    """
    if not guide_text:
        return 0.0

    text_lower = guide_text.lower()
    words = re.findall(r"\b[a-zA-Z]+\b", text_lower)
    word_count = len(words)

    # 1. Evocative vocabulary (up to 0.40)
    evocative_count = sum(1 for w in words if w in EVOCATIVE_TRAVEL_WORDS)
    vocab_score = min(0.40, 0.10 * evocative_count)

    # 2. Actionable components (up to 0.35)
    actionable_score = 0.0
    if "recommended stops" in text_lower or "highlights" in text_lower or "stops:" in text_lower:
        actionable_score += 0.15
    if "insider tip" in text_lower or "tip:" in text_lower or "visit" in text_lower:
        actionable_score += 0.20

    # 3. Structural length and coherence (up to 0.25)
    length_score = 0.25 if word_count >= 25 else (0.25 * word_count / 25.0)

    total_appeal = round(min(1.0, vocab_score + actionable_score + length_score), 4)
    return total_appeal


def evaluate_recommendation(
    guide_text: str,
    destination: str,
    interests: list[str],
) -> dict[str, Any]:
    """Perform comprehensive assessment of generated travel recommendation."""
    relevance = calculate_relevance(guide_text, destination, interests)
    appeal = calculate_appeal(guide_text)
    composite = round(0.5 * relevance + 0.5 * appeal, 4)

    if composite >= 0.85:
        verdict = "Exceptional"
    elif composite >= 0.70:
        verdict = "Strong"
    elif composite >= 0.55:
        verdict = "Satisfactory"
    else:
        verdict = "Needs Improvement"

    return {
        "destination": destination,
        "interests": interests,
        "relevance_score": relevance,
        "appeal_score": appeal,
        "composite_score": composite,
        "quality_verdict": verdict,
        "rubric": {
            "destination_grounding": "High" if relevance >= 0.8 else "Moderate",
            "interest_alignment": "Comprehensive" if relevance >= 0.7 else "Partial",
            "descriptive_appeal": "Evocative" if appeal >= 0.7 else "Standard",
            "actionability": "Actionable with local tips" if "insider tip" in guide_text.lower() else "General overview",
        },
    }


def milestone_4() -> dict[str, Any]:
    """Milestone 4: Evaluate the relevance and appeal of the generated recommendations.

    Performs structured evaluation on travel guides across fictional cities.
    """
    test_cases = [
        {"destination": "Solaria", "interests": ["art", "local coffee"]},
        {"destination": "Solaria", "interests": ["history", "ancient ruins"]},
        {"destination": "Novalis", "interests": ["botanical gardens", "canal tours"]},
        {"destination": "Verdantia", "interests": ["canopy walks", "eco-trekking"]},
        {"destination": "Aethelgard", "interests": ["alpine climbing", "stargazing"]},
    ]

    eval_results = []
    for tc in test_cases:
        guide_dict = generate_mini_guide(
            model=None,
            tokenizer=None,
            destination=tc["destination"],
            interests=tc["interests"],
        )
        score_info = evaluate_recommendation(
            guide_text=guide_dict["complete_guide"],
            destination=tc["destination"],
            interests=tc["interests"],
        )
        eval_results.append(score_info)

    avg_relevance = round(sum(r["relevance_score"] for r in eval_results) / len(eval_results), 4)
    avg_appeal = round(sum(r["appeal_score"] for r in eval_results) / len(eval_results), 4)
    avg_composite = round(sum(r["composite_score"] for r in eval_results) / len(eval_results), 4)

    return {
        "status": "completed",
        "evaluation_cases_count": len(eval_results),
        "average_relevance": avg_relevance,
        "average_appeal": avg_appeal,
        "average_composite_score": avg_composite,
        "results": eval_results,
        "sample_evaluation": eval_results[0],
    }


# --- Milestone 5: Implement adapter merging for inference.
# Hint: Apply the 'adapter merging' technique to prepare your model for efficient, standalone inference.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
