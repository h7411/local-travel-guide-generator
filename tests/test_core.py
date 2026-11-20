"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass."""
import pytest
import torch

from app.core import (
    LoRALinear,
    build_travel_model,
    calculate_appeal,
    calculate_relevance,
    convert_to_standalone_model,
    create_training_pairs,
    evaluate_recommendation,
    format_completion,
    format_prompt,
    generate_mini_guide,
    inject_lora,
    merge_adapters_for_inference,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
)
from app.model import TravelTokenizer


# Milestone 1: Prepare a small dataset of travel interests and corresponding local recommendations for a fictional city.
def test_milestone_1():
    result = milestone_1()
    assert result["status"] == "completed"
    assert result["records_count"] >= 10
    assert "Solaria" in result["fictional_cities"]
    assert result["solaria_records_count"] >= 5
    assert result["unique_interests_count"] >= 5

    # Validate training pairs structure
    training_pairs = result["training_pairs"]
    assert len(training_pairs) == result["records_count"]

    first_pair = training_pairs[0]
    assert "prompt" in first_pair
    assert "completion" in first_pair
    assert "full_text" in first_pair
    assert first_pair["full_text"] == first_pair["prompt"] + first_pair["completion"]
    assert first_pair["prompt"].startswith("Destination:")
    assert "Interests:" in first_pair["prompt"]
    assert first_pair["prompt"].endswith("-> Guide:")

    # Test format_prompt and format_completion helpers
    prompt = format_prompt("Solaria", ["art", "local coffee"])
    assert prompt == "Destination: Solaria | Interests: art, local coffee -> Guide:"

    comp = format_completion("Sunlit Mosaics", ["Old Quarter"], "Explore historic mosaics.", "Visit early.", destination="Solaria")
    assert "Sunlit Mosaics" in comp
    assert "Old Quarter" in comp
    assert "Insider Tip: Visit early." in comp


# Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).
def test_milestone_2():
    result = milestone_2(rank=2, alpha=4.0, epochs=2, hidden_dim=32)
    assert result["status"] == "completed"
    assert result["injected_layers_count"] > 0
    assert result["trainable_parameters"] > 0
    assert result["trainable_percentage"] < 5.0  # LoRA uses only a small fraction of parameters
    assert result["base_weights_frozen"] is True
    assert result["adapter_weights_trainable"] is True
    assert len(result["loss_history"]) == 2
    assert result["final_loss"] > 0.0

    # Unit test LoRALinear mechanics directly
    base = torch.nn.Linear(16, 32)
    lora = LoRALinear(base, rank=2, alpha=4.0)

    # Base weight must be frozen
    assert lora.base_layer.weight.requires_grad is False
    assert lora.lora_A.requires_grad is True
    assert lora.lora_B.requires_grad is True

    # At init, lora_B is zero, so output exactly matches base_layer
    x = torch.randn(2, 4, 16)
    with torch.no_grad():
        out_base = base(x)
        out_lora = lora(x)
    assert torch.allclose(out_base, out_lora, atol=1e-6)

    # Gradients only flow to adapter weights
    out_lora = lora(x)
    loss = out_lora.sum()
    loss.backward()
    assert base.weight.grad is None
    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None


# Milestone 3: Develop a Streamlit UI to input interests and destination, generating a mini-guide.
def test_milestone_3():
    result = milestone_3(destination="Solaria", interests=["art", "local coffee"])
    assert result["status"] == "completed"
    assert result["destination"] == "Solaria"
    assert "art" in result["interests"]
    assert "local coffee" in result["interests"]
    assert len(result["highlights"]) > 0
    assert len(result["recommendation"]) > 20
    assert len(result["insider_tip"]) > 5
    assert len(result["complete_guide"]) > 50

    # Test generate_mini_guide with another fictional city
    guide_nov = generate_mini_guide(
        model=None,
        tokenizer=None,
        destination="Novalis",
        interests=["canal tours", "botanical gardens"],
    )
    assert guide_nov["status"] == "success"
    assert guide_nov["destination"] == "Novalis"
    assert "Lotus" in guide_nov["complete_guide"] or "canal" in guide_nov["complete_guide"].lower()

    # Smoke test python compilation of Streamlit entry point
    import py_compile
    py_compile.compile("app/main.py", doraise=True)


# Milestone 4: Evaluate the relevance and appeal of the generated recommendations.
def test_milestone_4():
    result = milestone_4()
    assert result["status"] == "completed"
    assert result["evaluation_cases_count"] >= 5
    assert result["average_relevance"] >= 0.70
    assert result["average_appeal"] >= 0.70
    assert result["average_composite_score"] >= 0.70

    # Test individual metrics functions
    sample_text = (
        "[Solaria] Sunlit Bohemia: Mosaics and Artisanal Roasts. Recommended stops: Sunken Mosaic Gallery, Café Antiqua in Solaria. "
        "Begin your morning at Café Antiqua savoring a vibrant saffron roast brewed over hot sand. "
        "Stroll scenic lanes and admire stunning historic mosaics. Insider Tip: Arrive early for fresh pastries."
    )
    rel = calculate_relevance(sample_text, "Solaria", ["art", "local coffee"])
    assert rel >= 0.70

    app = calculate_appeal(sample_text)
    assert app >= 0.70

    eval_info = evaluate_recommendation(sample_text, "Solaria", ["art", "local coffee"])
    assert eval_info["quality_verdict"] in ("Strong", "Exceptional")
    assert "rubric" in eval_info


# Milestone 5: Implement adapter merging for inference.
def test_milestone_5():
    result = milestone_5()
    assert result["status"] == "completed"
    assert result["merged_layers_count"] > 0
    assert result["max_discrepancy"] < 1e-4
    assert result["outputs_match"] is True
    assert result["adapter_modules_removed"] is True
    assert result["standalone_inference_ready"] is True

    # Unit test LoRALinear merge and unmerge directly
    base = torch.nn.Linear(8, 16)
    lora = LoRALinear(base, rank=2, alpha=4.0)

    # Set non-zero adapter weights
    torch.nn.init.normal_(lora.lora_A, mean=0.0, std=0.1)
    torch.nn.init.normal_(lora.lora_B, mean=0.0, std=0.1)

    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        out_before_merge = lora(x)

    # Merge weights
    lora.merge()
    assert lora.merged is True

    with torch.no_grad():
        out_after_merge = lora(x)
        out_direct_base = lora.base_layer(x)

    assert torch.allclose(out_before_merge, out_after_merge, atol=1e-5)
    assert torch.allclose(out_after_merge, out_direct_base, atol=1e-5)

    # Unmerge weights
    lora.unmerge()
    assert lora.merged is False
    with torch.no_grad():
        out_unmerged = lora.base_layer(x)
    # Base layer is restored
    assert not torch.allclose(out_unmerged, out_after_merge, atol=1e-4)
