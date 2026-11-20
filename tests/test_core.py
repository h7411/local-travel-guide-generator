"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass."""
import pytest
import torch

from app.core import (
    LoRALinear,
    build_travel_model,
    create_training_pairs,
    format_completion,
    format_prompt,
    inject_lora,
    milestone_1,
    milestone_2,
)


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

    comp = format_completion("Sunlit Mosaics", ["Old Quarter"], "Explore historic mosaics.", "Visit early.")
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
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Evaluate the relevance and appeal of the generated recommendations.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Implement adapter merging for inference.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")
