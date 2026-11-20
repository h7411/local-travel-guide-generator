"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass."""
import pytest

from app.core import (
    create_training_pairs,
    format_completion,
    format_prompt,
    milestone_1,
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
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


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
