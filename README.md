# Local Travel Guide Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/r3tacuw4he/badge.svg)](https://ask.thili.ai/b/r3tacuw4he)

This project develops a Local Travel Guide Generator, a tool designed for travel agencies in the Travel & Hospitality domain. It leverages LoRA fine-tuning to adapt a language model, enabling it to create personalized local travel guides and recommendations for tourists based on their specific interests and chosen destination.

## What you are building

**Goal.** You will build a tool that fine-tunes a language model using LoRA to generate personalized local travel guides and recommendations for tourists based on their interests and destination.

**Done when.** You are done when you have a Streamlit UI that takes user interests and a destination, and then uses your LoRA-tuned LLM to generate relevant and appealing local travel recommendations after training and merging the adapter.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a small dataset of travel interests and corresponding local recommendations for a fictional city.**  
      Hint: Focus on structuring your text data for fine-tuning, ensuring clear input-output pairs.
- [ ] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).**  
      Hint: Use the 'LoRA adapter injection' and 'training loop' techniques from the drill to adapt the LLM.
- [ ] **3. Develop a Streamlit UI to input interests and destination, generating a mini-guide.**  
      Hint: Integrate your fine-tuned model with a Streamlit interface for user input and guide generation.
- [ ] **4. Evaluate the relevance and appeal of the generated recommendations.**  
      Hint: Manually review the generated recommendations to assess their relevance and appeal.
- [ ] **5. Implement adapter merging for inference.**  
      Hint: Apply the 'adapter merging' technique to prepare your model for efficient, standalone inference.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Expand the dataset to cover multiple fictional cities or more diverse interests to broaden the generator's scope.
- Add features to the Streamlit UI for saving, printing, or sharing generated guides.
- Experiment with different LLM base models or LoRA configurations to further optimize recommendation quality and creativity.
