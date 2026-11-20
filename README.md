# Local Travel Guide Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/r3tacuw4he/badge.svg)](https://ask.thili.ai/b/r3tacuw4he)

This project develops a Local Travel Guide Generator, a tool designed for travel agencies in the Travel & Hospitality domain. It leverages LoRA fine-tuning to adapt a language model, enabling it to create personalized local travel guides and recommendations for tourists based on their specific interests and chosen destination.

## What you are building

**Goal.** You will build a tool that fine-tunes a language model using LoRA to generate personalized local travel guides and recommendations for tourists based on their interests and destination.

**Done when.** You are done when you have a Streamlit UI that takes user interests and a destination, and then uses your LoRA-tuned LLM to generate relevant and appealing local travel recommendations after training and merging the adapter.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Prepare a small dataset of travel interests and corresponding local recommendations for a fictional city.**  
      Hint: Focus on structuring your text data for fine-tuning, ensuring clear input-output pairs.
- [x] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).**  
      Hint: Use the 'LoRA adapter injection' and 'training loop' techniques from the drill to adapt the LLM.
- [x] **3. Develop a Streamlit UI to input interests and destination, generating a mini-guide.**  
      Hint: Integrate your fine-tuned model with a Streamlit interface for user input and guide generation.
- [x] **4. Evaluate the relevance and appeal of the generated recommendations.**  
      Hint: Manually review the generated recommendations to assess their relevance and appeal.
- [x] **5. Implement adapter merging for inference.**  
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

## Architecture & Implementation Overview

1. **Structured Travel Dataset (`app/dataset.py`)**:
   - Comprehensive multi-city records across fictional destinations (**Solaria**, **Novalis**, **Verdantia**, and **Aethelgard**).
   - Structured prompt-completion formatting with input (`Destination: ... | Interests: ... -> Guide:`) and rich outputs.

2. **LoRA Adapter Injection & Training (`app/model.py`)**:
   - `LoRALinear`: Low-rank adaptation ($h = W_0 x + \frac{\alpha}{r} B A x$) with frozen pre-trained weights ($W_0$) and trainable rank decomposition matrices ($A \sim \mathcal{N}(0, \sigma^2), B = 0$).
   - `inject_lora`: Dynamically targets self-attention and projection linear layers while keeping > 98% of model parameters frozen.
   - `train_lora_model`: Optimization loop updating only adapter weights using AdamW with cross-entropy causal language modeling loss.

3. **Streamlit User Interface (`app/main.py`)**:
   - Interactive web application allowing tourists and travel agents to select destinations, choose interests, customize prompts, generate guides, and export itineraries.

4. **Relevance and Appeal Evaluation (`app/core.py`)**:
   - Automatic evaluation scoring for destination grounding, thematic interest alignment, evocative prose vocabulary, and actionability.

5. **Adapter Weight Merging (`app/model.py`)**:
   - Merges $\Delta W = \frac{\alpha}{r} (B A)$ directly into base weights $W_0$ and strips adapter branches for standalone inference with zero runtime latency overhead.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever reaches GitHub or GitLab, treat it as leaked and make a new one.

## Stretch ideas

- Expand the dataset to cover multiple fictional cities or more diverse interests to broaden the generator's scope.
- Add features to the Streamlit UI for saving, printing, or sharing generated guides.
- Experiment with different LLM base models or LoRA configurations to further optimize recommendation quality and creativity.
