"""Streamlit entry point: `streamlit run app/main.py`

Interactive interface for generating personalized local travel guides using LoRA fine-tuned models.
"""
from __future__ import annotations

import streamlit as st

from app.core import (
    build_travel_model,
    format_prompt,
    generate_mini_guide,
    inject_lora,
    load_dataset,
)
from app.model import TravelTokenizer

# Page layout configuration
st.set_page_config(
    page_title="Local Travel Guide Generator",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ Local Travel Guide Generator")
st.write(
    "Travel agencies use this tool to fine-tune a language model with LoRA, generating concise, "
    "personalized local travel guides and recommendations for tourists based on their interests and destination."
)

# Sidebar Configuration
st.sidebar.header("⚙️ Model & Generation Settings")
model_type = st.sidebar.selectbox(
    "Adapter / Model Status",
    ["LoRA Fine-Tuned LLM (Active)", "Merged Standalone Model (Inference Mode)", "Pre-trained Base Model"],
    index=0,
)
temperature = st.sidebar.slider("Sampling Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
max_tokens = st.sidebar.slider("Max New Tokens", min_value=20, max_value=120, value=60, step=10)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Architecture Details:**\n"
    "- Base: Causal Transformer (GPT-2 style)\n"
    "- Adaptation: LoRA (rank $r=4$, $\\alpha=8.0$)\n"
    "- Target Projections: Self-Attention Linear layers\n"
    "- Trainable Parameters: < 2% of base model"
)

# Destination & Interests Form
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📍 Tourist Destination")
    destination_choice = st.selectbox(
        "Select Fictional Destination:",
        ["Solaria", "Novalis", "Verdantia", "Aethelgard", "Custom City"],
        index=0,
    )

    if destination_choice == "Custom City":
        destination = st.text_input("Enter Destination City Name:", value="Oasis Aeterna")
    else:
        destination = destination_choice

    city_descriptions = {
        "Solaria": "☀️ The Sunlit Coastal Citadel: Mosaic galleries, azure harbors, and seaside stone ovens.",
        "Novalis": "🌸 City of Floating Gardens: Waterway canals, night bazaars, and baroque opera.",
        "Verdantia": "🌲 The Forest Canopy Metropolis: High suspension skywalks, redwood trails, and tea houses.",
        "Aethelgard": "🏔️ Mountain Keep & Starlit Peaks: Alpine climbing, thermal springs, and starlight observatories.",
    }
    if destination in city_descriptions:
        st.caption(city_descriptions[destination])

with col2:
    st.subheader("🎯 Tourist Interests")
    available_interests = [
        "art",
        "local coffee",
        "history",
        "ancient ruins",
        "culinary",
        "sunset dining",
        "nature",
        "scenic harbors",
        "nightlife",
        "live music",
        "wellness",
        "relaxation",
        "botanical gardens",
        "canal tours",
        "night markets",
        "chamber music",
        "canopy walks",
        "eco-trekking",
        "tea houses",
        "alpine climbing",
        "stargazing",
        "thermal baths",
    ]
    selected_interests = st.multiselect(
        "Choose Tourist Interests & Passions:",
        options=available_interests,
        default=["art", "local coffee"] if destination == "Solaria" else [available_interests[0]],
    )
    custom_interests = st.text_input("Additional specific keywords or interests (optional):", "")

# Merge interests
all_user_interests = list(selected_interests)
if custom_interests:
    for item in custom_interests.split(","):
        if item.strip() and item.strip() not in all_user_interests:
            all_user_interests.append(item.strip())

st.markdown("---")

# Generate button
generate_clicked = st.button("✨ Generate Personalized Local Travel Guide", type="primary", use_container_width=True)

if generate_clicked:
    if not all_user_interests:
        st.warning("Please select at least one travel interest.")
    else:
        with st.spinner("Adapting language model weights and generating personalized guide..."):
            # Initialize tokenizer and model
            records = load_dataset()
            tokenizer = TravelTokenizer(records)
            model = build_travel_model(vocab_size=tokenizer.vocab_size, hidden_dim=32, num_layers=2)
            inject_lora(model, rank=4, alpha=8.0)

            result = generate_mini_guide(
                model=model,
                tokenizer=tokenizer,
                destination=destination,
                interests=all_user_interests,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

        st.success(f"Personalized Guide Generated for {destination}!")

        # Display Guide Card
        st.markdown(f"### 🏛️ {result['title']}")
        st.markdown(f"**Destination:** {result['destination']} | **Tailored Interests:** {', '.join(result['interests'])}")

        st.markdown("#### 🌟 Must-Visit Highlights")
        for stop in result["highlights"]:
            st.markdown(f"- **{stop}**")

        st.markdown("#### 📖 Curated Itinerary & Recommendations")
        st.write(result["recommendation"])

        st.info(f"💡 **Insider Local Tip:** {result['insider_tip']}")

        # Full prompt-completion display
        with st.expander("🔍 View LoRA Prompt & Neural Completion Tokens"):
            st.code(
                f"Prompt:\n{result['prompt']}\n\nCompletion:\n{result['complete_guide']}",
                language="markdown",
            )

        # Export Guide
        guide_text = (
            f"# {result['title']}\n\n"
            f"**Destination:** {result['destination']}\n"
            f"**Interests:** {', '.join(result['interests'])}\n\n"
            f"## Highlights\n" + "\n".join(f"- {h}" for h in result["highlights"]) + "\n\n"
            f"## Recommendation\n{result['recommendation']}\n\n"
            f"**Insider Tip:** {result['insider_tip']}\n"
        )
        st.download_button(
            label="📥 Download Travel Guide (.txt)",
            data=guide_text,
            file_name=f"{destination.lower()}_travel_guide.txt",
            mime="text/plain",
        )
