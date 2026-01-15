import json
import streamlit as st
import torch
import numpy as np
import streamlit.components.v1 as components
from src.model.model_utils import ModelLoader, load_bert_tokenizer
from src.utils import load_config_file
from src.inference import predict_baseline, predict_bilstm, predict_bilstm_attn, predict_bert
from src.visualization import (
    create_attention_heatmap_html,
    create_slot_predictions_html,
    create_intent_prediction_html,
    create_feature_importance_html,
    create_model_info_html,
    create_model_row_html
)


st.set_page_config(layout="wide", page_title="SLU Multi-Model Dashboard", page_icon="🧠")

# Custom CSS for minimal white design
st.markdown("""
<style>
    /* Global white minimalistic theme */
    .main {
        background-color: #ffffff;
    }
    
    /* Remove padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Header styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 300;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 15px;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1E3A8A;
        box-shadow: 0 0 0 3px rgba(30, 58, 138, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-weight: 500;
        font-size: 15px;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 58, 138, 0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: #f9fafb;
        padding: 8px;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #6b7280;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: 500;
        background: transparent;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: white;
        color: #1E3A8A;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        color: #1E3A8A;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 400;
        color: #1f2937;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
        letter-spacing: 0.3px;
    }
    
    /* Model Card Container */
    .model-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border-top: 4px solid #1a1a1a; /* Default border */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
    }

    /* Hover effect matching the inference rows */
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.1);
    }

    /* Specific Model Color Borders (Syncing with theme_colors in visualization.py) */
    .card-baseline { border-top-color: #374151; }
    .card-bilstm { border-top-color: #92400E; }
    .card-attn { border-top-color: #065F46; }
    .card-bert { border-top-color: #1E3A8A; }

    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 8px;
        color: #1f2937;
    }

    .card-tag {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 700;
        margin-bottom: 12px;
        display: block;
    }

    /* Input Container */
    .input-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 40px 20px;
        background: #ffffff;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
        border: 1px solid #f0f2f6;
    }

    /* Modernized Search Bar */
    div[data-baseweb="input"] {
        border-radius: 12px !important;
        border: 1px solid #e5e7eb !important;
        padding: 4px 8px !important;
        transition: all 0.3s ease !important;
    }

    div[data-baseweb="input"]:focus-within {
        border-color: #1E3A8A !important;
        box-shadow: 0 0 0 4px rgba(30, 58, 138, 0.1) !important;
    }

    /* Refined Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 14px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 15px rgba(30, 58, 138, 0.2) !important;
    }

    /* Style for the Information Background Wrapper */
    .info-wrapper {
        background-color: #f8fafc; /* Very light grey-blue */
        margin: -2rem -4rem 2rem -4rem; /* Extend to edges of container */
        padding: 3rem 4rem;
        border-bottom: 1px solid #e2e8f0;
    }

    /* Modern Section Divider with Text */
    .section-divider {
        display: flex;
        align-items: center;
        text-align: center;
        margin: 3rem 0;
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .section-divider::before, .section-divider::after {
        content: '';
        flex: 1;
        border-bottom: 1px solid #e2e8f0;
    }

    .section-divider:not(:empty)::before { margin-right: 1.5rem; }
    .section-divider:not(:empty)::after { margin-left: 1.5rem; }

    /* Target the input-container to make it "pop" more */
    .input-container {
        background: white !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 10px 10px -5px rgba(0, 0, 0, 0.02) !important;
        transform: translateY(-50px); /* Overlap the grey section slightly */
        z-index: 10;
    }

    /* Footer Styling */
    .footer {
        position: relative;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: #6b7280;
        text-align: center;
        padding: 2rem 0;
        border-top: 1px solid #e5e7eb;
        margin-top: 4rem;
    }
    .footer a {
        color: #1E3A8A;
        text-decoration: none;
        margin: 0 15px;
        font-weight: 500;
        transition: color 0.2s ease;
    }
    .footer a:hover {
        color: #3B82F6;
    }
    .footer-icons {
        font-size: 1.2rem;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_resources():
    """Load config, vocabs and all 4 models."""
    config = load_config_file("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load vocabularies
    vocabs = {
        'word_to_id': json.load(open('files/checkpoints/word_to_id.json', 'r')),
        'slot_to_id': json.load(open('files/checkpoints/full_slot_mapping.json', 'r')),
        'intent_to_id': json.load(open('files/checkpoints/intent_to_id.json', 'r')),
        'id_to_slot': {v: k for k, v in json.load(open('files/checkpoints/full_slot_mapping.json', 'r')).items()},
        'id_to_intent': {v: k for k, v in json.load(open('files/checkpoints/intent_to_id.json', 'r')).items()}
    }
    
    # Load models
    model_loader = ModelLoader(config, device, vocabs)
    tokenizer = load_bert_tokenizer(config['bert_model_name'])
    
    models = {
        'Baseline': model_loader.load_baseline('files/checkpoints/baseline_model.joblib'),
        'JointBiLSTM': model_loader.load_bilstm('files/checkpoints/jointbilstm_model.pth'),
        'JointBiLSTM+Attn': model_loader.load_bilstm('files/checkpoints/jointbilstm_attn_model.pth', attn=True),
        'JointBERT': model_loader.load_bert('files/checkpoints/bert_model.pth')
    }
    
    return config, models, vocabs, tokenizer, device


def main():
    st.set_page_config(layout="wide", page_title="Slot Filling & Intent Detection Multi-Model Intelligence", page_icon="🧠")
    st.markdown("""
        <style>
        .main { background-color: #F8F9FB; }
        .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .result-card { background-color: white; padding: 20px; border-radius: 12px; border: 1px solid #E6E9EF; }
        </style>
    """, unsafe_allow_html=True)

    # Header & Context
    st.markdown('<h1 class="main-title">🧠 Slot Filling & Intent Detection</h1>', unsafe_allow_html=True)
    
    tab_intro, tab_analysis = st.tabs(["📄 Project Background", "🚀 Live Model Analysis"])

    # Brief Description and Instructions
    with tab_intro:
        # Submission Context
        st.markdown("""
        ### Welcome!
        In this project, we analyze and compare different approaches for **Slot Filling** and **Intent Detection**, 
        fundamental tasks in Natural Language Understanding (NLU).

        Project repo: https://github.com/archeltaneka/slot-filling-intent-detection
        
        > **Academic Submission:** This project was submitted for **FIT5149 - Applied Data Analysis** as part of a Master's degree at **Monash University**.
        """)
        
        # Dataset Section
        st.markdown("""
        ### 📊 The ATIS Dataset
        This project uses the ATIS (Airline Travel Information Systems) dataset, a well-known benchmark dataset for slot filling and intent detection tasks. 
        """)
        col1, col2, col3 = st.columns(3)
        col1.metric("Training Samples", "4,478")
        col2.metric("Unique Words", "869")
        col3.metric("Intent Classes", "21")
        
        st.markdown("""
        **Format:** `word1:slot1 word2:slot2 ... ⇐⇒ intent_label`
        
        **Example 1: Flight Booking**
        - *Input:* "i want to fly from baltimore to dallas round trip"
        - *Intent:* `atis_flight`
        - *Slots:* `baltimore:B-fromloc.city_name`, `dallas:B-toloc.city_name`
        
        **Example 2: Restrictions**
        - *Input:* "what is the baggage allowance"
        - *Intent:* `atis_restriction`
        - *Slots:* `what:O`, `is:O`, `the:O`, `baggage:O`, `allowance:O`
        """)
    
    with tab_analysis:
        with st.spinner("Loading models..."):
            config, models, vocabs, tokenizer, device = load_all_resources()
                
        st.markdown("### 🔍 About the Models")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        with m_col1:
            st.markdown("""
            <div class="model-card card-baseline">
                <span class="card-tag" style="color: #374151;">Statistical ML</span>
                <div class="card-title">Baseline</div>
                <p style="font-size: 0.9rem; color: #4b5563;">
                    A traditional approach using <b>Random Forest</b> for intent and <b>CRF</b> for slots. 
                    Fast and efficient, it serves as the performance floor for our benchmarks.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with m_col2:
            st.markdown("""
            <div class="model-card card-bilstm">
                <span class="card-tag" style="color: #92400E;">Recurrent Neural Net</span>
                <div class="card-title">JointBiLSTM</div>
                <p style="font-size: 0.9rem; color: #4b5563;">
                    A bidirectional RNN that captures <b>sequential context</b>. It learns word 
                    dependencies by looking at both the past and future of the token stream.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with m_col3:
            st.markdown("""
            <div class="model-card card-attn">
                <span class="card-tag" style="color: #065F46;">RNN + Attention</span>
                <div class="card-title">JointBiLSTM+Attn</div>
                <p style="font-size: 0.9rem; color: #4b5563;">
                    Adds an <b>Attention mechanism</b> to the BiLSTM, allowing the model to 
                    mathematically focus on specific key tokens like city names or verbs.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with m_col4:
            st.markdown("""
            <div class="model-card card-bert">
                <span class="card-tag" style="color: #1E3A8A;">Transformer</span>
                <div class="card-title">JointBERT</div>
                <p style="font-size: 0.9rem; color: #4b5563;">
                    State-of-the-art <b>Transformer architecture</b>. Uses deep self-attention 
                    and pre-trained embeddings to understand complex linguistic nuances.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Input section
        st.markdown('<div class="section-divider">Model Analysis Workshop</div>', unsafe_allow_html=True)
        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            # Using a label for accessibility but hiding it visually for a clean look
            user_input = st.text_input(
                "Enter Utterance",
                value="book a flight from London to Paris tomorrow",
                placeholder="e.g., show me flights from Boston to New York",
                label_visibility="collapsed"
            )
            
            # Vertical spacing
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            
            run_btn = st.button("🚀 Run Multi-Model Analysis", use_container_width=True)
            
        # Example chips below the main input
        st.markdown("<br>", unsafe_allow_html=True)
        _, example_col, _ = st.columns([1, 3, 1])
        with example_col:
            st.write("✨ **Try these:**")
            ex_cols = st.columns(3)
            if ex_cols[0].button("📍 Distance query", use_container_width=True):
                user_input = "how far is the airport from downtown"
            if ex_cols[1].button("✈️ Airline info", use_container_width=True):
                user_input = "which airlines fly from dallas"
            if ex_cols[2].button("💰 Fare check", use_container_width=True):
                user_input = "cheapest flight to miami"

        # Prepare data for rendering
        model_names = ['Baseline', 'JointBiLSTM', 'JointBiLSTM+Attn', 'JointBERT']
        
        # If button is NOT clicked, we create a 'dummy' results object with empty values
        if not run_btn:
            tokens = user_input.split() if user_input else ["Enter", "a", "query"]
            display_results = {
                name: {
                    'intent': '---', 
                    'conf': 0.0, 
                    'slots': ['O'] * len(tokens), 
                    'attention': [0.0] * len(tokens)
                } for name in model_names
            }
        else:
            # Actual Inference Logic
            tokens = user_input.split()
            display_results = {}
            for name in model_names:
                try:
                    if name == 'Baseline':
                        display_results[name] = predict_baseline(models[name], user_input, vocabs)
                    elif name == 'JointBiLSTM':
                        display_results[name] = predict_bilstm(models[name], user_input, vocabs, device)
                    elif name == 'JointBiLSTM+Attn':
                        display_results[name] = predict_bilstm_attn(models[name], user_input, vocabs, device)
                    elif name == 'JointBERT':
                        display_results[name] = predict_bert(models[name], tokenizer, user_input, vocabs, device)
                except:
                    display_results[name] = None

        # SIDE-BY-SIDE COMPARISON SECTIONS
        st.markdown("---")
        
        # SECTION: INTENT
        st.markdown("""
        ### 🎯 Intent Detection
        There are 21 different unique intent types, all prefixed with `atis_`.
        """)
        st.write("> **How to read:** Compares the predicted category. Higher percentages indicate higher model certainty.")

        # Dictionary of descriptions for Intent
        intent_desc = {
            'Baseline': "Uses a Random Forest classifier on TF-IDF features to categorize the overall goal of the sentence.",
            'JointBiLSTM': "Learns a hidden vector representation of the entire sequence to classify intent.",
            'JointBiLSTM+Attn': "Uses an attention layer to weigh specific 'trigger words' more heavily when determining the intent.",
            'JointBERT': "Leverages the [CLS] token's deep contextual embedding to achieve state-of-the-art classification."
        }

        for model_name in ['Baseline', 'JointBiLSTM', 'JointBiLSTM+Attn', 'JointBERT']:
            if display_results[model_name]:
                st.caption(f"**{model_name}:** {intent_desc[model_name]}")
                components.html(create_model_row_html(model_name, tokens, display_results[model_name], "intent"), height=80)

        # SECTION: SLOTS
        st.markdown("""
        ### 🏷️ Slot Filling (NER)
        There are 121 unique slot labels using BIO tagging scheme.
        """)
        st.write("> **How to read:** Highlights specific entities. Red/Green/Purple/Blue chips represent recognized slots.")

        slot_desc = {
            'Baseline': "Utilizes Conditional Random Fields (CRF) to model dependencies between neighboring labels.",
            'JointBiLSTM': "Uses a Softmax layer over BiLSTM hidden states to predict a BIO tag for every word.",
            'JointBiLSTM+Attn': "Combines sequence context with attention scores to improve entity boundary detection.",
            'JointBERT': "Performs token-level classification using the output of the Transformer's final encoder layer."
        }
        for name in model_names:
            if display_results[name]:
                # Create copies to avoid modifying the original result dictionary
                display_tokens = list(tokens)
                display_slots = list(display_results[name]['slots'])
                
                # SPECIAL HANDLING FOR JointBERT:
                # BERT models typically prepend [CLS]. We check if the first token is [CLS] 
                # or if the slot list is longer than the original input tokens.
                if name == 'JointBERT':
                    # If the predict_bert returns the [CLS] token in the 'slots' list
                    # We slice from index 1 to end.
                    if len(display_slots) > len(tokens):
                        display_slots = display_slots[1:]
                    
                    # If the 'tokens' list for BERT specifically includes '[CLS]' at index 0:
                    display_tokens = display_tokens[1:]

                # Create a modified data object for the visualization
                processed_data = display_results[name].copy()
                processed_data['slots'] = display_slots

                st.caption(f"**{name}:** {slot_desc[name]}")
                components.html(
                    create_model_row_html(name, tokens, processed_data, "slots"), 
                    height=150, 
                    scrolling=False
                )

        # SECTION: ATTENTION
        st.markdown("""
        ### ⚡ Attention & Feature Importance
        """)
        st.write("> **How to read:** Darker purple indicates the 'keywords' the model prioritized for its decision.")

        attn_desc = {
            'Baseline': "Displays 'Feature Importance' scores derived from the Random Forest decision paths.",
            'JointBiLSTM': "Shows gradient-based saliency scores, indicating which words most affected the hidden state.",
            'JointBiLSTM+Attn': "Visualizes the explicit Attention weights calculated by the dedicated attention layer.",
            'JointBERT': "Shows a pooled summary of attention across multiple Transformer heads for the final prediction."
        }
        for name in model_names:
            # Check if the model actually supports attention visualization
            if display_results[name]:
                st.caption(f"**{name}:** {attn_desc[name]}")
                components.html(create_model_row_html(name, tokens, display_results[name], "attention"), height=110)
            elif run_btn:
                st.text(f"Notice: {name} uses internal feature importance rather than attention heads.")

    # Footer Section
    st.markdown("""
        <div class="footer">
            <p>Built with ❤️ as part of FIT5149 - Applied Data Analysis</p>
            <div class="footer-links">
                <a href="https://github.com/archeltaneka" target="_blank">📂 GitHub</a>
                <a href="https://archeltaneka.github.io" target="_blank">🌐 My Website</a>
                <a href="https://linkedin.com/in/archel-taneka-sutanto" target="_blank">🔗 LinkedIn</a>
            </div>
            <p style="font-size: 0.8rem; margin-top: 15px;">
                © 2026 Archel Taneka Sutanto | All rights reserved.
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()