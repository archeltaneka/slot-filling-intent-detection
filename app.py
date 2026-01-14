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
    
    /* Info styling */
    .stAlert {
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        background: #f9fafb;
    }
    
    /* Legend box */
    .legend-box {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    
    .legend-title {
        font-size: 14px;
        font-weight: 600;
        color: #374151;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .legend-item {
        display: inline-flex;
        align-items: center;
        margin-right: 16px;
        font-size: 13px;
        color: #6b7280;
    }
    
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 4px;
        margin-right: 6px;
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
    st.markdown('<h1 class="main-title">🧠 Slot Filling & Intent Detection Multi-Model Intelligence</h1>', unsafe_allow_html=True)
    
    tab_intro, tab_analysis = st.tabs(["📄 Project Background", "🚀 Live Model Analysis"])

    # Brief Description and Instructions
    with tab_intro:
        # Submission Context
        st.markdown("""
        ### Welcome!
        In this project, we analyze and compare different approaches for **Slot Filling** and **Intent Detection**, 
        fundamental tasks in Natural Language Understanding (NLU).
        
        > **Academic Submission:** This project was submitted for **FIT5149 - Applied Data Analysis** as part of a Master's degree at **Monash University**.
        """)
        
        # Dataset Section (Compressed)
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
            <div style="border: 1px solid #ddd; padding: 15px; border-radius: 10px; height: 100%;">
                <h4>Baseline</h4>
                <p style='font-size: 0.85em; color: #555;'><b>Type:</b> Statistical ML</p>
                <p style='font-size: 0.9em;'>Uses a combination of Random Forest for intent and CRF for slots. Fast and lightweight, but struggles with complex word dependencies.</p>
            </div>
            """, unsafe_allow_html=True)

        with m_col2:
            st.markdown("""
            <div style="border: 1px solid #ddd; padding: 15px; border-radius: 10px; height: 100%;">
                <h4>JointBiLSTM</h4>
                <p style='font-size: 0.85em; color: #555;'><b>Type:</b> Recurrent Neural Net</p>
                <p style='font-size: 0.9em;'>Processes text in both directions. The architecture introduces two heads, one for intent and one for slots. Good at capturing sequential context but treats all words with equal importance.</p>
            </div>
            """, unsafe_allow_html=True)

        with m_col3:
            st.markdown("""
            <div style="border: 1px solid #ddd; padding: 15px; border-radius: 10px; height: 100%;">
                <h4>BiLSTM+Attn</h4>
                <p style='font-size: 0.85em; color: #555;'><b>Type:</b> RNN with Attention</p>
                <p style='font-size: 0.9em;'>Similar with the JointBILSTM model, but it includes an <b>Attention Mechanism</b> that helps the model 'focus' on key trigger words like verbs or city names.</p>
            </div>
            """, unsafe_allow_html=True)

        with m_col4:
            st.markdown("""
            <div style="border: 1px solid #ddd; padding: 15px; border-radius: 10px; height: 100%;">
                <h4>JointBERT</h4>
                <p style='font-size: 0.85em; color: #555;'><b>Type:</b> Transformer</p>
                <p style='font-size: 0.9em;'>State-of-the-art model pre-trained on billions of words. High accuracy on ambiguous queries due to deep context awareness.</p>
            </div>
            """, unsafe_allow_html=True)

        # Input section
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Input Utterance",
                "book a flight from London to Paris tomorrow",
                label_visibility="collapsed",
                placeholder="Enter your utterance here..."
            )
        with col2:
            run_btn = st.button("🚀 Analyze", use_container_width=True)

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
        st.write("> **How to read:** Highlights specific entities. Red/Green/Purple chips represent recognized slots.")

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
                    # Option 1: If your predict_bert returns the [CLS] token in the 'slots' list
                    # We slice from index 1 to end.
                    if len(display_slots) > len(tokens):
                        display_slots = display_slots[1:]
                    
                    # If your 'tokens' list for BERT specifically includes '[CLS]' at index 0:
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


if __name__ == "__main__":
    main()