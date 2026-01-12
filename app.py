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
    st.set_page_config(layout="wide", page_title="SLU Intelligence Dashboard", page_icon="🧠")
    st.markdown("""
        <style>
        .main { background-color: #F8F9FB; }
        .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .result-card { background-color: white; padding: 20px; border-radius: 12px; border: 1px solid #E6E9EF; }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-title">🧠 SLU Multi-Model Intelligence Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Compare intent detection and slot filling across multiple neural architectures</p>', unsafe_allow_html=True)
    
    # Load resources
    with st.spinner("Loading models..."):
        config, models, vocabs, tokenizer, device = load_all_resources()
    
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
    
    # Example queries
    with st.expander("💡 Try Example Queries"):
        examples = [
            "book a flight from London to Paris tomorrow",
            "show me restaurants near times square",
            "what's the weather in seattle this weekend",
            "i need a hotel in tokyo for 3 nights"
        ]
        cols = st.columns(4)
        for idx, (col, example) in enumerate(zip(cols, examples)):
            if col.button(f"Example {idx+1}", key=f"ex_{idx}", use_container_width=True):
                user_input = example
                st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if run_btn and user_input:
        st.markdown("<br>", unsafe_allow_html=True)
        
        tab_intent, tab_slots, tab_attn = st.tabs([
            "🎯 Intent Predictions", 
            "🏷️ Slot Predictions", 
            "⚡ Attention & Features"
        ])

        # Run all models
        tokens = user_input.split()
        results = {}

        try:
            results['Baseline'] = predict_baseline(models['Baseline'], user_input, vocabs)
        except Exception as e:
            st.error(f"Error running Baseline model: {str(e)}")
            results['Baseline'] = None
        
        try:
            results['JointBiLSTM'] = predict_bilstm(models['JointBiLSTM'], user_input, vocabs, device)
        except Exception as e:
            st.error(f"Error running JointBiLSTM model: {str(e)}")
            results['JointBiLSTM'] = None
        
        try:
            results['JointBiLSTM+Attn'] = predict_bilstm_attn(models['JointBiLSTM+Attn'], user_input, vocabs, device)
        except Exception as e:
            st.error(f"Error running JointBiLSTM+Attn model: {str(e)}")
            results['JointBiLSTM+Attn'] = None
        
        try:
            results['JointBERT'] = predict_bert(models['JointBERT'], tokenizer, user_input, vocabs, device)
        except Exception as e:
            st.error(f"Error running JointBERT model: {str(e)}")
            results['JointBERT'] = None

        with tab_intent:
            for model_name in ['Baseline', 'JointBiLSTM', 'JointBiLSTM+Attn', 'JointBERT']:
                if results[model_name]:
                    components.html(
                        create_model_row_html(
                            model_name,
                            tokens,
                            results[model_name],
                            "intent"
                        ),
                        height=80,
                        scrolling=False
                    )

        with tab_slots:
            st.markdown('<div class="section-header">Slot Filling (Named Entity Recognition)</div>', unsafe_allow_html=True)
            
            # Legend for slot colors
            st.markdown("""
            <div class="legend-box">
                <div class="legend-title">Slot Type Legend</div>
                <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px;">
                    <div class="legend-item">
                        <div class="legend-color" style="background: #EF553B;"></div>
                        <span>Origin/Departure</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #00CC96;"></div>
                        <span>Destination</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #AB63FA;"></div>
                        <span>Date/Time</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #FFA15A;"></div>
                        <span>Entity Type 4</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #f3f4f6;"></div>
                        <span>Outside (O)</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            for model_name in ['Baseline', 'JointBiLSTM', 'JointBiLSTM+Attn', 'JointBERT']:
                if results[model_name]:
                    components.html(
                        create_model_row_html(
                            model_name,
                            tokens,
                            results[model_name],
                            "slots"
                        ),
                        height=150,
                        scrolling=False
                    )

        with tab_attn:
            st.markdown('<div class="section-header">Attention Weights & Feature Importance</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="legend-box">
                <div class="legend-title">Interpreting Attention Scores</div>
                <p style="margin: 0; font-size: 13px; color: #6b7280; line-height: 1.6;">
                    Darker purple indicates higher attention weight. These scores reveal which tokens 
                    the model focuses on when making predictions. Values are normalized between 0 and 1.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            for model_name in ['Baseline', 'JointBiLSTM', 'JointBiLSTM+Attn', 'JointBERT']:
                if results[model_name]:
                    components.html(
                        create_model_row_html(
                            model_name,
                            tokens,
                            results[model_name],
                            "attention"
                        ),
                        height=120,
                        scrolling=False
                    )
        
        # Model comparison info
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📊 Model Architecture Comparison"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **JointBERT**
                - Transformer-based pre-trained model
                - Multi-head self-attention mechanism
                - Best for context understanding
                
                **JointBiLSTM+Attn**
                - BiLSTM with attention layer
                - Explicit attention weights
                - Good interpretability
                """)
            with col2:
                st.markdown("""
                **JointBiLSTM**
                - Bidirectional LSTM architecture
                - Gradient-based saliency
                - Efficient inference
                
                **Baseline**
                - CRF for slots + Random Forest for intent
                - Traditional ML approach
                - Fast and lightweight
                """)


if __name__ == "__main__":
    main()