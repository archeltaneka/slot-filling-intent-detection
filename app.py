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
    create_model_info_html
)


st.set_page_config(
    layout="wide",
    page_title="SLU Attention Visualizer",
    page_icon="🔍"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
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
        'Baseline (CRF+RF)': model_loader.load_baseline('files/checkpoints/baseline_model.joblib'),
        'JointBiLSTM': model_loader.load_bilstm('files/checkpoints/jointbilstm_model.pth'),
        'JointBiLSTM+Attn': model_loader.load_bilstm('files/checkpoints/jointbilstm_attn_model.pth', attn=True),
        'JointBERT': model_loader.load_bert('files/checkpoints/bert_model.pth')
    }
    
    return config, models, vocabs, tokenizer, device


def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 SLU Attention Visualizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Visualize attention mechanisms across different slot-filling and intent detection models</p>', unsafe_allow_html=True)
    
    # Load resources
    with st.spinner("Loading models..."):
        config, models, vocabs, tokenizer, device = load_all_resources()
    
    # Input section
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_input(
            "Enter your query:",
            value="book a flight from boston to new york",
            placeholder="Type your query here...",
            help="Enter a natural language query for slot filling and intent detection"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)
    
    # Example queries
    with st.expander("📝 Example Queries"):
        examples = [
            "book a flight from boston to new york",
            "show me flights on monday",
            "i want to fly to paris tomorrow",
            "find me a restaurant in san francisco",
            "what is the weather like today"
        ]
        cols = st.columns(len(examples))
        for idx, (col, example) in enumerate(zip(cols, examples)):
            if col.button(f"Example {idx+1}", key=f"ex_{idx}", use_container_width=True):
                text_input = example
                st.rerun()
    
    if text_input and (analyze_button or True):  # Always show results when there's input
        st.markdown("---")
        st.markdown("## 📊 Model Predictions & Attention Visualizations")
        
        # Create tabs for each model
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔹 Baseline (CRF+RF)",
            "🔹 JointBiLSTM",
            "🔹 JointBiLSTM+Attn",
            "🔹 JointBERT"
        ])
        
        # Tab 1: Baseline Model
        with tab1:
            with st.spinner("Running Baseline model..."):
                try:
                    result = predict_baseline(models['Baseline (CRF+RF)'], text_input, vocabs)
                    
                    # Model info
                    st.markdown(create_model_info_html(
                        "Baseline Model (CRF + Random Forest)",
                        "Uses Conditional Random Fields for slot filling and Random Forest for intent classification. "
                        "The visualization shows CRF marginal probabilities as attention-like scores."
                    ), unsafe_allow_html=True)
                    
                    # Intent prediction
                    components.html(create_intent_prediction_html(result['intent']), height=120, scrolling=False)
                    
                    # Slot predictions
                    components.html(create_slot_predictions_html(result['slots'], "Slot Predictions"), height=180, scrolling=False)
                    
                    # Attention visualization
                    components.html(create_feature_importance_html(
                        result['tokens'],
                        result['attention'],
                        "CRF Marginal Probabilities (Attention-like)"
                    ), height=180, scrolling=False)
                    
                except Exception as e:
                    st.error(f"Error running Baseline model: {str(e)}")
                    st.exception(e)
        
        # Tab 2: JointBiLSTM
        with tab2:
            with st.spinner("Running JointBiLSTM model..."):
                try:
                    result = predict_bilstm(models['JointBiLSTM'], text_input, vocabs, device)
                    
                    # Model info
                    st.markdown(create_model_info_html(
                        "JointBiLSTM Model",
                        "Bidirectional LSTM with two classification heads (slot and intent). "
                        "Uses gradient-based saliency to show which tokens influence the predictions most."
                    ), unsafe_allow_html=True)
                    
                    # Intent prediction
                    components.html(create_intent_prediction_html(result['intent']), height=120, scrolling=False)
                    
                    # Slot predictions
                    components.html(create_slot_predictions_html(result['slots'], "Slot Predictions"), height=180, scrolling=False)
                    
                    # Attention visualization
                    components.html(create_attention_heatmap_html(
                        result['tokens'],
                        result['attention'],
                        "Gradient-Based Saliency (Attention-like)"
                    ), height=180, scrolling=False)
                    
                except Exception as e:
                    st.error(f"Error running JointBiLSTM model: {str(e)}")
                    st.exception(e)
        
        # Tab 3: JointBiLSTM+Attn
        with tab3:
            with st.spinner("Running JointBiLSTM+Attention model..."):
                try:
                    result = predict_bilstm_attn(models['JointBiLSTM+Attn'], text_input, vocabs, device)
                    
                    # Model info
                    st.markdown(create_model_info_html(
                        "JointBiLSTM with Attention",
                        "Bidirectional LSTM with explicit attention mechanism for intent classification. "
                        "The attention weights show how much each token contributes to the intent prediction."
                    ), unsafe_allow_html=True)
                    
                    # Intent prediction
                    components.html(create_intent_prediction_html(result['intent']), height=120, scrolling=False)
                    
                    # Slot predictions
                    components.html(create_slot_predictions_html(result['slots'], "Slot Predictions"), height=180, scrolling=False)
                    
                    # Attention visualization
                    components.html(create_attention_heatmap_html(
                        result['tokens'],
                        result['attention'],
                        "Attention Weights"
                    ), height=180, scrolling=False)
                    
                    # Additional info
                    st.info(f"ℹ️ Attention weights sum: {result['attention'].sum():.4f} (should be ≈1.0)")
                    
                except Exception as e:
                    st.error(f"Error running JointBiLSTM+Attn model: {str(e)}")
                    st.exception(e)
        
        # Tab 4: JointBERT
        with tab4:
            with st.spinner("Running JointBERT model..."):
                try:
                    result = predict_bert(models['JointBERT'], tokenizer, text_input, vocabs, device)
                    
                    # Model info
                    st.markdown(create_model_info_html(
                        "JointBERT Model",
                        "BERT-based model with fine-tuned classification heads. "
                        "Shows averaged multi-head self-attention from all BERT layers, focused on [CLS] token attention."
                    ), unsafe_allow_html=True)
                    
                    # Intent prediction
                    components.html(create_intent_prediction_html(result['intent']), height=120, scrolling=False)
                    
                    # Slot predictions
                    components.html(create_slot_predictions_html(result['slots'], "Slot Predictions (BERT Tokens)"), height=180, scrolling=False)
                    
                    # Attention visualization
                    components.html(create_attention_heatmap_html(
                        result['tokens'],
                        result['attention'],
                        "BERT Attention (Averaged across layers & heads)"
                    ), height=180, scrolling=False)
                    
                    # Additional info
                    st.info(f"ℹ️ BERT uses {len(result['bert_attentions'])} transformer layers with multi-head attention")
                    
                except Exception as e:
                    st.error(f"Error running JointBERT model: {str(e)}")
                    st.exception(e)
        
        # Footer with model comparison
        st.markdown("---")
        st.markdown("### 📈 Model Comparison")
        st.markdown("""
        - **Baseline (CRF+RF)**: Traditional ML approach, shows CRF confidence scores
        - **JointBiLSTM**: Neural approach without explicit attention, uses gradient saliency
        - **JointBiLSTM+Attn**: Adds explicit attention mechanism for better interpretability
        - **JointBERT**: Transformer-based with multi-head self-attention across all layers
        """)


if __name__ == "__main__":
    main()
