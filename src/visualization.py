import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
import base64

INTENT_COLORS = {"primary": "#636EFA", "background": "#E5ECF6"}
SLOT_PALETTE = ["#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]
ATTENTION_CMAP = "Purples"

def normalize_attention(attention_weights):
    """Normalize attention weights to [0, 1] range."""
    attention = np.array(attention_weights)
    if attention.max() > attention.min():
        attention = (attention - attention.min()) / (attention.max() - attention.min())
    return attention


def get_color_from_weight(weight, colormap=ATTENTION_CMAP):
    cmap = plt.get_cmap(colormap)
    rgba = cmap(0.1 + weight * 0.7)  # Scale for readability
    return mcolors.rgb2hex(rgba[:3])


def create_model_row_html(model_name, tokens, data, mode):
    """
    Create a single row showing model name and its output.
    
    Args:
        model_name: name of the model
        data: the data to visualize (intent string, slots list, or attention dict)
        visualization_type: "intent", "slots", or "attention"
    
    Returns:
        HTML string with the model row
    """

    theme_colors = {
        "JointBERT": "#1E3A8A",
        "JointBiLSTM+Attn": "#065F46",
        "JointBiLSTM": "#92400E",
        "Baseline": "#374151"
    }
    color = theme_colors.get(model_name, "#1a1a1a")
    
    html = f"""
    <div style="
        display: flex; 
        align-items: stretch; 
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 12px;
        transition: box-shadow 0.2s ease;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    " onmouseover="this.style.boxShadow='0 4px 12px rgba(0,0,0,0.08)'" onmouseout="this.style.boxShadow='none'">
        <div style="
            width: 160px;
            min-width: 160px;
            background: linear-gradient(135deg, {color}15 0%, {color}05 100%);
            border-right: 3px solid {color};
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 16px 12px;
        ">
            <div style="
                font-weight: 600;
                color: {color};
                font-size: 13px;
                text-align: center;
                letter-spacing: 0.3px;
            ">
                {model_name}
            </div>
        </div>
        <div style="
            display: flex; 
            flex-wrap: wrap; 
            gap: 8px; 
            flex-grow: 1;
            padding: 16px 20px;
            align-items: center;
            background: #fafbfc;
        ">
    """
    
    if mode == "intent":
        intent_name = data['intent']
        conf = data['conf']
        html += f"""
            <div style="
                background: linear-gradient(135deg, {color} 0%, {color}dd 100%);
                color: white;
                padding: 12px 24px;
                border-radius: 6px;
                display: flex;
                align-items: center;
                gap: 12px;
                box-shadow: 0 2px 8px {color}30;
            ">
                <span style="font-size: 16px; font-weight: 600; letter-spacing: 0.3px;">{intent_name}</span>
                <span style="
                    background: rgba(255,255,255,0.2);
                    padding: 4px 10px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: 500;
                ">{conf:.1%}</span>
            </div>
        """
    
    elif mode == "slots":
        unique_tags = list(set([s for s in data['slots'] if s != 'O']))
        tag_map = {tag: SLOT_PALETTE[i % len(SLOT_PALETTE)] for i, tag in enumerate(unique_tags)}
        
        for token, slot in zip(tokens, data['slots']):
            bg = tag_map.get(slot, "#f3f4f6")
            txt = "white" if slot != 'O' else "#6b7280"
            border = bg if slot != 'O' else "#e5e7eb"
            
            html += f"""
                <div style="
                    text-align: center;
                    background: white;
                    border-radius: 6px;
                    border: 2px solid {border};
                    padding: 4px;
                    min-width: 50px;
                    transition: transform 0.15s ease;
                " onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0)'">
                    <div style="
                        background: {bg}; 
                        color: {txt}; 
                        padding: 6px 12px; 
                        border-radius: 4px; 
                        font-weight: 500;
                        font-size: 13px;
                        font-family: 'SF Mono', 'Consolas', monospace;
                    ">{token}</div>
                    <div style="
                        font-size: 10px; 
                        color: {bg if slot != 'O' else '#9ca3af'}; 
                        font-weight: 700;
                        margin-top: 4px;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    ">{slot}</div>
                </div>
            """
    
    elif mode == "attention":
        weights = np.array(data['attention'])
        # Normalize weights for color mapping
        norm_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
        
        for token, w, raw_w in zip(tokens, norm_weights, weights):
            bg = get_color_from_weight(w)
            txt = "white" if w > 0.5 else "#374151"
            
            html += f"""
                <div style="
                    text-align: center;
                    background: white;
                    border-radius: 6px;
                    border: 2px solid {bg};
                    padding: 4px;
                    min-width: 50px;
                    transition: all 0.15s ease;
                " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 8px rgba(0,0,0,0.1)'" 
                   onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none'">
                    <div style="
                        background: {bg}; 
                        color: {txt}; 
                        padding: 8px 12px; 
                        border-radius: 4px; 
                        min-width: 50px;
                        font-weight: 500;
                        font-size: 13px;
                        font-family: 'SF Mono', 'Consolas', monospace;
                    ">{token}</div>
                    <div style="
                        font-size: 10px; 
                        color: #6b7280; 
                        font-family: 'SF Mono', 'Consolas', monospace; 
                        margin-top: 4px;
                        font-weight: 600;
                    ">{raw_w:.3f}</div>
                </div>
            """
    
    html += """
        </div>
    </div>
    """
    
    return html


def create_attention_heatmap_html(tokens, attention_weights, title="Attention Weights"):
    """
    Create an HTML visualization of attention weights as a heatmap.
    
    Args:
        tokens: list of token strings
        attention_weights: numpy array of attention weights
        title: title for the visualization
    
    Returns:
        HTML string with the heatmap
    """
    # Normalize attention
    attention = normalize_attention(attention_weights)
    
    # Create HTML
    html = f"""
    <div style="margin: 24px 0;">
        <h4 style="color: #1a1a1a; margin-bottom: 16px; font-weight: 400; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">{title}</h4>
        <div style="display: flex; flex-wrap: wrap; gap: 6px; align-items: center; padding: 20px; background: #fafafa; border-radius: 4px; border: 1px solid #f0f0f0;">
    """
    
    for token, weight in zip(tokens, attention):
        color = get_color_from_weight(weight, colormap='Greys')
        # Calculate text color based on background brightness
        text_color = '#1a1a1a' if weight < 0.5 else '#ffffff'
        
        html += f"""
            <div style="position: relative; display: inline-block;">
                <span style="
                    background-color: {color};
                    color: {text_color};
                    padding: 10px 16px;
                    border-radius: 4px;
                    font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
                    font-size: 13px;
                    font-weight: 400;
                    display: inline-block;
                    border: 1px solid #e0e0e0;
                ">{token}</span>
                <span style="
                    position: absolute;
                    bottom: -20px;
                    left: 50%;
                    transform: translateX(-50%);
                    font-size: 10px;
                    color: #999;
                    white-space: nowrap;
                    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                ">{weight:.3f}</span>
            </div>
        """
    
    html += """
        </div>
    </div>
    """
    
    return html


def create_slot_predictions_html(slot_predictions, title="Slot Predictions"):
    """
    Create an HTML visualization of slot predictions.
    
    Args:
        slot_predictions: list of (token, slot_label) tuples
        title: title for the visualization
    
    Returns:
        HTML string with slot predictions
    """
    # Minimalistic color mapping for different slot types
    slot_colors = {
        'O': '#f5f5f5',
        'B-': '#1a1a1a',
        'I-': '#666666',
    }
    
    html = f"""
    <div style="margin: 24px 0;">
        <h4 style="color: #1a1a1a; margin-bottom: 16px; font-weight: 400; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">{title}</h4>
        <div style="display: flex; flex-wrap: wrap; gap: 6px; padding: 20px; background: #fafafa; border-radius: 4px; border: 1px solid #f0f0f0;">
    """
    
    for token, slot in slot_predictions:
        # Determine color based on slot prefix
        if slot.startswith('B-'):
            color = slot_colors['B-']
            text_color = '#ffffff'
            border_color = '#1a1a1a'
        elif slot.startswith('I-'):
            color = slot_colors['I-']
            text_color = '#ffffff'
            border_color = '#666666'
        else:
            color = slot_colors['O']
            text_color = '#666666'
            border_color = '#e0e0e0'
        
        html += f"""
            <div style="display: inline-block; text-align: center; margin: 2px;">
                <div style="
                    background-color: {color};
                    color: {text_color};
                    padding: 10px 16px;
                    border-radius: 4px 4px 0 0;
                    font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
                    font-size: 13px;
                    font-weight: 400;
                    border: 1px solid {border_color};
                    border-bottom: none;
                ">{token}</div>
                <div style="
                    background-color: #ffffff;
                    color: #666;
                    padding: 6px 12px;
                    border-radius: 0 0 4px 4px;
                    font-size: 10px;
                    border: 1px solid {border_color};
                    border-top: none;
                    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                    text-transform: uppercase;
                    letter-spacing: 0.3px;
                ">{slot}</div>
            </div>
        """
    
    html += """
        </div>
    </div>
    """
    
    return html


def create_intent_prediction_html(intent, confidence=None):
    """
    Create an HTML visualization of intent prediction.
    
    Args:
        intent: predicted intent string
        confidence: optional confidence score
    
    Returns:
        HTML string with intent prediction
    """
    confidence_text = f" · {confidence:.1%}" if confidence is not None else ""
    
    html = f"""
    <div style="margin: 24px 0;">
        <h4 style="color: #1a1a1a; margin-bottom: 16px; font-weight: 400; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">Intent Prediction</h4>
        <div style="
            background: #1a1a1a;
            color: white;
            padding: 24px 32px;
            border-radius: 4px;
            font-size: 20px;
            font-weight: 300;
            text-align: center;
            letter-spacing: 0.3px;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        ">
            {intent}{confidence_text}
        </div>
    </div>
    """
    
    return html


def create_feature_importance_html(tokens, importance_scores, title="Feature Importance"):
    """
    Create an HTML visualization of feature importance (for baseline model).
    Similar to attention heatmap but with different styling.
    
    Args:
        tokens: list of token strings
        importance_scores: numpy array of importance scores
        title: title for the visualization
    
    Returns:
        HTML string with feature importance
    """
    # Normalize importance
    importance = normalize_attention(importance_scores)
    
    html = f"""
    <div style="margin: 24px 0;">
        <h4 style="color: #1a1a1a; margin-bottom: 16px; font-weight: 400; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">{title}</h4>
        <div style="display: flex; flex-wrap: wrap; gap: 6px; align-items: center; padding: 20px; background: #fafafa; border-radius: 4px; border: 1px solid #f0f0f0;">
    """
    
    for token, score in zip(tokens, importance):
        color = get_color_from_weight(score, colormap='Greys')
        text_color = '#1a1a1a' if score < 0.5 else '#ffffff'
        
        html += f"""
            <div style="position: relative; display: inline-block;">
                <span style="
                    background-color: {color};
                    color: {text_color};
                    padding: 10px 16px;
                    border-radius: 4px;
                    font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
                    font-size: 13px;
                    font-weight: 400;
                    display: inline-block;
                    border: 1px solid #e0e0e0;
                ">{token}</span>
                <span style="
                    position: absolute;
                    bottom: -20px;
                    left: 50%;
                    transform: translateX(-50%);
                    font-size: 10px;
                    color: #999;
                    white-space: nowrap;
                    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                ">{score:.3f}</span>
            </div>
        """
    
    html += """
        </div>
    </div>
    """
    
    return html


def create_model_info_html(model_name, description):
    """
    Create an HTML info box for model description.
    
    Args:
        model_name: name of the model
        description: description text
    
    Returns:
        HTML string with model info
    """
    html = f"""
    <div style="
        background: #fafafa;
        border-left: 3px solid #1a1a1a;
        padding: 20px 24px;
        margin: 24px 0;
        border-radius: 4px;
    ">
        <h4 style="margin: 0 0 8px 0; color: #1a1a1a; font-weight: 400; font-size: 16px;">{model_name}</h4>
        <p style="margin: 0; color: #666; line-height: 1.6; font-size: 14px;">{description}</p>
    </div>
    """
    
    return html