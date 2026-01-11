import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
import base64


def normalize_attention(attention_weights):
    """Normalize attention weights to [0, 1] range."""
    attention = np.array(attention_weights)
    if attention.max() > attention.min():
        attention = (attention - attention.min()) / (attention.max() - attention.min())
    return attention


def get_color_from_weight(weight, colormap='YlOrRd'):
    """
    Map attention weight to color using matplotlib colormap.
    
    Args:
        weight: float in [0, 1]
        colormap: matplotlib colormap name
    
    Returns:
        RGB color as hex string
    """
    cmap = plt.get_cmap(colormap)
    rgba = cmap(weight)
    return mcolors.rgb2hex(rgba[:3])


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
    <div style="margin: 20px 0;">
        <h4 style="color: #1f77b4; margin-bottom: 10px;">{title}</h4>
        <div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 15px; background: #f8f9fa; border-radius: 8px;">
    """
    
    for token, weight in zip(tokens, attention):
        color = get_color_from_weight(weight, colormap='YlOrRd')
        # Calculate text color based on background brightness
        text_color = '#000000' if weight < 0.6 else '#ffffff'
        
        html += f"""
            <div style="position: relative; display: inline-block;">
                <span style="
                    background-color: {color};
                    color: {text_color};
                    padding: 8px 12px;
                    border-radius: 6px;
                    font-family: 'Courier New', monospace;
                    font-size: 14px;
                    font-weight: 500;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    display: inline-block;
                ">{token}</span>
                <span style="
                    position: absolute;
                    bottom: -20px;
                    left: 50%;
                    transform: translateX(-50%);
                    font-size: 10px;
                    color: #666;
                    white-space: nowrap;
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
    # Color mapping for different slot types
    slot_colors = {
        'O': '#e0e0e0',
        'B-': '#4CAF50',
        'I-': '#81C784',
    }
    
    html = f"""
    <div style="margin: 20px 0;">
        <h4 style="color: #1f77b4; margin-bottom: 10px;">{title}</h4>
        <div style="display: flex; flex-wrap: wrap; gap: 8px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
    """
    
    for token, slot in slot_predictions:
        # Determine color based on slot prefix
        if slot.startswith('B-'):
            color = slot_colors['B-']
            text_color = '#ffffff'
        elif slot.startswith('I-'):
            color = slot_colors['I-']
            text_color = '#ffffff'
        else:
            color = slot_colors['O']
            text_color = '#666666'
        
        html += f"""
            <div style="display: inline-block; text-align: center; margin: 4px;">
                <div style="
                    background-color: {color};
                    color: {text_color};
                    padding: 8px 12px;
                    border-radius: 6px 6px 0 0;
                    font-family: 'Courier New', monospace;
                    font-size: 14px;
                    font-weight: 500;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">{token}</div>
                <div style="
                    background-color: #ffffff;
                    color: #333;
                    padding: 4px 8px;
                    border-radius: 0 0 6px 6px;
                    font-size: 11px;
                    border: 1px solid {color};
                    border-top: none;
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
    confidence_text = f" ({confidence:.2%})" if confidence is not None else ""
    
    html = f"""
    <div style="margin: 20px 0;">
        <h4 style="color: #1f77b4; margin-bottom: 10px;">Intent Prediction</h4>
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            font-size: 24px;
            font-weight: 600;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
    <div style="margin: 20px 0;">
        <h4 style="color: #1f77b4; margin-bottom: 10px;">{title}</h4>
        <div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 15px; background: #f8f9fa; border-radius: 8px;">
    """
    
    for token, score in zip(tokens, importance):
        color = get_color_from_weight(score, colormap='Blues')
        text_color = '#000000' if score < 0.6 else '#ffffff'
        
        html += f"""
            <div style="position: relative; display: inline-block;">
                <span style="
                    background-color: {color};
                    color: {text_color};
                    padding: 8px 12px;
                    border-radius: 6px;
                    font-family: 'Courier New', monospace;
                    font-size: 14px;
                    font-weight: 500;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    display: inline-block;
                ">{token}</span>
                <span style="
                    position: absolute;
                    bottom: -20px;
                    left: 50%;
                    transform: translateX(-50%);
                    font-size: 10px;
                    color: #666;
                    white-space: nowrap;
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
        background: #e3f2fd;
        border-left: 4px solid #2196F3;
        padding: 15px;
        margin: 20px 0;
        border-radius: 4px;
    ">
        <h4 style="margin: 0 0 10px 0; color: #1976D2;">{model_name}</h4>
        <p style="margin: 0; color: #424242; line-height: 1.6;">{description}</p>
    </div>
    """
    
    return html
