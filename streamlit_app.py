"""
Text Emotion Analysis - Streamlit Web App
Beautiful UI for emotion detection using trained BERT model
"""

import streamlit as st
import torch
import pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np
from PIL import Image
import time
import os

# Configuration
MODEL_DIR = 'saved_model'
MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 128

# Emotion to emoji mapping
EMOTION_EMOJI = {
    'admiration': '😍',
    'amusement': '😂',
    'anger': '😡',
    'annoyance': '😤',
    'approval': '👍',
    'caring': '🤗',
    'confusion': '😕',
    'curiosity': '🤔',
    'desire': '😋',
    'disappointment': '😞',
    'disapproval': '👎',
    'disgust': '🤢',
    'embarrassment': '😳',
    'excitement': '🤩',
    'fear': '😨',
    'gratitude': '🙏',
    'grief': '😢',
    'joy': '😊',
    'love': '😍',
    'nervousness': '😰',
    'optimism': '😄',
    'pride': '😎',
    'realization': '💡',
    'relief': '😌',
    'remorse': '😔',
    'sadness': '😢',
    'surprise': '😲',
    'neutral': '😐'
}

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, 'tokenizer'))
        
        # Load emotion labels
        with open(os.path.join(MODEL_DIR, 'emotion_labels.pkl'), 'rb') as f:
            emotion_labels = pickle.load(f)
        
        # Initialize model architecture
        model = BERTEmotionClassifier(len(emotion_labels), MODEL_NAME)
        
        # Load trained weights
        model_path = os.path.join(MODEL_DIR, 'model.pt')
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        return model, tokenizer, emotion_labels
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure you've trained the model first by running: `python train_model.py`")
        return None, None, None

class BERTEmotionClassifier(torch.nn.Module):
    """BERT-based multi-label emotion classifier"""
    
    def __init__(self, num_labels, model_name):
        super(BERTEmotionClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.dropout(pooled_output)
        return self.classifier(output)

def predict_emotions(text, model, tokenizer, emotion_labels, device='cpu'):
    """Predict emotions for given text"""
    # Tokenize input
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.sigmoid(outputs)
    
    # Get predictions
    probs = probs.cpu().numpy()[0]
    
    # Get top emotions
    results = []
    for emotion, prob in zip(emotion_labels, probs):
        if prob > 0.5:  # Threshold
            results.append({
                'emotion': emotion,
                'probability': float(prob),
                'emoji': EMOTION_EMOJI.get(emotion, '🤷')
            })
    
    # Sort by probability
    results.sort(key=lambda x: x['probability'], reverse=True)
    
    return results

def main():
    # Page config
    st.set_page_config(
        page_title="Emotion Analysis",
        page_icon="😊",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for beautiful design
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 18px;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #764ba2;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 30px;
        font-size: 18px;
        border-radius: 25px;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .emotion-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .subtitle {
        color: white;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown("<h1>😊 Text Emotion Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Discover the emotions hidden in your text using AI</p>", unsafe_allow_html=True)
    
    # Load model
    with st.spinner('Loading AI model...'):
        model, tokenizer, emotion_labels = load_model_and_tokenizer()
    
    if model is None or tokenizer is None or emotion_labels is None:
        st.error("❌ Model not found. Please train the model first by running `python train_model.py`")
        return
    
    # Container for input
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Text input
    text_input = st.text_area(
        "Enter your text here:",
        height=200,
        placeholder="Type or paste any text... For example: 'I'm so excited to go on vacation tomorrow!'",
        key="input_text"
    )
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Emotions", use_container_width=True)
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("## 📊 About")
        st.markdown("""
        This app uses a **BERT-based AI model** to detect emotions in text.
        
        ### 🎯 How it works:
        1. Enter any text in the input box
        2. Click the "Analyze Emotions" button
        3. See detected emotions with confidence scores
        
        ### 💡 Tips:
        - Try different types of text
        - Be creative with your inputs
        - Check the emojis for visual feedback
        
        ### 📚 Emotions detected:
        The model can detect 27 different emotions including joy, sadness, anger, love, and many more!
        """)
    
    # Process and display results
    if analyze_button:
        if not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze!")
        else:
            with st.spinner('🤖 Analyzing emotions... This may take a moment'):
                # Predict emotions
                results = predict_emotions(text_input, model, tokenizer, emotion_labels)
                
                # Add a small delay for smoother UX
                time.sleep(0.5)
            
            # Display results
            if results:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 🎭 Detected Emotions:")
                
                # Create columns for display
                for idx, result in enumerate(results):
                    col1, col2, col3 = st.columns([1, 3, 2])
                    
                    with col1:
                        emoji_size = "font-size: 3rem;"
                        st.markdown(f"<div style='text-align: center; {emoji_size}'>{result['emoji']}</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"**{result['emotion'].title()}**")
                        probability = result['probability']
                        progress = f"<div style='background: linear-gradient(to right, #667eea 0%, #764ba2 {probability*100}%, #f0f0f0 {probability*100}%); height: 10px; border-radius: 5px;'></div>"
                        st.markdown(progress, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"{probability*100:.1f}%")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                
                # Summary card
                st.markdown("<br>", unsafe_allow_html=True)
                st.success(f"✅ Found **{len(results)}** emotion(s) in your text!")
                
            else:
                st.info("🤷 Couldn't detect any strong emotions. Try a different text!")
                
                # Show top 3 predictions even below threshold
                with st.spinner('🤖 Checking all emotions...'):
                    encoding = tokenizer(
                        text_input,
                        truncation=True,
                        padding='max_length',
                        max_length=MAX_LENGTH,
                        return_tensors='pt'
                    )
                    
                    with torch.no_grad():
                        outputs = model(encoding['input_ids'], encoding['attention_mask'])
                        probs = torch.sigmoid(outputs).cpu().numpy()[0]
                    
                    top3_idx = np.argsort(probs)[-3:][::-1]
                    
                    st.markdown("### 💭 Top possible emotions:")
                    for idx in top3_idx:
                        emotion = emotion_labels[idx]
                        prob = probs[idx]
                        emoji = EMOTION_EMOJI.get(emotion, '🤷')
                        st.markdown(f"{emoji} **{emotion.title()}**: {prob*100:.1f}%")

if __name__ == '__main__':
    main()

