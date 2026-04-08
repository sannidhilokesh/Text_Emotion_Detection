import streamlit as st
import torch
import pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np
import time
import os
import gdown
import zipfile

# ================== DOWNLOAD MODEL ==================
MODEL_DIR = "saved_model"

def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
        file_id = "1W47iBT0hpnHI2yICi1-H2DAA5il0AoU3"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        st.write("⬇️ Downloading model... please wait")
        gdown.download(url, "model.zip", quiet=False)
        
        with zipfile.ZipFile("model.zip", 'r') as zip_ref:
            zip_ref.extractall(".")   # VERY IMPORTANT
        
        st.success("✅ Model downloaded successfully!")

download_model()

# ================== CONFIG ==================
MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 128

# ================== EMOJIS ==================
EMOTION_EMOJI = {
    'admiration': '😍','amusement': '😂','anger': '😡','annoyance': '😤',
    'approval': '👍','caring': '🤗','confusion': '😕','curiosity': '🤔',
    'desire': '😋','disappointment': '😞','disapproval': '👎','disgust': '🤢',
    'embarrassment': '😳','excitement': '🤩','fear': '😨','gratitude': '🙏',
    'grief': '😢','joy': '😊','love': '😍','nervousness': '😰',
    'optimism': '😄','pride': '😎','realization': '💡','relief': '😌',
    'remorse': '😔','sadness': '😢','surprise': '😲','neutral': '😐'
}

# ================== MODEL CLASS ==================
class BERTEmotionClassifier(torch.nn.Module):
    def __init__(self, num_labels, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.dropout(pooled_output)
        return self.classifier(output)

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "tokenizer"))
    
    with open(os.path.join(MODEL_DIR, "emotion_labels.pkl"), "rb") as f:
        labels = pickle.load(f)
    
    model = BERTEmotionClassifier(len(labels), MODEL_NAME)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model.pt"), map_location="cpu"))
    model.eval()
    
    return model, tokenizer, labels

# ================== PREDICT ==================
def predict(text, model, tokenizer, labels):
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(encoding['input_ids'], encoding['attention_mask'])
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    results = []
    for e, p in zip(labels, probs):
        if p > 0.5:
            results.append((e, p, EMOTION_EMOJI.get(e, "🤷")))
    
    return sorted(results, key=lambda x: x[1], reverse=True)

# ================== UI ==================
def main():
    st.set_page_config(page_title="Emotion Analysis", page_icon="😊")
    
    st.title("😊 Text Emotion Analysis")
    st.write("Detect emotions from text using AI")
    
    model, tokenizer, labels = load_model()
    
    text = st.text_area("Enter your text:")
    
    if st.button("Analyze"):
        if text.strip() == "":
            st.warning("Enter text first")
        else:
            with st.spinner("Analyzing..."):
                results = predict(text, model, tokenizer, labels)
                time.sleep(0.5)
            
            if results:
                st.subheader("Detected Emotions:")
                for e, p, emoji in results:
                    st.write(f"{emoji} **{e}** - {p*100:.1f}%")
            else:
                st.info("No strong emotions detected")

if __name__ == "__main__":
    main()
