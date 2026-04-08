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
    if not os.path.exists(os.path.join(MODEL_DIR, "model.pt")):
        
        os.makedirs(MODEL_DIR, exist_ok=True)  # 🔥 ADD THIS
        
        file_id = "1W47iBT0hpnHI2yICi1-H2DAA5il0AoU3"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        st.write("⬇️ Downloading model... please wait")
        gdown.download(url, "model.zip", quiet=False)
        
        with zipfile.ZipFile("model.zip", 'r') as zip_ref:
            zip_ref.extractall(".")

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
        if p > 0.3:   # 🔥 FIXED THRESHOLD
            results.append((e, p, EMOTION_EMOJI.get(e, "🤷")))
    
    return sorted(results, key=lambda x: x[1], reverse=True), probs

# ================== UI ==================
def main():
    st.set_page_config(page_title="Emotion Analysis", page_icon="😊")

    # 🔥 BEAUTIFUL CSS BACK
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1>😊 Text Emotion Analysis</h1>", unsafe_allow_html=True)
    st.write("Discover the emotions hidden in your text using AI")

    model, tokenizer, labels = load_model()

    text = st.text_area("Enter your text:", height=150)

    if st.button("🔍 Analyze Emotions"):
        if text.strip() == "":
            st.warning("Enter text first")
        else:
            with st.spinner("Analyzing..."):
                results, probs = predict(text, model, tokenizer, labels)
                time.sleep(0.5)

            if results:
                st.subheader("🎭 Detected Emotions:")
                for e, p, emoji in results:
                    st.write(f"{emoji} **{e}** - {p*100:.1f}%")
            else:
                st.info("🤷 Couldn't detect strong emotions")

                # 🔥 SHOW TOP 3 ALWAYS
                top3_idx = np.argsort(probs)[-3:][::-1]
                st.subheader("💭 Top possible emotions:")
                for idx in top3_idx:
                    st.write(f"{EMOTION_EMOJI.get(labels[idx])} {labels[idx]} - {probs[idx]*100:.1f}%")

if __name__ == "__main__":
    main()
