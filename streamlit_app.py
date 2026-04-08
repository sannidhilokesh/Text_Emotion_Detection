import streamlit as st
import torch
import pickle
from transformers import AutoTokenizer, AutoModel
import numpy as np
import time
import os
import gdown
import zipfile
import matplotlib.pyplot as plt

# ================== DOWNLOAD MODEL ==================
MODEL_DIR = "saved_model"

def download_model():
    if not os.path.exists(os.path.join(MODEL_DIR, "model.pt")):
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        file_id = "1W47iBT0hpnHI2yICi1-H2DAA5il0AoU3"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        with st.spinner("🤖 Loading AI model... please wait"):
            gdown.download(url, "model.zip", quiet=True)
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

# ================== MODEL ==================
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

    # strong emotions
    for e, p in zip(labels, probs):
        if p > 0.3:
            results.append((e, p, EMOTION_EMOJI.get(e, "🤷")))

    # fallback top 3
    if not results:
        top_indices = np.argsort(probs)[-3:][::-1]
        for idx in top_indices:
            e = labels[idx]
            p = probs[idx]
            results.append((e, p, EMOTION_EMOJI.get(e, "🤷")))

    results = sorted(results, key=lambda x: x[1], reverse=True)

    return results

# ================== UI ==================
def main():
    st.set_page_config(page_title="Emotion Analysis", page_icon="😊")

    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #667eea, #764ba2); }
    h1 { color: white; text-align: center; }
    .card {
        background: rgba(255,255,255,0.15);
        padding: 10px;
        border-radius: 12px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1>😊 Text Emotion Analysis</h1>", unsafe_allow_html=True)

    model, tokenizer, labels = load_model()

    text = st.text_area("Enter your text:", height=150)

    if st.button("🔍 Analyze Emotions"):
        if text.strip() == "":
            st.warning("Enter text first")
        else:
            with st.spinner("Analyzing..."):
                results = predict(text, model, tokenizer, labels)
                time.sleep(0.5)

            st.subheader("🎭 Detected Emotions:")

            # 🔥 CARDS + % + BAR
            for e, p, emoji in results:
                st.markdown(f"<div class='card'>{emoji} <b>{e}</b> ({p*100:.1f}%)</div>", unsafe_allow_html=True)
                st.progress(float(p))

            # 🔥 DONUT CHART (UNIQUE)
            labels_chart = [e for e, _, _ in results]
            values_chart = [p for _, p, _ in results]

            colors = ['#ff758c', '#667eea', '#42e695', '#f9ca24']

            fig, ax = plt.subplots()
            ax.pie(values_chart, labels=labels_chart, autopct='%1.1f%%',
                   startangle=90, colors=colors[:len(values_chart)])

            centre_circle = plt.Circle((0, 0), 0.60, fc='white')
            fig.gca().add_artist(centre_circle)

            ax.set_title("🎭 Emotion Distribution")

            st.pyplot(fig)

if __name__ == "__main__":
    main()
