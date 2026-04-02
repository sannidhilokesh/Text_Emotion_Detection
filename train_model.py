"""
Text Emotion Analysis - Model Training Script
Uses BERT for multi-label emotion classification
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MODEL_NAME = 'distilbert-base-uncased'  # Faster alternative to bert-base-uncased

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    # Convert to string
    text = str(text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def load_and_preprocess_data(file_path='goemotions.csv'):
    """Load and preprocess the dataset"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # Clean text column
    print("Cleaning text...")
    df['text'] = df['text'].apply(clean_text)
    
    # Get emotion columns (all columns except 'text')
    emotion_columns = [col for col in df.columns if col != 'text']
    
    print(f"Found {len(emotion_columns)} emotion categories:")
    print(f"Emotions: {', '.join(emotion_columns)}")
    
    # Remove rows with empty text
    df = df[df['text'].str.len() > 0]
    
    print(f"Total samples: {len(df)}")
    
    return df, emotion_columns

class EmotionDataset(Dataset):
    """Dataset class for emotion classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

class BERTEmotionClassifier(nn.Module):
    """BERT-based multi-label emotion classifier"""
    
    def __init__(self, num_labels, model_name):
        super(BERTEmotionClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.dropout(pooled_output)
        return self.classifier(output)

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Training')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(data_loader)

def evaluate(model, data_loader, loss_fn, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            
            total_loss += loss.item()
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            predictions.extend(probs.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions)
    actual_labels = np.array(actual_labels)
    
    # Threshold predictions
    predicted_labels = (predictions > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(actual_labels, predicted_labels)
    f1_macro = f1_score(actual_labels, predicted_labels, average='macro', zero_division=0)
    f1_micro = f1_score(actual_labels, predicted_labels, average='micro', zero_division=0)
    precision = precision_score(actual_labels, predicted_labels, average='macro', zero_division=0)
    recall = recall_score(actual_labels, predicted_labels, average='macro', zero_division=0)
    hamming = hamming_loss(actual_labels, predicted_labels)
    
    return total_loss / len(data_loader), accuracy, f1_macro, f1_micro, precision, recall, hamming

def save_model(model, tokenizer, emotion_labels, model_dir='saved_model'):
    """Save the trained model and tokenizer"""
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Saving model to {model_dir}...")
    
    # Save model state
    model_path = os.path.join(model_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)
    
    # Save tokenizer
    tokenizer_path = os.path.join(model_dir, 'tokenizer')
    tokenizer.save_pretrained(tokenizer_path)
    
    # Save emotion labels
    labels_path = os.path.join(model_dir, 'emotion_labels.pkl')
    with open(labels_path, 'wb') as f:
        pickle.dump(emotion_labels, f)
    
    print("Model saved successfully!")

def main():
    # Load and preprocess data
    df, emotion_columns = load_and_preprocess_data()
    
    # Prepare data
    texts = df['text'].values
    labels = df[emotion_columns].values
    
    print(f"\nLabel distribution:")
    print(f"Total labels per emotion: {labels.sum(axis=0)}")
    
    # Split data
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Initialize tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    train_dataset = EmotionDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    test_dataset = EmotionDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    print(f"\nInitializing model: {MODEL_NAME}")
    model = BERTEmotionClassifier(num_labels=len(emotion_columns), model_name=MODEL_NAME)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\nStarting training...")
    best_f1 = 0
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        test_loss, accuracy, f1_macro, f1_micro, precision, recall, hamming = evaluate(
            model, test_loader, criterion, device
        )
        
        print(f"\nTest Metrics:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score (Macro): {f1_macro:.4f}")
        print(f"  F1 Score (Micro): {f1_micro:.4f}")
        print(f"  Precision (Macro): {precision:.4f}")
        print(f"  Recall (Macro): {recall:.4f}")
        print(f"  Hamming Loss: {hamming:.4f}")
        
        # Save best model
        if f1_macro > best_f1:
            best_f1 = f1_macro
            print(f"\nNew best F1 score! Saving model...")
            save_model(model, tokenizer, emotion_columns)
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)

if __name__ == '__main__':
    main()

