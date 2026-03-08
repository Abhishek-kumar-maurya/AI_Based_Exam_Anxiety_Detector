import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Ensure the root project directory is in PYTHONPATH so we can import preprocessing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.preprocess import load_and_preprocess_data, TextPreprocessor

class AnxietyDataset(Dataset):
    def __init__(self, texts, labels, preprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.preprocessor.preprocess(text)
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        return item

def train_model():
    print("Loading data...")
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'anxiety_dataset.csv')
    df = load_and_preprocess_data(dataset_path)
    
    texts = df['text'].tolist()
    labels = df['target'].tolist()
    
    # Split into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    preprocessor = TextPreprocessor()
    
    train_dataset = AnxietyDataset(train_texts, train_labels, preprocessor)
    val_dataset = AnxietyDataset(val_texts, val_labels, preprocessor)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Initializing model...")
    # 0: Low Anxiety, 1: Moderate Anxiety, 2: High Anxiety
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 3
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")
        
    print("Evaluating model...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seq_labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(seq_labels.cpu().numpy())
            
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    print("Saving model to model/bert_anxiety_model...")
    save_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'bert_anxiety_model')
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    preprocessor.tokenizer.save_pretrained(save_path)
    print("Training complete! Model and tokenizer saved.")

if __name__ == "__main__":
    train_model()
