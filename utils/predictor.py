import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

class AnxietyPredictor:
    """
    Wrapper class to load the model and tokenizer and process individual text inferences.
    """
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), '..', 'model', 'bert_anxiety_model')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check if the model directory exists
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found at {model_dir}. Please train the model first.")
            
        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        # Mapping definition
        self.INT_MAP = {
            0: 'Low Anxiety',
            1: 'Moderate Anxiety',
            2: 'High Anxiety'
        }

    def predict(self, text: str):
        # Preprocess
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Calculate probabilities and get the highest one
            probs = F.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)
            
            confidence_val = confidence.item()
            class_idx = predicted_class.item()
            
        anxiety_level = self.INT_MAP.get(class_idx, "Unknown")
        
        return {
            "anxiety_level": anxiety_level,
            "confidence": round(confidence_val, 4)
        }
