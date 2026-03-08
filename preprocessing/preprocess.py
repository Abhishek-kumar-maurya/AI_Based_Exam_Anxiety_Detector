import pandas as pd
from transformers import BertTokenizer

# Map generic mental health labels to our specific anxiety levels
LABEL_MAP = {
    'Normal': 'Low Anxiety',
    'Anxiety': 'Moderate Anxiety',
    'Stress': 'Moderate Anxiety',
    'Depression': 'High Anxiety',
    'Suicidal': 'High Anxiety'
}

# Map anxiety levels to integer labels for the neural network
INT_MAP = {
    'Low Anxiety': 0,
    'Moderate Anxiety': 1,
    'High Anxiety': 2
}

# Reverse mapping for inference
REVERSE_INT_MAP = {v: k for k, v in INT_MAP.items()}

def load_and_preprocess_data(csv_path: str):
    """
    Loads the dataset and maps textual labels to integer classes.
    """
    df = pd.read_csv(csv_path)
    # Ensure no missing values in needed columns
    df = df.dropna(subset=['text', 'label'])
    
    # Map raw labels to Anxiety Levels
    df['anxiety_level'] = df['label'].map(LABEL_MAP)
    
    # Drop unmapped if any
    df = df.dropna(subset=['anxiety_level'])
    
    # Map to integers for model training
    df['target'] = df['anxiety_level'].map(INT_MAP)
    
    return df

class TextPreprocessor:
    """
    Handles tokenization and text formatting using BERT tokenizer.
    """
    def __init__(self, model_name='bert-base-uncased', max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def preprocess(self, text: str):
        """
        Preprocesses a single string.
        """
        # Lowercasing is handled natively by bert-base-uncased, but we ensure string type
        text = str(text)
        
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return encoded
        
    def preprocess_batch(self, texts: list):
        """
        Preprocesses a list of strings.
        """
        texts = [str(t) for t in texts]
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return encoded
