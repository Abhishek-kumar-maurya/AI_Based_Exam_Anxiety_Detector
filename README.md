# AI Based Exam Anxiety Detector

An NLP-based full-stack system that analyzes student text input and predicts exam anxiety levels (Low, Moderate, High) using a fine-tuned BERT model, FastAPI backend, and Streamlit frontend UI.

## Project Structure

```text
exam_anxiety_detector/
│
├── dataset/
│   └── anxiety_dataset.csv     # Sample dataset mapping text to mental health labels
├── model/
│   └── bert_anxiety_model/     # Directory where the trained model is saved
├── backend/
│   └── main.py                 # FastAPI application serving the /predict endpoint
├── training/
│   └── train_model.py          # PyTorch script to train the BERT classifier
├── preprocessing/
│   └── preprocess.py           # Functions for formatting text and labels for training
├── frontend/
│   └── app.py                  # Streamlit UI for user interaction
├── utils/
│   └── predictor.py            # Utility class connecting the backend and the ML model
├── requirements.txt            # Project dependencies
└── README.md                   # This documentation
```

## 1. Environment Setup

1. Make sure you have **Python 3.9+** installed.
2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # For Windows:
   venv\Scripts\activate
   
   # For Mac/Linux:
   source venv/bin/activate
   ```
3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Train Model

Before starting the backend API, you must tune the BERT model and save its weights locally.

1. Ensure you're in the project's root directory.
2. Run the training script:
   ```bash
   python training/train_model.py
   ```
> **Note:** This loads `dataset/anxiety_dataset.csv`, maps the labels, tokenizes the text, trains for 3 epochs, outputs metrics (Accuracy, Precision, Recall, F1 Score, Confusion Matrix), and saves everything down to `model/bert_anxiety_model/`.

## 3. Run Backend API

Once the model has finished training, start the inference engine:

```bash
uvicorn backend.main:app --reload
```
The FastAPI instance will boot at `http://localhost:8000`. You can test endpoints via the Swagger interface situated at `http://localhost:8000/docs`.

## 4. Run Frontend Streamlit UI

Keep the backend terminal running. Open a **new terminal window**, activate your virtual environment, and run the Streamlit user interface:

```bash
streamlit run frontend/app.py
```

Streamlit will launch a webpage on your browser (usually `http://localhost:8501`). Here, you can enter text, click "Analyze", and see your colored anxiety level prediction along with actionable study-management tips.
