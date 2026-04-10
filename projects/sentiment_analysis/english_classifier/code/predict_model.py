# ENGLISH SENTIMENT ANALYSIS MODEL PREDICTION

# IMPORTS
# Path Handling
from pathlib import Path

# Model and Vectorizer Laoding
import pickle

# Preprocessing pipelin
from preprocess_en import preprocess_text

# FILE PATHS
BASE_DIR = Path(__file__).resolve().parents[1] # Base directory
DATA_DIR = BASE_DIR / "data" # Data folder
MODEL_PATH = DATA_DIR / "sentiment_model.pkl" # Saved trained model
VECTORIZER_PATH = DATA_DIR / "vectorizer_en.pkl" # Saved TF-IDF vector

# LOAD MODEL AND VECTORIZER
with open(MODEL_PATH, "rb") as mf:
    model = pickle.load(mf)

with open(VECTORIZER_PATH, "rb") as vf:
    vectorizer = pickle.load(vf)

# GLOABL LABEL MAP
LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


# SINGLE PREDICTION FUNCTION
def predict_sentiment(text):
    # Preprocess input text
    processed_text = preprocess_text(text)

    # Convert text into TF_IDF vector
    vector = vectorizer.transform([processed_text])

    # Predict Sentiment
    prediction = model.predict(vector)[0]

    return LABEL_MAP.get(prediction, "unknown")

# PROBABILITY PREDICTION
def predict_proba(text):
    processed_text = preprocess_text(text)
    vector = vectorizer.transform([processed_text])

    probs = model.predict_proba(vector)[0]

    return {
        "negative": probs[0],
        "neutral": probs[1],
        "positive": probs[2]
    }

# BATCH PREDICTION FUNCTION
def predict_batch_detailed(text_list):
    results = []

    for text in text_list:
        processed = preprocess_text(text)
        vector = vectorizer.transform([processed])

        pred = model.predict(vector)[0]
        probs = model.predict_proba(vector)[0]

        results.append({
            "text": text,
            "prediction": LABEL_MAP.get(pred, "unknown"),
            "probabilities": {
                "negative": probs[0],
                "neutral": probs[1],
                "positive": probs[2]
            }
        })

    return results

# EXAMPLE USAGE
if __name__ == "__main__":
    # Single prediction example
    sample_text = "I absolutely love this product, it's amazing!"
    result = predict_sentiment(sample_text)
    proba = predict_proba(sample_text)

    print("\nSENTIMENT PREDICTION (SINGLE TEXT)")
    print(f"Input Text: {sample_text}")
    print(f"Prediction: {result}")
    print("Probabilities:")
    for label, score in proba.items():
        print(f"  {label}: {score:.3f}")

    # Batch Prediction Example (DETAILED)
    examples = [
        "This is the worst experience I have ever had.",
        "I really enjoyed this movie!",
        "It was okay, nothing special."
    ]

    batch_results = predict_batch_detailed(examples)

    print("\nSENTIMENT PREDICTION (BATCH TEXT)")

    for item in batch_results:
        print(f"\nText: {item['text']}")
        print(f"Prediction: {item['prediction']}")
        print("Probabilities:")
        for label, score in item["probabilities"].items():
            print(f"  {label}: {score:.3f}")