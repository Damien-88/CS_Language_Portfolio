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


def _proba_by_label(processed_text):
    """
    Return class probabilities keyed by sentiment label, using model.classes_
    to avoid relying on fixed class order.
    """
    vector = vectorizer.transform([processed_text])
    probs = model.predict_proba(vector)[0]

    by_label = {}
    for cls, prob in zip(model.classes_, probs):
        by_label[LABEL_MAP.get(int(cls), "unknown")] = float(prob)

    for label in ("negative", "neutral", "positive"):
        by_label.setdefault(label, 0.0)

    return by_label


# SINGLE PREDICTION FUNCTION
def predict_sentiment(text, min_confidence=0.45, neutral_margin=0.12):
    # Preprocess input text
    processed_text = preprocess_text(text)
    scores = _proba_by_label(processed_text)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_label, top_prob = ranked[0]
    second_prob = ranked[1][1]

    # If confidence is weak or classes are too close, prefer neutral.
    if top_label != "neutral" and (top_prob < min_confidence or (top_prob - second_prob) < neutral_margin):
        return "neutral"

    return top_label

# PROBABILITY PREDICTION
def predict_proba(text):
    processed_text = preprocess_text(text)
    return _proba_by_label(processed_text)

# BATCH PREDICTION FUNCTION
def predict_batch_detailed(text_list):
    results = []

    for text in text_list:
        processed = preprocess_text(text)
        scores = _proba_by_label(processed)
        pred_label = predict_sentiment(text)

        results.append({
            "text": text,
            "prediction": pred_label,
            "probabilities": scores
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