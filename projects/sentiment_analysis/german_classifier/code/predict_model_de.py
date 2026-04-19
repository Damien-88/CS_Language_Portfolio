# GERMAN SENTIMENT ANALYSIS MODEL PREDICTION

# IMPORTS
# Path Handling
from pathlib import Path

# Model and Vectorizer Laoding
import pickle

# Preprocessing pipelin
from preprocess_de import preprocess_text

# FILE PATHS
BASE_DIR = Path(__file__).resolve().parents[1] # Base directory
DATA_DIR = BASE_DIR / "data" # Data folder
MODEL_PATH = DATA_DIR / "sentiment_model_de.pkl" # Saved trained model
VECTORIZER_PATH = DATA_DIR / "vectorizer_de.pkl" # Saved TF-IDF vector

# LOAD MODEL AND VECTORIZER
with open(MODEL_PATH, "rb") as mf:
    model = pickle.load(mf) # Load trained classifier

with open(VECTORIZER_PATH, "rb") as vf:
    vectorizer = pickle.load(vf) # Load TF-IDF vectorizer

# GLOBAL LABEL MAP
LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


def proba_by_label(processed_text):
    """
    Return class probabilities keyed by sentiment label, using model.classes_
    to avoid relying on fixed class order.
    """
    vector = vectorizer.transform([processed_text]) # Vectorize input text
    probs = model.predict_proba(vector)[0] # Predict class probabilities

    by_label = {} # Label -> probability
    
    # Map class indices to sentiment labels.
    for cls, prob in zip(model.classes_, probs):
        by_label[LABEL_MAP.get(int(cls), "unknown")] = float(prob)

    # Ensure all expected labels are present.
    for label in ("negative", "neutral", "positive"):
        by_label.setdefault(label, 0.0)

    return by_label


def label_from_scores(scores, min_confidence = 0.45, neutral_margin = 0.12):
    """
    Convert probability dictionary to final sentiment label.
    """
    ranked = sorted(scores.items(), key = lambda x: x[1], reverse = True) # Highest score first
    top_label, top_prob = ranked[0] # Top prediction
    second_prob = ranked[1][1] # Runner-up score

    # If confidence is weak or classes are too close, prefer neutral.
    if top_label != "neutral" and (top_prob < min_confidence or (top_prob - second_prob) < neutral_margin):
        return "neutral"

    return top_label


# SINGLE PREDICTION FUNCTION
def predict_sentiment(text, min_confidence = 0.45, neutral_margin = 0.12):
    # Preprocess then score the text.
    processed_text = preprocess_text(text) # Clean and normalize input
    scores = proba_by_label(processed_text) # Get per-label scores
    return label_from_scores(scores, min_confidence, neutral_margin)

# PROBABILITY PREDICTION
def predict_proba(text):
    processed_text = preprocess_text(text) # Clean and normalize input
    return proba_by_label(processed_text) # Return full score map

# BATCH PREDICTION FUNCTION
def predict_batch_detailed(text_list, min_confidence = 0.45, neutral_margin = 0.12):
    results = [] # Accumulate per-text outputs

    # Process each text independently.
    for text in text_list:
        processed = preprocess_text(text) # Clean and normalize input
        scores = proba_by_label(processed) # Get per-label scores
        pred_label = label_from_scores(scores, min_confidence, neutral_margin) # Reuse computed scores

        results.append({
            "text": text, # Original text
            "prediction": pred_label, # Final label
            "probabilities": scores # Label probabilities
        })

    return results

# EXAMPLE USAGE
if __name__ == "__main__":
    # Single prediction example
    sample_text = "Ich liebe dieses Produkt, es ist fantastisch!"
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
        "Das ist die schlimmste Erfahrung, die ich je gemacht habe.",
        "Ich habe diesen Film wirklich genossen!",
        "Es war in Ordnung, nichts Besonderes."
    ]

    batch_results = predict_batch_detailed(examples)

    print("\nSENTIMENT PREDICTION (BATCH TEXT)")

    for item in batch_results:
        print(f"\nText: {item['text']}")
        print(f"Prediction: {item['prediction']}")
        print("Probabilities:")
        for label, score in item["probabilities"].items():
            print(f"  {label}: {score:.3f}")