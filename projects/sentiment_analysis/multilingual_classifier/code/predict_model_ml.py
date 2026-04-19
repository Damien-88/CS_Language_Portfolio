# MULTILINGUAL SENTIMENT ANALYSIS MODEL PREDICTION

# IMPORTS
# Path Handling
from pathlib import Path

# Model and Vectorizer Loading
import pickle

# Preprocessing Pipeline
from preprocess_ml import preprocess_text


class MultilingualSentimentPredictor:
	# GLOBAL LABEL MAP
	LABEL_MAP = {
		0: "negative",
		1: "neutral",
		2: "positive",
	}

	# LANGUAGE ALIASES
	# Normalize user language inputs to preprocess_ml supported names.
	LANGUAGE_MAP = {
		"en": "english",
		"english": "english",
		"de": "german",
		"german": "german",
		"deutsch": "german",
		"ru": "russian",
		"russian": "russian",
		"русский": "russian",
	}

	def __init__(self):
		# FILE PATHS
		self.base_dir = Path(__file__).resolve().parents[1]  # Base directory.
		self.data_dir = self.base_dir / "data"  # Data folder.
		self.model_path = self.data_dir / "sentiment_model_ml.pkl"  # Saved trained model.
		self.vectorizer_path = self.data_dir / "vectorizer_ml.pkl"  # Saved TF-IDF vectorizer.

		# LOAD MODEL AND VECTORIZER
		with open(self.model_path, "rb") as mf:
			self.model = pickle.load(mf)  # Load trained classifier.

		with open(self.vectorizer_path, "rb") as vf:
			self.vectorizer = pickle.load(vf)  # Load fitted TF-IDF vectorizer.

	def normalize_language(self, language: str = "english") -> str:
		# Normalize free-form language input to a supported key.
		key = str(language).strip().lower()
		return self.LANGUAGE_MAP.get(key, "english")  # Default to English for unknown aliases.

	def proba_by_label(self, processed_text: str) -> dict:
		"""
		Return class probabilities keyed by sentiment label, using model.classes_
		to avoid relying on fixed class order.
		"""
		vector = self.vectorizer.transform([processed_text])  # Convert text to TF-IDF features.
		probs = self.model.predict_proba(vector)[0]  # Get class probabilities.

		by_label = {}  # Label -> probability map.
		# Map each model class index to its sentiment label.
		for cls, prob in zip(self.model.classes_, probs):
			by_label[self.LABEL_MAP.get(int(cls), "unknown")] = float(prob)

		# Ensure all expected labels are present.
		for label in ("negative", "neutral", "positive"):
			by_label.setdefault(label, 0.0)

		return by_label

	def label_from_scores(self, scores: dict, min_confidence: float = 0.45, neutral_margin: float = 0.12) -> str:
		"""
		Convert probability dictionary to final sentiment label.
		"""
		ranked = sorted(scores.items(), key = lambda x: x[1], reverse = True)  # Rank labels by confidence.
		top_label, top_prob = ranked[0]  # Highest-probability label.
		second_prob = ranked[1][1]  # Second-best probability.

		# If confidence is weak or classes are too close, prefer neutral.
		if top_label != "neutral" and (top_prob < min_confidence or (top_prob - second_prob) < neutral_margin):
			return "neutral"

		return top_label

	# SINGLE PREDICTION FUNCTION
	def predict_sentiment(
		self,
		text: str,
		language: str = "english",
		min_confidence: float = 0.45,
		neutral_margin: float = 0.12,
	) -> str:
		normalized_language = self.normalize_language(language)  # Resolve language alias.
		processed_text = preprocess_text(text, language = normalized_language)  # Clean and normalize input text.
		scores = self.proba_by_label(processed_text)  # Get per-label probabilities.
		return self.label_from_scores(scores, min_confidence, neutral_margin)  # Convert scores to final label.

	# PROBABILITY PREDICTION
	def predict_proba(self, text: str, language: str = "english") -> dict:
		normalized_language = self.normalize_language(language)  # Resolve language alias.
		processed_text = preprocess_text(text, language = normalized_language)  # Clean and normalize input text.
		return self.proba_by_label(processed_text)  # Return label probabilities.

	# BATCH PREDICTION FUNCTION
	def predict_batch_detailed(
		self,
		text_list: list,
		language: str = "english",
		min_confidence: float = 0.45,
		neutral_margin: float = 0.12,
	) -> list:
		results = []  # Store batch outputs.
		normalized_language = self.normalize_language(language)  # Resolve language alias once per batch.

		# Process each text independently.
		for text in text_list:
			processed = preprocess_text(text, language = normalized_language)  # Clean and normalize input text.
			scores = self.proba_by_label(processed)  # Get per-label probabilities.
			pred_label = self.label_from_scores(scores, min_confidence, neutral_margin)  # Reuse computed scores.
			results.append(
				{
					"text": text,  # Original input text.
					"prediction": pred_label,  # Predicted sentiment label.
					"probabilities": scores,  # Probability distribution.
				}
			)

		return results


# Module-level predictor instance for convenience functions.
predictor = MultilingualSentimentPredictor()


def predict_sentiment(text, language = "english", min_confidence = 0.45, neutral_margin = 0.12):
	# Wrapper for backward-compatible function-style API.
	return predictor.predict_sentiment(text, language, min_confidence, neutral_margin)


def predict_proba(text, language = "english"):
	# Wrapper for backward-compatible function-style API.
	return predictor.predict_proba(text, language)


def predict_batch_detailed(text_list, language = "english", min_confidence = 0.45, neutral_margin = 0.12):
	# Wrapper for backward-compatible function-style API.
	return predictor.predict_batch_detailed(text_list, language, min_confidence, neutral_margin)


# EXAMPLE USAGE
if __name__ == "__main__":
	# Single prediction examples by language.
	sample_inputs = [
		("I absolutely love this product, it's amazing!", "english"),
		("Ich liebe dieses Produkt, es ist fantastisch!", "german"),
		("Мне очень нравится этот продукт, он отличный!", "russian"),
	]

	print("\nSENTIMENT PREDICTION (SINGLE TEXT)")
	for sample_text, sample_language in sample_inputs:
		result = predict_sentiment(sample_text, language = sample_language)
		proba = predict_proba(sample_text, language = sample_language)

		print(f"\nInput Text: {sample_text}")
		print(f"Language: {sample_language}")
		print(f"Prediction: {result}")
		print("Probabilities:")
		for label, score in proba.items():
			print(f"  {label}: {score:.3f}")

	# Batch prediction example (detailed).
	examples = [
		"This is the worst experience I have ever had.",
		"I really enjoyed this movie!",
		"It was okay, nothing special.",
	]

	batch_results = predict_batch_detailed(examples, language = "english")

	print("\nSENTIMENT PREDICTION (BATCH TEXT)")
	for item in batch_results:
		print(f"\nText: {item['text']}")
		print(f"Prediction: {item['prediction']}")
		print("Probabilities:")
		for label, score in item["probabilities"].items():
			print(f"  {label}: {score:.3f}")
