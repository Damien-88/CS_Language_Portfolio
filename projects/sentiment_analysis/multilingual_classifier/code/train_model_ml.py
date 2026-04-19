# MULTILINGUAL SENTIMENT ANALYSIS MODEL TRAINING

# IMPORTS
# Data handling
import pandas as pd

# Path handling
from pathlib import Path

# Model persistence
import pickle

# ML models
from sklearn.model_selection import train_test_split  # Split data into train/test sets.
from sklearn.feature_extraction.text import TfidfVectorizer  # Convert text to TF-IDF features.
from sklearn.linear_model import LogisticRegression  # Baseline classifier.
from sklearn.calibration import CalibratedClassifierCV  # Calibrate probability outputs.
from sklearn.metrics import accuracy_score, classification_report  # Evaluate model quality.

# Preprocessing pipeline
from preprocess_ml import preprocess_text


class MultilingualSentimentTrainer:
	"""Trainer for English, German, and Russian sentiment data."""

	# Map sentiment labels to numeric classes used by the classifier.
	LABEL_MAP = {
		"negative": 0,
		"neutral": 1,
		"positive": 2,
	}

	# Normalize language aliases so preprocessing gets supported language names.
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

	def __init__(self, data_file: Path | None = None, default_language: str = "english"):
		# FILE PATHS
		self.base_dir = Path(__file__).resolve().parents[1]  # Base classifier directory.
		self.data_dir = self.base_dir / "data"  # Data folder.

		self.data_file = data_file or (self.data_dir / "train_ml.csv")  # Input training CSV.
		self.model_file = self.data_dir / "sentiment_model_ml.pkl"  # Output model path.
		self.vectorizer_file = self.data_dir / "vectorizer_ml.pkl"  # Output vectorizer path.

		self.default_language = self.normalize_language(default_language)  # Fallback language for rows without language.

		# Classifier and vectorizer objects are populated during training.
		self.vectorizer = None
		self.model = None

	def normalize_language(self, value: str) -> str:
		"""Normalize language aliases to preprocess_ml supported names."""
		key = str(value).strip().lower()  # Normalize input token for map lookup.
		# Default to trainer fallback language when alias is unknown.
		return self.LANGUAGE_MAP.get(key, self.default_language if hasattr(self, "default_language") else "english")

	def load_data(self) -> pd.DataFrame:
		"""Load and validate training data."""
		df = pd.read_csv(self.data_file, encoding = "utf-8")  # Load CSV.

		required_columns = {"text", "sentiment"}  # Core columns required for supervised training.
		missing = required_columns - set(df.columns)  # Detect missing required fields.
		if missing:
			raise ValueError(f"Missing required column(s): {sorted(missing)}")  # Fail fast on invalid schema.

		# Keep only columns required for training plus optional language column.
		keep_cols = ["text", "sentiment"]
		if "language" in df.columns:
			keep_cols.append("language")  # Preserve language if dataset provides it.
		df = df[keep_cols]  # Drop unrelated columns to avoid accidental leakage.

		# Standardize label formatting before mapping to class ids.
		df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()  # Normalize label text.
		df["sentiment"] = df["sentiment"].map(self.LABEL_MAP)  # Convert labels to numeric classes.

		# Drop rows with missing text or unmapped labels.
		df = df.dropna(subset = ["text", "sentiment"])  # Remove unusable training rows.
		df["sentiment"] = df["sentiment"].astype(int)  # Ensure target dtype is integer.

		# Build language column for preprocessing.
		if "language" in df.columns:
			df["language"] = df["language"].apply(self.normalize_language)  # Normalize per-row language tags.
		else:
			df["language"] = self.default_language  # Use fallback language when column is absent.

		print("Dataset size:", df.shape)  # Confirm final row/column count.
		print("Class distribution:\n", df["sentiment"].value_counts())  # Show class balance.
		print("Language distribution:\n", df["language"].value_counts())  # Show language balance.
		return df  # Return validated and standardized dataframe.

	def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Apply language-aware text preprocessing row-by-row."""
		# Keep train-time preprocessing identical to prediction-time preprocessing.
		df["cleaned_text"] = df.apply(
			lambda row: preprocess_text(row["text"], language = row["language"]),  # Route each row through language-specific pipeline.
			axis = 1,  # Apply row-wise because preprocessing depends on row language.
		)
		return df[df["cleaned_text"].str.len() > 0].copy()  # Drop empty post-processed rows.

	def build_vectorizer(self) -> TfidfVectorizer:
		"""Create TF-IDF vectorizer with shared multilingual-friendly settings."""
		# Match tuned settings from language-specific trainers.
		return TfidfVectorizer(
			max_features = 10000,  # Slightly larger vocabulary.
			ngram_range = (1, 2),  # Unigrams and bigrams.
			min_df = 2,  # Reduce noisy one-off terms.
			max_df = 0.9,  # Remove very common terms.
			sublinear_tf = True,  # Use sublinear term frequency scaling.
			token_pattern = r"(?u)\b\w+\b",  # Include single-character words (important for Cyrillic too).
		)

	def build_model(self) -> CalibratedClassifierCV:
		"""Create calibrated logistic-regression model."""
		# Logistic regression for strong baseline.
		base_model = LogisticRegression(
			max_iter = 2000,  # Increase iterations for convergence.
			class_weight = "balanced",  # Handle class imbalance.
			C = 1.5,  # Slightly stronger regularization to prevent overfitting.
			multi_class = "auto",  # Let sklearn choose best multiclass strategy.
		)

		# Calibrate probabilities so confidence scores are more meaningful.
		return CalibratedClassifierCV(
			estimator = base_model,  # Use logistic regression as base model.
			method = "sigmoid",  # Platt scaling for calibration.
			cv = 3,  # 3-fold cross-validation for calibration.
		)

	def train(self) -> None:
		"""Run full training pipeline and save artifacts."""
		df = self.load_data()  # Load and validate input dataset.
		df = self.preprocess_dataframe(df)  # Build cleaned text used for model features.

		# SPLIT DATA
		X = df["cleaned_text"]  # Features.
		y = df["sentiment"]  # Target variable.

		X_train, X_test, y_train, y_test = train_test_split(
			X,
			y,
			test_size = 0.2,  # 20% for testing.
			random_state = 42,  # Reproducibility.
			stratify = y,  # Preserve class balance in both splits.
		)

		# VECTORIZE TEXT TF-IDF
		self.vectorizer = self.build_vectorizer()  # Initialize vectorizer.
		X_train_vec = self.vectorizer.fit_transform(X_train)  # Fit on train split only.
		X_test_vec = self.vectorizer.transform(X_test)  # Transform test split using same vocabulary.

		# TRAIN MODEL
		self.model = self.build_model()  # Build calibrated classifier.
		self.model.fit(X_train_vec, y_train)  # Fit model on training features.

		# EVALUATE MODEL
		y_pred = self.model.predict(X_test_vec)  # Predict class labels on held-out set.
		accuracy = accuracy_score(y_test, y_pred)  # Compute scalar accuracy.

		print("MODEL PERFORMANCE")  # Evaluation summary header.
		print(f"Accuracy: {accuracy:.4f}\n")  # Print formatted accuracy.
		print(classification_report(y_test, y_pred))  # Print full precision/recall/f1 breakdown.

		self.save_artifacts()  # Persist trained artifacts for inference.

	def save_artifacts(self) -> None:
		"""Persist trained model and vectorizer."""
		with open(self.model_file, "wb") as mf:
			pickle.dump(self.model, mf)  # Save trained classifier.

		with open(self.vectorizer_file, "wb") as vf:
			pickle.dump(self.vectorizer, vf)  # Save fitted TF-IDF vectorizer.

		print("\nTraining Complete")  # Completion banner.
		print(f"Model saved to {self.model_file}")  # Output model path.
		print(f"Vectorizer saved to {self.vectorizer_file}")  # Output vectorizer path.


if __name__ == "__main__":
	trainer = MultilingualSentimentTrainer(default_language = "english")  # Default language when dataset omits language column.
	trainer.train()  # Execute end-to-end training run.
