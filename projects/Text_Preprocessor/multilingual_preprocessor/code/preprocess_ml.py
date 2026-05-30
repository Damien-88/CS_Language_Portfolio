# Multilingual Text Preprocessing Pipeline
# Import necessary libraries
import re
import unicodedata
from pathlib import Path
from typing import List, Dict
from utils_ml import read_text_file, write_text_file, create_db, insert_rows

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem.snowball import RussianStemmer, GermanStemmer
from nltk import word_tokenize

class MultilingualPreprocessor:
    def __init__(self, language = "english", country_code = "en"):
        self.language = language.lower()
        self.country_code = country_code.lower()

        if language.lower() == "deutsch":
            self.language = "german"
        elif language.lower() == "русский":
            self.language = "russian"
        elif language.lower() != "english" and language.lower() != "german" and language.lower() != "russian":
            print(f"This preprocessor has not yet been optimize for {language}.")
            print("Defaulting to English.")
        
        # Attempt to import spaCy and load the model
        try:
            import spacy
            if self.language == "english":
                self.nlp = spacy.load("en_core_web_sm")
            else:
                spacy_model = f"{self.country_code}_core_news_sm"
                self.nlp = spacy.load(spacy_model)
        except Exception:
            self.nlp = None

        # Check NLTK resources
        # Run nltk.download("punkt"), nltk.download("stopwords") if not already downloaded
        self.stopwords = set(stopwords.words(self.language))

        if self.language == "german":
            self.stemmer = GermanStemmer()
        elif self.language == "russian":
            self.stemmer = RussianStemmer()
        else:  # English
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
            self.contractions = {
                "n't": "not",
                "'re": "are",
                "'s": "is",
                "'d": "would",
                "'ll": "will",
                "'ve": "have",
                "'m": "am"
            }
        
        self.DB_PATH = Path(__file__).resolve().parents[1] / "data" / f"processed_results_{self.country_code}.db"

    # Clean
    def clean_text(self, text):
        # Normalize unicode characters
        text = unicodedata.normalize("NFKC", text)
        text = text.lower()

        # Remove URLs and mentions
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"@\w+", "", text)

        if self.language == "english":
            # Remove punctuation
            text = re.sub(r"[^a-z0-9\s]", " ", text)
        elif self.language == "german":
            # Remove punctuation (Keep German Characters)
            text = re.sub(r"[^a-z0-9\säöüß]", " ", text)
        elif self.language == "russian":
            # Remove punctuation (Keep Cyrillic Characters and Numbers)
            text = re.sub(r"[^а-яё0-9\s]", " ", text)
        
        # Remove whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # Tokenize (Word)
    def tokenize_text(self, text):
        # Use spaCy if available
        if self.nlp:
            doc = self.nlp(text)

            return [t.text for t in doc]
        # Fallback
        else:
            return word_tokenize(text, self.language)

    # Merge Contractions
    def merge_contractions(self, tokens):
        # Initialize a list to store merged tokens
        merged_tokens = []
        # Flag to skip the next token if already merged
        skip_next = False

        # Iterate over each token with its index
        for i, t in enumerate(tokens):
            # If previous iteration merged this token, skip it
            if skip_next:
                skip_next = False
                continue
            # If current token is a contraction and not the first token
            if t in self.contractions and i > 0:
                # Merge with previous token and add to result
                merged_tokens.append(tokens[i - 1] + self.contractions[t])
            # If next token is a contraction
            if i + 1 < len(tokens) and tokens[i + 1] in self.contractions:
                # Merge current token with contraction and add to result
                merged_tokens.append(tokens[i] + self.contractions[tokens[i + 1]])
                # Set flag to skip next token
                skip_next = True
            else:
                # Otherwise, add current token as is
                merged_tokens.append(t)

        # Return the list of merged tokens
        return merged_tokens

    # Remove Stopwords
    def remove_stopwords(self, tokens):
        if self.language == "english":
            tokens = self.merge_contractions(tokens)

        return [token for token in tokens if token not in self.stopwords]

    # Stem Tokens
    def stem_tokens(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]

    # Lemmatize Tokens
    def lemmatize_tokens(self, tokens):
        # Use spaCy lammas if abvailable
        if self.nlp:
            doc = self.nlp(" ".join(tokens))
            return [token.lemma_ for token in doc]
        # Use NLTK as fallback
        elif not self.nlp and self.language == "english":
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        # No NLTK as fallback return message and tokens instead
        else:
            print("spaCy not available, returning original tokens as lemmas.")
            return tokens

    # File Processer
    def process_file(self, input_path):
        # Read all lines from the input file
        lines = read_text_file(input_path)
        # List to store processed rows
        processed_rows = []

        # Process each line in the file
        for line in lines:
            cleaned = self.clean_text(line)           # Clean the text
            tokens = self.tokenize_text(cleaned)      # Tokenize the cleaned text
            tokens = self.remove_stopwords(tokens)    # Remove stopwords from tokens
            lemmas = self.lemmatize_tokens(tokens)    # Lemmatize the tokens

            # Append processed data as a dictionary
            processed_rows.append(
                {
                    "original": line,            # Original line
                    "cleaned": cleaned,          # Cleaned text
                    "tokens": tokens,            # List of tokens
                    "lemmas": lemmas             # List of lemmas
                }
            )
        
        # Create DB and insert rows with utils.py
        create_db(self.DB_PATH, table_name = f"processed_{self.country_code}")
        insert_rows(self.DB_PATH, processed_rows, table_name = f"processed_{self.country_code}")

        # Preview the first 5 processed rows
        for row in processed_rows[:5]:
            print(f"Original: {row['original']}")    # Print original text
            print(f"Cleaned: {row['cleaned']}")      # Print cleaned text
            print(f"Tokens: {row['tokens']}")        # Print tokens
            print(f"Lemmas: {row['lemmas']}")        # Print lemmas
            print("-" * 40)                          # Print separator

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description = "Multilingual Text Preprocessor")
    parser.add_argument("--in", dest = "input", default = "../data/sample_ml.txt", help = "Input file path")
    parser.add_argument("--lang", dest = "language", default = "english", help = "Language (english/german/russian)")
    parser.add_argument("--code", dest = "code", default = "en", help = "Country code (en/de/ru)")

    args = parser.parse_args()

    preprocessor = MultilingualPreprocessor(language = args.language, country_code = args.code)
    preprocessor.process_file(args.input)