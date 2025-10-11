# English Text Preprocessing Pipeline
# Import necessary libraries
import re
import unicodedata
from pathlib import Path
from typing import List, Dict
from utils import read_text_file, write_text_file, create_db, insert_rows

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize

# Attempt to import spaCy and load the English model
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

# Check NLTK resources
# Run nltk.download("punkt"), nltk.download("stopwords"), nltk.download("wordnet") if not already downloaded
STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "processed_results.db"

# Contraction list
CONTRACTIONS = {
    "n't": "not",
    "'re": "are",
    "'s": "is",
    "'d": "would",
    "'ll": "will",
    "'ve": "have",
    "'m": "am"
}

# Clean
def clean_text(text):
    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    # Remove URLs and mentions
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    
    # Remove whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Tokenize (Word)
def tokenize_text(text):
    # Use spaCy if available
    if nlp:
        doc = nlp(text)

        return [t.text for t in doc]
    # Use NLTK as fallback
    else:
        return word_tokenize(text)

# Merge Contractions
def merge_contractions(tokens):
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
        if t in CONTRACTIONS and i > 0:
            # Merge with previous token and add to result
            merged_tokens.append(tokens[i - 1] + CONTRACTIONS[t])
        # If next token is a contraction
        if i + 1 < len(tokens) and tokens[i + 1] in CONTRACTIONS:
            # Merge current token with contraction and add to result
            merged_tokens.append(tokens[i] + CONTRACTIONS[tokens[i + 1]])
            # Set flag to skip next token
            skip_next = True
        else:
            # Otherwise, add current token as is
            merged_tokens.append(t)

    # Return the list of merged tokens
    return merged_tokens
            

# Remove Stopwords
def remove_stopwords(tokens):
    return [token for token in tokens if token not in STOPWORDS]

# Stem Tokens
def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]

# Lemmatize Tokens
def lemmatize_tokens(tokens):
    # Use spaCy lammas if abvailable
    if nlp:
        doc = nlp(" ".join(tokens))
        return [token.lemma_ for token in doc]
    # Use NLTK as fallback
    else:
        return [lemmatizer.lemmatize(token) for token in tokens]

# File Processer
def process_file(input_path):
    # Read all lines from the input file
    lines = read_text_file(input_path)
    # List to store processed rows
    processed_rows = []

    # Process each line in the file
    for line in lines:
        cleaned = clean_text(line)           # Clean the text
        tokens = tokenize_text(cleaned)      # Tokenize the cleaned text
        tokens = remove_stopwords(tokens)    # Remove stopwords from tokens
        lemmas = lemmatize_tokens(tokens)    # Lemmatize the tokens

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
    create_db(DB_PATH)
    insert_rows(DB_PATH, processed_rows)

    # Preview the first 5 processed rows
    for row in processed_rows[:5]:
        print(f"Original: {row['original']}")    # Print original text
        print(f"Cleaned: {row['cleaned']}")      # Print cleaned text
        print(f"Tokens: {row['tokens']}")        # Print tokens
        print(f"Lemmas: {row['lemmas']}")        # Print lemmas
        print("-" * 40)                          # Print separator

# Main Execution
if __name__ == "__main__":
    import argparse
    
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description = "English Text Preprocessor")
    # Add input file argument
    parser.add_argument(
        "--in", dest = "input", 
        default = "../data/sample_en.txt", 
        help = "input file path"
    )
    # Parse command-line arguments
    args = parser.parse_args()
    # Process the input file
    process_file(args.input)