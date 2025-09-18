# File I/O Helpers
from pathlib import Path

# Read text file and return lines
def read_text_file(file_path):
    path = Path(file_path)

    return path.read_text(encoding = "utf-8").splitlines()

# Write line to text file
def write_text_file(file_path, lines):
    path = Path(file_path)

    path.write_text("\n".join(lines), encoding = "utf-8")

# Database Helpers
import sqlite3

# Create a SQLite database and a table for processed text data
def create_db(db_path, table_name = "processed"):
    # Connect to the SQLite database at the given path
    conn = sqlite3.connect(db_path)
    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()
    # Execute SQL to create the table if it doesn't exist
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,  
            original_text TEXT,
            cleaned_text TEXT,
            tokens TEXT,
            lemmas TEXT
        )
        """
    )
    # Commit the changes to the database
    conn.commit()
    # Close the database connection
    conn.close()

# Insert multiple rows of processed text data into the database
def insert_rows(db_path, rows, table_name = "processed"):
    # Connect to the SQLite database at the given path
    conn = sqlite3.connect(db_path)
    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()

    # Iterate over each row of data to insert
    for row in rows:
        # Execute SQL to insert the row into the table
        cursor.execute(
            f"""
            INSERT INTO {table_name} (original_text, cleaned_text, tokens, lemmas)
            VALUES (?, ?, ?, ?)
            """,
            (
                row["original"],                # Original text
                row["cleaned"],                 # Cleaned text
                " ".join(row["tokens"]),        # Tokens joined as a space-separated string
                " ".join(row["lemmas"])         # Lemmas joined as a space-separated string
            )
        )

    # Commit the changes to the database
    conn.commit()
    # Close the database connection
    conn.close()