# File I/O Helpers
from pathlib import Path
import sqlite3

# Read text file and return lines
def read_text_file(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return path.read_text(encoding = "utf=8").splitlines()

# Write line to text file
def write_text_file(file_path, lines):
    path = Path(file_path)
    path.write_text("\n".join(lines), encoding = "utf-8")

# Create SQLite DB and table for processed data
def create_db(db_path, table_name = "processed_data"):
    # Connect to SQLite DB at given path
    # Wrap to auto close connection
    with sqlite3.connect(db_path) as conn:
        # Connect cursor object ot exectue SQL commands
        cursor = conn.cursor()
        # Execute SQL to create table if not exists
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
        # Commit changes to DB
        conn.commit()

# Insert multiple rows of processed data into DB
def insert_rows(db_path, rows, table_name = "processed"):
    # Connect to SQLite database at given path
    # Wrap to auto close connection
    with sqlite3.connect(db_path) as conn:
        # Create cursor object to execctue SQL commands
        cursor = conn.cursor()

        # Iterate over each row of data to insert
        for row in rows:
            # Execute SQL to indsert row into table
            cursor.execute(f"""
                INSERT INTO {table_name} (
                original_text,
                cleaned_text,
                tokens,
                lemmas
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    row["original"],            # Original text
                    row["cleaned"],             # Cleaned text
                    " ".join(row["tokens"]),    # Tokens joined as space separated string
                    " ".join(row["lemmas"])     # Lemmas joined as space separated string
                )
            )
        # Commit changes to DB
        conn.commit()  