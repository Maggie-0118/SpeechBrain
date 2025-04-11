import sqlite3
from config import DB_FILE

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS voice_transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            speaker TEXT,
            confidence REAL,
            transcript TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(speaker, confidence, transcript):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO voice_transcripts (speaker, confidence, transcript)
        VALUES (?, ?, ?)
    """, (speaker, confidence, transcript))
    conn.commit()
    conn.close()

def get_history(speaker):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT transcript FROM voice_transcripts
        WHERE speaker=? ORDER BY timestamp
    """, (speaker,))
    rows = cursor.fetchall()
    conn.close()
    return "\n".join([row[0] for row in rows])