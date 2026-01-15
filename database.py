import sqlite3
import json
from datetime import datetime

DB_NAME = "hiresync_factory.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            ats_score REAL,
            behavior_score REAL,
            final_score REAL,
            details TEXT,
            timestamp DATETIME
        )
    ''')
    conn.commit()
    conn.close()

def save_candidate(name, ats, behavior, final, details_dict):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO candidates (name, ats_score, behavior_score, final_score, details, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (name, ats, behavior, final, json.dumps(details_dict), datetime.now()))
    conn.commit()
    conn.close()

def get_all_candidates():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT name, ats_score, behavior_score, final_score, timestamp FROM candidates ORDER BY timestamp DESC")
    data = cursor.fetchall()
    conn.close()
    return data

init_db()