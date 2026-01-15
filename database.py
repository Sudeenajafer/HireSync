import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "hiresync_enterprise.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Create Positions Table
    cursor.execute('''CREATE TABLE IF NOT EXISTS positions 
        (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, jd TEXT, questions TEXT)''')
    
    # Create Candidates Table (Ensuring 'role' is included)
    cursor.execute('''CREATE TABLE IF NOT EXISTS candidates 
        (id INTEGER PRIMARY KEY AUTOINCREMENT, 
         name TEXT, 
         role TEXT, 
         ats_score REAL, 
         behavior_score REAL, 
         final_score REAL, 
         transcript TEXT, 
         timestamp DATETIME)''')
    
    conn.commit()
    conn.close()

def add_job(title, jd, questions):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO positions (title, jd, questions) VALUES (?, ?, ?)", (title, jd, questions))
    conn.commit()
    conn.close()

def save_candidate(name, role, ats, behavior, final, transcript):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Explicitly naming columns in INSERT for safety
    cursor.execute('''INSERT INTO candidates 
        (name, role, ats_score, behavior_score, final_score, transcript, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)''', 
        (name, role, ats, behavior, final, transcript, datetime.now()))
    conn.commit()
    conn.close()

def get_job_titles():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM positions")
    titles = [row[0] for row in cursor.fetchall()]
    conn.close()
    return titles

def get_jd_by_title(title):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT jd FROM positions WHERE title=?", (title,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else ""

def get_all_candidates_df():
    conn = sqlite3.connect(DB_NAME)
    try:
        # Fetch data into a clean DataFrame for the HR table
        df = pd.read_sql_query("SELECT name, role, ats_score, behavior_score, final_score, timestamp FROM candidates ORDER BY timestamp DESC", conn)
        # Rename columns for professional display
        df.columns = ["Candidate Name", "Applied Position", "ATS Match %", "Behavioral Score", "Overall Fit", "Date Submitted"]
    except Exception as e:
        print(f"Database Read Error: {e}")
        df = pd.DataFrame(columns=["Status"], data=[["No candidates found"]])
    conn.close()
    return df

# Initialize on import
init_db()