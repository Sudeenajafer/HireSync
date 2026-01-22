import sqlite3
import pandas as pd
import os
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
LOCAL_DB = "hiresync_local.db"
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def init_local_db():
    conn = sqlite3.connect(LOCAL_DB)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS positions 
        (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, jd TEXT NOT NULL, questions TEXT, timestamp DATETIME)''')
    conn.commit()
    conn.close()

def add_job(title, jd, questions):
    conn = sqlite3.connect(LOCAL_DB)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO positions (title, jd, questions, timestamp) VALUES (?, ?, ?, ?)", (title, jd, questions, datetime.now()))
    conn.commit()
    conn.close()

def delete_job(title):
    conn = sqlite3.connect(LOCAL_DB)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM positions WHERE title=?", (title,))
    conn.commit()
    conn.close()
    return f"🗑️ Position '{title}' removed."

def get_job_titles():
    init_local_db()
    conn = sqlite3.connect(LOCAL_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM positions")
    titles = [row[0] for row in cursor.fetchall()]
    conn.close()
    return titles

def get_jd_by_title(title):
    conn = sqlite3.connect(LOCAL_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT jd FROM positions WHERE title=?", (title,))
    res = cursor.fetchone()
    conn.close()
    return res[0] if res else ""

def get_all_positions_df():
    conn = sqlite3.connect(LOCAL_DB)
    df = pd.read_sql_query("SELECT title, jd, timestamp FROM positions", conn)
    conn.close()
    if not df.empty: df.columns = ["Job Title", "Job Description", "Date Created"]
    return df

def get_candidates_by_role(role):
    response = supabase.table("candidates").select("name, role, final_score, created_at").eq("role", role).execute()
    df = pd.DataFrame(response.data)
    if not df.empty:
        df.columns = ["Name", "Role", "Score", "Date Applied"]
        df["Deep Analysis"] = "🔍 VIEW DETAILS"
    return df

def get_candidate_details(name):
    response = supabase.table("candidates").select("*").eq("name", name).execute()
    return response.data[0] if response.data else None

init_local_db()