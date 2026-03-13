import sqlite3
import pandas as pd
import os
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
LOCAL_DB = "hiresync_local.db"
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# --- Update these functions in database.py ---

def init_local_db():
    conn = sqlite3.connect(LOCAL_DB)
    cursor = conn.cursor()
    # Added 'vacancies' column to the positions table
    cursor.execute('''CREATE TABLE IF NOT EXISTS positions 
        (id INTEGER PRIMARY KEY AUTOINCREMENT, 
         title TEXT NOT NULL, 
         jd TEXT NOT NULL, 
         vacancies INTEGER DEFAULT 1, 
         questions TEXT, 
         timestamp DATETIME)''')
    conn.commit()
    conn.close()

def add_job(title, jd, vacancies, questions):
    conn = sqlite3.connect(LOCAL_DB)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO positions (title, jd, vacancies, questions, timestamp) VALUES (?, ?, ?, ?, ?)", 
                   (title, jd, vacancies, questions, datetime.now()))
    conn.commit()
    conn.close()

def get_vacancy_count(title):
    """Retrieves the K value for a specific job."""
    conn = sqlite3.connect(LOCAL_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT vacancies FROM positions WHERE title=?", (title,))
    res = cursor.fetchone()
    conn.close()
    return res[0] if res else 1

def get_top_k_candidates(role_title, k):
    """
    MSc Logic: Rank-ordered retrieval.
    Fetches top K candidates from Supabase ordered by final_score.
    """
    if not supabase: return pd.DataFrame()
    try:
        response = supabase.table("candidates")\
            .select("name, role, final_score, behavior_grade, created_at")\
            .eq("role", role_title)\
            .order("final_score", desc=True)\
            .limit(k)\
            .execute()
        
        df = pd.DataFrame(response.data)
        if not df.empty:
            df.columns = ["Name", "Role", "Overall Score", "AI Grade", "Date Applied"]
            # Add a Rank column for visual clarity
            df.insert(0, "Rank", range(1, len(df) + 1))
            # Add the button column for consistency
            df["Deep Analysis"] = "🔍 VIEW DETAILS"
        return df
    except Exception as e:
        print(f"Ranking Error: {e}")
        return pd.DataFrame()

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
    response = supabase.table("candidates").select("name, role, final_score, created_at").eq("role", role).order("final_score", desc=True).execute()
    df = pd.DataFrame(response.data)
    if not df.empty:
        df.columns = ["Name", "Role", "Score", "Date Applied"]
        df["Deep Analysis"] = "🔍 VIEW DETAILS"
    return df

def get_candidate_details(name):
    response = supabase.table("candidates").select("*").eq("name", name).execute()
    return response.data[0] if response.data else None

init_local_db()