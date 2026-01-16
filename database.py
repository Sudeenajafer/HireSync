import sqlite3
import pandas as pd
import os
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- 1. LOCAL SQLITE CONFIG (For Job Positions) ---
LOCAL_DB = "hiresync_local.db"

def init_local_db():
    """Initializes the local database for storing Job Descriptions."""
    conn = sqlite3.connect(LOCAL_DB)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            jd TEXT NOT NULL,
            questions TEXT,
            timestamp DATETIME
        )
    ''')
    conn.commit()
    conn.close()

def add_job(title, jd, questions):
    """Saves a new job opening to the local HR database."""
    conn = sqlite3.connect(LOCAL_DB)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO positions (title, jd, questions, timestamp) VALUES (?, ?, ?, ?)",
        (title, jd, questions, datetime.now())
    )
    conn.commit()
    conn.close()

# ... (Keep existing imports at the top of database.py)

def delete_job(title):
    """MSc Logic: Administrative revocation of a job position."""
    conn = sqlite3.connect(LOCAL_DB)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM positions WHERE title=?", (title,))
    conn.commit()
    conn.close()
    return f"🗑️ Position '{title}' has been removed from the system."

def get_all_positions_df():
    """
    Fetches all active job openings including the full JD 
    for the HR management table.
    """
    conn = sqlite3.connect(LOCAL_DB)
    # Added 'jd' to the SELECT statement
    df = pd.read_sql_query("SELECT title, jd, timestamp, questions FROM positions ORDER BY timestamp DESC", conn)
    conn.close()
    
    if not df.empty:
        # Update column headers to include Job Description
        df.columns = ["Job Title", "Job Description", "Date Created", "Interview Questions"]
    return df

# ... (Keep the rest of your database.py functions)
def get_job_titles():
    """Fetches list of job titles for the Candidate Dropdown."""
    init_local_db() # Ensure table exists
    conn = sqlite3.connect(LOCAL_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM positions ORDER BY timestamp DESC")
    titles = [row[0] for row in cursor.fetchall()]
    conn.close()
    return titles if titles else ["No Positions Available"]

def get_jd_by_title(title):
    """Retrieves the JD text for Gemini to analyze against the resume."""
    conn = sqlite3.connect(LOCAL_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT jd FROM positions WHERE title=?", (title,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else ""

# --- 2. CLOUD SUPABASE CONFIG (For Candidate Results) ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize client safely
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None
    print("⚠️ Warning: Supabase credentials missing from .env")

def get_cloud_candidates_df():
    """Fetches applicant summary from Supabase for the HR Table."""
    if not supabase: return pd.DataFrame(columns=["Error"], data=[["Supabase Not Connected"]])
    try:
        response = supabase.table("candidates").select("name, role, final_score, created_at").order("created_at", desc=True).execute()
        df = pd.DataFrame(response.data)
        if not df.empty:
            df.columns = ["Name", "Role", "Overall Fit", "Date Applied"]
        return df
    except Exception as e:
        print(f"Supabase Table Error: {e}")
        return pd.DataFrame(columns=["Status"], data=[["No data found"]])

def get_candidate_details(name):
    """Fetches the full dossier (URLs, Transcript) for a specific candidate."""
    if not supabase: return None
    try:
        response = supabase.table("candidates").select("*").eq("name", name).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        print(f"Details Fetch Error: {e}")
        return None

# Initialize the local database file immediately on run
init_local_db()