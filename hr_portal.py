import gradio as gr
import PyPDF2
from features import extract_features
from candidate_portal import create_radar_chart # <--- ADD THIS IMPORT

def run_hr_ats(name, res_file, jd_file, matcher): # <--- REMOVED create_chart_fn
    if not all([name, res_file, jd_file]): return "⚠️ Inputs missing", None, 0, gr.Tabs()
    
    reader = PyPDF2.PdfReader(res_file.name)
    resume_text = " ".join([p.extract_text() for p in reader.pages])
    with open(jd_file.name, 'r') as f: jd_text = f.read()

    res = matcher.analyze_resume_llm(resume_text, jd_text)
    if not res: return "❌ Analysis Failed", None, 0, gr.Tabs()

    # FIX: Use the imported function directly
    chart = create_radar_chart(res) 
    
    html = f"""<div class='report-card'>
        <h3 style='color:#38bdf8; margin:0;'>HR Review: {res['final_score']}% Match</h3>
        <p style='color:white;'>{res['reasoning']}</p>
        <p style='color:#22c55e;'>✅ Data Saved. Redirecting to Stage 2...</p>
    </div>"""
    return html, chart, res['final_score'], gr.Tabs(selected=1)


def get_cloud_applicants():
    # Fetch from Supabase instead of SQLite
    response = supabase.table("candidates").select("*").order("created_at", desc=True).execute()
    return pd.DataFrame(response.data)