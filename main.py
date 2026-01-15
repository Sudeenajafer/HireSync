import gradio_client.utils as client_utils

# --- 1. THE NUCLEAR MONKEY PATCH FOR PYTHON 3.14 ---
# This prevents the 'bool is not container' crash in Gradio 6.0
def total_bypass_parser(*args, **kwargs): return "Any"
client_utils.json_schema_to_python_type = total_bypass_parser
client_utils._json_schema_to_python_type = total_bypass_parser
def patched_get_type(schema):
    if not isinstance(schema, dict): return "unknown"
    return "const" if "const" in schema else "enum" if "enum" in schema else "unknown"
client_utils.get_type = patched_get_type

import gradio as gr
import plotly.graph_objects as go
import PyPDF2
import os
import json
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv
from features import extract_features
from static_ffmpeg import add_paths

# Initialization
load_dotenv()
add_paths()

# --- 2. THE GEMINI ATS ENGINE ---
class ATSMatcher:
    def __init__(self):
        # Uses the modern 2026 google-genai SDK
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def validate_name(self, name):
        if not name or len(name.split()) < 2: 
            return False, "🛑 Please enter both First and Last name."
        if not re.match(r"^[a-zA-Z\s]+$", name):
            return False, "🛑 Name should only contain letters."
        return True, ""

    def validate_resume(self, text):
        markers = ['education', 'experience', 'skills']
        if not any(m in text.lower() for m in markers):
            return False, "🛑 This PDF doesn't look like a professional resume."
        return True, ""

    def analyze_resume_llm(self, resume_text, jd_text):
        """MSc Logic: Deep Semantic ATS Analysis using Gemini 1.5 Flash"""
        prompt = f"""
        Act as an expert ATS. Analyze the Resume against the Job Description.
        Return ONLY a JSON object:
        {{
            "education_score": int(0-100),
            "skills_score": int(0-100),
            "experience_score": int(0-100),
            "final_score": int(0-100),
            "matched_skills": ["skill1", "skill2"],
            "missing_skills": ["skillA"],
            "reasoning": "string",
            "suggestions": "string"
        }}
        Resume: {resume_text[:5000]}
        JD: {jd_text[:3000]}
        """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Gemini Error: {e}")
            return None

matcher = ATSMatcher()

# --- 3. UI HELPER FUNCTIONS ---
def create_radar_chart(data):
    fig = go.Figure()
    categories = ['Education', 'Skills', 'Experience']
    scores = [data['education_score'], data['skills_score'], data['experience_score']]
    fig.add_trace(go.Scatterpolar(r=scores + [scores[0]], theta=categories + [categories[0]], fill='toself', line_color='#38bdf8'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, paper_bgcolor='rgba(0,0,0,0)', font_color="white",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

# Logic for both Candidate and HR ATS Checks
def run_deep_ats(name, res_file, jd_file, is_hr=True):
    if not all([name, res_file, jd_file]): 
        return "⚠️ All inputs required.", None, 0, gr.Tabs()
    
    is_v, err = matcher.validate_name(name)
    if not is_v: return f"<div class='report-card'>{err}</div>", None, -1, gr.Tabs()

    # Extraction
    reader = PyPDF2.PdfReader(res_file.name)
    resume_text = " ".join([p.extract_text() for p in reader.pages])
    with open(jd_file.name, 'r', encoding='utf-8') as f: jd_text = f.read()

    # Gemini API Call
    res = matcher.analyze_resume_llm(resume_text, jd_text)
    if not res: return "❌ AI Matcher Unavailable", None, 0, gr.Tabs()

    chart = create_radar_chart(res)
    matched_html = "".join([f"<span style='border:1px solid #10b981; color:#10b981; padding:2px 8px; border-radius:5px; margin:2px; display:inline-block;'>{s}</span>" for s in res['matched_skills']])
    
    report_html = f"""
    <div style='background:#111827; padding:20px; border-radius:15px; border:1px solid #38bdf8;'>
        <h2 style='color:white; margin:0;'>AI Evaluation: {res['final_score']}%</h2>
        <p style='color:#94a3b8; font-size:14px; margin:10px 0;'>{res['reasoning']}</p>
        <div style='margin-bottom:10px;'>{matched_html}</div>
        <p style='color:#10b981; font-size:12px;'><b>Suggestions:</b> {res['suggestions']}</p>
    </div>
    """
    # If HR, switch to Interview Tab (index 1), if Candidate, stay (index 0)
    target_tab = 1 if is_hr else 0
    return report_html, chart, res['final_score'], gr.Tabs(selected=target_tab)

def run_interview_analysis(video_path):
    if not video_path: return "⚠️ Video required.", 0, "00:00"
    try:
        res = extract_features(None, None, video_path, skip_ats=True)
        color = "#4ade80" if res['behavior_score'] > 0.7 else "#f87171"
        html = f"""
        <div style="background:#0f172a; padding:20px; border-radius:15px; border:2px solid {color}; max-width:500px;">
            <div style="display:flex; justify-content:space-between; color:white;">
                <h3 style="color:#38bdf8; margin:0;">MSc Behavioral Assessment</h3>
                <span style="background:#334155; padding:3px 10px; border-radius:10px; font-size:12px;">⏱️ {res['duration']}</span>
            </div>
            <div style="background:#1e293b; padding:15px; border-radius:12px; text-align:center; margin:15px 0;">
                <h1 style="color:{color}; font-size:40px; margin:0;">{res['behavior_grade']}</h1>
                <p style="color:white; margin:0;">Geometric Mean Score: {res['behavior_score']}</p>
            </div>
            <div style="background:#000; color:#4ade80; padding:10px; border-radius:8px; height:100px; overflow-y:auto; font-family:monospace; font-size:11px;">{res['transcript']}</div>
        </div>
        """
        return html, res['behavior_score'], f"⏱️ Duration: {res['duration']}"
    except Exception as e: return f"Error: {e}", 0, "00:00"

def generate_final_decision(name, ats, behavior):
    if ats <= 0 or behavior <= 0: return "⚠️ Pipeline incomplete."
    final = round((ats * 0.6) + (behavior * 0.4), 2)
    return f"""<div style="background:#1e293b; padding:30px; border-radius:15px; border:2px solid #38bdf8; text-align:center; max-width:450px; margin:auto;">
        <h2 style="color:white;">{name.upper()}</h2>
        <h1 style="color:#38bdf8; font-size:65px; margin:0;">{final}</h1>
        <h2 style="color:#4ade80;">{"✅ RECOMMENDED" if final > 0.75 else "⏳ REVIEW"}</h2>
    </div>"""

# --- 4. GUI LAYOUT ---
compact_css = """
/* HIDE GRADIO FOOTER COMPLETELY */
footer { display: none !important; }
.gradio-container { max-width: 1000px !important; background-color: #0b0f19 !important; }

/* PROFESSIONAL REPORT CARDS */
.report-card { 
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
    border-radius: 15px; 
    border: 1px solid #38bdf8; 
    padding: 20px; 
}

/* ERROR BOX STYLING */
.error-box { 
    background: #450a0a; 
    padding: 15px; 
    border-radius: 10px; 
    border: 1px solid #ef4444; 
    color: #f87171; 
    font-weight: bold; 
}
"""


with gr.Blocks(title="HireSync AI Pro") as demo:
    ats_score_state = gr.State(0)
    int_score_state = gr.State(0)

    gr.Markdown("<h1 style='text-align:center; color:#38bdf8;'>🤖 HireSync AI: Multimodal Talent Suite</h1>")

    with gr.Tabs() as main_tabs:
        
        # --- PORTAL 1: CANDIDATE ---
        with gr.TabItem("👤 Candidate Portal"):
            with gr.Row():
                with gr.Column():
                    c_name = gr.Textbox(label="Full Name")
                    c_res = gr.File(label="Upload Resume (PDF)")
                    c_jd = gr.File(label="Target Job (TXT)")
                    c_btn = gr.Button("Evaluate My Resume", variant="primary")
                with gr.Column():
                    c_chart = gr.Plot()
                    c_out = gr.HTML("<div style='text-align:center; padding-top:100px;'>Results will appear here</div>")
            
            c_btn.click(lambda n, r, j: run_deep_ats(n, r, j, is_hr=False), [c_name, c_res, c_jd], [c_out, c_chart])

        # --- PORTAL 2: HR ADMIN ---
        with gr.TabItem("🏢 HR Admin Portal"):
            with gr.Tabs() as hr_tabs:
                
                with gr.TabItem("1. Document Review", id=0):
                    with gr.Row():
                        with gr.Column():
                            hr_name = gr.Textbox(label="Candidate Name")
                            hr_res = gr.File(label="Resume")
                            hr_jd = gr.File(label="Job Description")
                            hr_btn1 = gr.Button("Analyze Technical Fit", variant="primary")
                        with gr.Column():
                            hr_chart = gr.Plot()
                            hr_out1 = gr.HTML()

                with gr.TabItem("2. Interview Session", id=1):
                    with gr.Row():
                        with gr.Column():
                            hr_vid = gr.Video(label="Recorded Interview", sources=["webcam", "upload"])
                            dur_label = gr.Markdown("⏱️ **Duration: 00:00**")
                            hr_btn2 = gr.Button("Run Behavioral AI", variant="primary")
                        with gr.Column():
                            hr_out2 = gr.HTML()

                with gr.TabItem("3. Final Result", id=2):
                    hr_btn3 = gr.Button("🏆 GENERATE HIRING REPORT", variant="primary", size="lg")
                    hr_out3 = gr.HTML("<div style='text-align:center; padding:50px;'>Complete steps 1 and 2.</div>")

    # HR Portal Connections
    hr_btn1.click(run_deep_ats, [hr_name, hr_res, hr_jd], [hr_out1, hr_chart, ats_score_state, hr_tabs])
    hr_btn2.click(run_interview_analysis, [hr_vid], [hr_out2, int_score_state, dur_label])
    hr_btn3.click(generate_final_decision, [hr_name, ats_score_state, int_score_state], [hr_out3])

if __name__ == "__main__":
    demo.queue().launch(
        inbrowser=True,   # Opens your REAL browser (Chrome/Edge) automatically
        inline=False,     # DISBALES the VS Code internal window (the cause of the error)
        share=True,       # Provides HTTPS (Browsers require HTTPS for Webcam/Mic)
        debug=True
    )