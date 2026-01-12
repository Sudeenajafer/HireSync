import gradio as gr
import pandas as pd
import os
import PyPDF2
from features import extract_features
from src.ats_matcher import ATSMatcher

# Initialize the backend engine
matcher = ATSMatcher()

# --- STAGE 1: ATS ANALYSIS ---
def run_ats_stage(resume_pdf, jd_file, name_text):
    if not resume_pdf or not jd_file or not name_text:
        return "<div style='color:#f87171;'>⚠️ All inputs (Name, Resume, JD) are required.</div>", -1

    try:
        # 1. VALIDATIONS
        is_name_v, name_e = matcher.validate_name(name_text)
        if not is_name_v: return f"<div style='color:#f87171;'>{name_e}</div>", -1
        
        # Extract text from PDF
        reader = PyPDF2.PdfReader(resume_pdf.name)
        resume_text = " ".join([page.extract_text() for page in reader.pages])
        
        is_res_v, res_e = matcher.validate_resume(resume_text)
        if not is_res_v: return f"<div style='color:#f87171;'>{res_e}</div>", -1

        # 2. MATCHING
        score, matching_skills = matcher.analyze_resume(resume_text, jd_file.name)
        
        # Skill Badge UI
        skills_html = "".join([f"<span style='background:#1e293b; color:#38bdf8; padding:4px 8px; margin:2px; border-radius:5px; display:inline-block; border:1px solid #38bdf8; font-size:11px;'>{s.upper()}</span>" for s in matching_skills])
        
        html = f"""
        <div style="background:#0f172a; padding:15px; border-radius:12px; border:1px solid #38bdf8; max-width:400px;">
            <h4 style="color:#38bdf8; margin:0;">Phase 1: Resume Alignment</h4>
            <h1 style="font-size:40px; color:white; margin:10px 0;">{score*100:.1f}%</h1>
            <p style="color:white; font-size:13px; margin-bottom:5px;"><b>Key Skills Found:</b></p>
            <div>{skills_html}</div>
            <p style="color:#22c55e; font-size:12px; margin-top:10px; font-weight:bold;">✅ Data Validated & Saved.</p>
        </div>
        """
        return html, score

    except Exception as e:
        return f"<div style='color:red;'>ATS Error: {str(e)}</div>", -1

# --- STAGE 2: INTERVIEW ANALYSIS ---
def run_interview_stage(video_path):
    if not video_path: 
        return "<div style='color:#f87171;'>⚠️ Please provide a video recording.</div>", 0, "⏱️ Duration: 00:00"
    
    try:
        # Process video/audio for behavioral metrics
        res = extract_features(None, None, video_path, skip_ats=True)
        
        # Compact Behavioral Dashboard with Grading
        html = f"""
        <div style="background:#0f172a; padding:15px; border-radius:12px; border:1px solid #a855f7; max-width:420px; font-family:sans-serif;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <h4 style="color:#a855f7; margin:0;">Phase 2: Behavioral Analysis</h4>
                <span style="background:#a855f7; color:white; padding:2px 8px; border-radius:8px; font-size:11px;">⏱️ {res['duration']}</span>
            </div>

            <div style="text-align:center; margin:10px 0; background:#1e293b; padding:10px; border-radius:10px; border:1px solid #334155;">
                <p style="color:#94a3b8; font-size:10px; margin:0; letter-spacing:1px;">PERFORMANCE GRADE</p>
                <h2 style="color:#4ade80; margin:2px 0;">{res['behavior_grade']}</h2>
                <p style="color:white; font-size:14px; margin:0;">AI Fit Score: {res['behavior_score']}</p>
            </div>

            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:5px; margin-bottom:15px; text-align:center;">
                <div style="background:#161b22; padding:6px; border-radius:6px;">
                    <p style="color:#94a3b8; font-size:9px; margin:0;">FLUENCY</p><b style="color:white; font-size:13px;">{res['fluency']}</b>
                </div>
                <div style="background:#161b22; padding:6px; border-radius:6px;">
                    <p style="color:#94a3b8; font-size:9px; margin:0;">COMM.</p><b style="color:white; font-size:13px;">{res['communication']}</b>
                </div>
                <div style="background:#161b22; padding:6px; border-radius:6px;">
                    <p style="color:#94a3b8; font-size:9px; margin:0;">ATTENTION</p><b style="color:white; font-size:13px;">{res['attention']}</b>
                </div>
            </div>

            <p style="color:white; margin:0 0 5px 0; font-size:12px;"><b>Full Transcription:</b></p>
            <div style="background:#000; color:#4ade80; padding:10px; border-radius:6px; height:130px; overflow-y:auto; font-family:monospace; font-size:11px; border:1px solid #333; line-height:1.4;">
                {res['transcript']}
            </div>
            <p style="color:#22c55e; font-size:11px; margin-top:8px; font-weight:bold;">✅ Behavior Analyzed. Proceed to Final Report.</p>
        </div>
        """
        return html, res['behavior_score'], f"⏱️ Detected Duration: {res['duration']}"
    
    except Exception as e: 
        return f"<div style='color:red;'>Interview Error: {str(e)}</div>", 0, "Error"

# --- STAGE 3: FINAL EVALUATION ---
def generate_final_report(name, ats_score, int_score):
    if ats_score <= 0 or int_score <= 0:
        return "<div style='color:#f87171; text-align:center; padding:20px; border:1px solid #450a0a; border-radius:10px;'>⚠️ Error: Both Phase 1 (ATS) and Phase 2 (Interview) must be completed.</div>"
    
    # Fusion Logic: 60% Technical/Resume + 40% Behavioral/Interview
    final_suitability = round((ats_score * 0.6) + (int_score * 0.4), 2)
    
    # Recommendation Logic
    if final_suitability >= 0.80:
        rec = "✅ STRONGLY RECOMMENDED"
        color = "#22c55e"
    elif final_suitability >= 0.65:
        rec = "⏳ CONSIDER FOR NEXT ROUND"
        color = "#f59e0b"
    else:
        rec = "❌ NOT RECOMMENDED"
        color = "#ef4444"

    html = f"""
    <div style="background:#1e293b; padding:30px; border-radius:15px; border:3px solid {color}; text-align:center; max-width:500px; margin:auto;">
        <h3 style="color:#38bdf8; margin:0;">FINAL EVALUATION: {name.upper()}</h3>
        <h1 style="color:white; font-size:60px; margin:15px 0;">{final_suitability}</h1>
        <h2 style="color:{color}; margin-bottom:20px;">{rec}</h2>
        
        <div style="display:flex; justify-content:space-around; border-top:1px solid #334155; padding-top:20px;">
            <div style="text-align:center;">
                <p style="color:#94a3b8; font-size:12px; margin:0;">ATS SCORE</p>
                <b style="color:white; font-size:18px;">{ats_score*100:.0f}%</b>
            </div>
            <div style="text-align:center;">
                <p style="color:#94a3b8; font-size:12px; margin:0;">INTERVIEW SCORE</p>
                <b style="color:white; font-size:18px;">{int_score}</b>
            </div>
        </div>
    </div>
    """
    return html

# --- UI LAYOUT ---
# compact_css limits the width to make the app look like a professional dashboard
compact_css = """
.gradio-container { max-width: 950px !important; margin: auto !important; }
.tabs { border-radius: 12px; overflow: hidden; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=compact_css) as demo:
    # App State
    ats_score_state = gr.State(0)
    int_score_state = gr.State(0)

    gr.Markdown("<h1 style='text-align:center; color:#38bdf8;'>🤖 HireSync AI: Recruitment Pipeline</h1>")
    
    with gr.Tabs():
        # PHASE 1
        with gr.TabItem("📄 Phase 1: ATS Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    name_in = gr.Textbox(label="Candidate Full Name", placeholder="e.g. Sudeena Jafer")
                    res_in = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
                    jd_in = gr.File(label="Upload Job Description (TXT)", file_types=[".txt"])
                    ats_btn = gr.Button("🚀 START ATS MATCHING", variant="primary")
                with gr.Column(scale=1):
                    ats_out = gr.HTML("<div style='text-align:center; padding-top:100px; color:#666;'>Awaiting document upload...</div>")

        # PHASE 2
        with gr.TabItem("🎥 Phase 2: Behavioral Interview"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid_in = gr.Video(label="Interview Recording (10m Limit)", sources=["webcam", "upload"])
                    dur_label = gr.Markdown("⏱️ **Duration: 00:00**")
                    int_btn = gr.Button("🚀 START BEHAVIORAL ANALYSIS", variant="primary")
                with gr.Column(scale=1):
                    int_out = gr.HTML("<div style='text-align:center; padding-top:100px; color:#666;'>Awaiting video recording...</div>")

        # PHASE 3
        with gr.TabItem("📊 Final Scorecard"):
            with gr.Column():
                gr.Markdown("<p style='text-align:center;'>Fusion of technical suitability and behavioral performance metrics.</p>")
                final_btn = gr.Button("🏆 GENERATE FINAL EVALUATION", variant="primary", size="lg")
                final_out = gr.HTML("<div style='text-align:center; padding:50px; color:#666;'>Complete Phases 1 & 2 to unlock.</div>")

    # Wire up the logic
    ats_btn.click(run_ats_stage, inputs=[res_in, jd_in, name_in], outputs=[ats_out, ats_score_state])
    
    int_btn.click(run_interview_stage, inputs=[vid_in], outputs=[int_out, int_score_state, dur_label])
    
    final_btn.click(generate_final_report, inputs=[name_in, ats_score_state, int_score_state], outputs=[final_out])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, share=False)