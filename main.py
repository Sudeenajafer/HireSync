import gradio_client.utils as client_utils

# --- 1. THE NUCLEAR MONKEY PATCH FOR PYTHON 3.14 ---
# This disables the recursive API parser that is crashing on FileData schemas.
def total_bypass_parser(*args, **kwargs):
    return "Any"

# Overwrite both internal parsing functions to stop the 500 error
client_utils.json_schema_to_python_type = total_bypass_parser
client_utils._json_schema_to_python_type = total_bypass_parser

def patched_get_type(schema):
    if not isinstance(schema, dict): return "unknown"
    return "const" if "const" in schema else "enum" if "enum" in schema else "unknown"
client_utils.get_type = patched_get_type
# --------------------------------------------------

import gradio as gr
import os
import PyPDF2
from features import extract_features
from src.ats_matcher import ATSMatcher

# Ensure FFmpeg is recognized for Stage 2
try:
    from static_ffmpeg import add_paths
    add_paths()
except:
    pass

matcher = ATSMatcher()

# --- STAGE 1: ATS ---
def run_ats_stage(resume_pdf, jd_file, name_text):
    if not resume_pdf or not jd_file or not name_text:
        return "<div style='color:#f87171;'>⚠️ All inputs (Name, Resume, JD) are required.</div>", -1
    try:
        is_name_v, name_e = matcher.validate_name(name_text)
        if not is_name_v: return f"<div style='background:#450a0a; padding:15px; border-radius:10px; color:#f87171;'>{name_e}</div>", -1
        
        reader = PyPDF2.PdfReader(resume_pdf.name)
        resume_text = " ".join([page.extract_text() for page in reader.pages])
        
        is_res_v, res_e = matcher.validate_resume(resume_text)
        if not is_res_v: return f"<div style='background:#450a0a; padding:15px; border-radius:10px; color:#f87171;'>{res_err}</div>", -1

        score, skills = matcher.analyze_resume(resume_text, jd_file.name)
        badges = "".join([f"<span style='background:#1e293b; color:#38bdf8; padding:4px 8px; margin:2px; border-radius:5px; display:inline-block; border:1px solid #38bdf8; font-size:11px;'>{s.upper()}</span>" for s in skills])
        
        return f"""
        <div style="background:#0f172a; padding:15px; border-radius:12px; border:1px solid #38bdf8;">
            <h4 style="color:#38bdf8; margin:0;">Phase 1: Resume Match</h4>
            <h1 style="font-size:40px; color:white; margin:10px 0;">{score*100:.1f}%</h1>
            <div>{badges}</div>
            <p style="color:#22c55e; font-weight:bold; margin-top:10px;">✅ Data Saved. Proceed to Stage 2.</p>
        </div>
        """, score
    except Exception as e:
        return f"ATS Error: {str(e)}", -1

# --- STAGE 2: INTERVIEW ---
def run_interview_stage(video_input):
    if not video_input: return "⚠️ Error: No video.", 0, "00:00"
    try:
        video_path = video_input if isinstance(video_input, str) else video_input.name
        res = extract_features(None, None, video_path, skip_ats=True)
        color = "#4ade80" if res['behavior_score'] > 0.7 else "#f87171"
        
        html = f"""
        <div style="background:#0f172a; padding:20px; border-radius:15px; border:2px solid {color}; max-width:500px;">
            <div style="display:flex; justify-content:space-between; color:white; margin-bottom:10px;">
                <h3 style="color:#38bdf8; margin:0;">MSc Assessment</h3>
                <span style="background:#334155; padding:3px 10px; border-radius:10px; font-size:11px;">⏱️ {res['duration']}</span>
            </div>
            <div style="background:#1e293b; padding:15px; border-radius:12px; text-align:center; margin-bottom:15px;">
                <h1 style="color:{color}; font-size:48px; margin:5px 0;">{res['behavior_grade']}</h1>
                <p style="color:white; margin:0;">AI Behavioral Score: {res['behavior_score']}</p>
            </div>
            <div style="background:#000; color:#4ade80; padding:10px; border-radius:8px; height:120px; overflow-y:auto; font-family:monospace; font-size:11px;">
                {res['transcript']}
            </div>
            <p style="color:#94a3b8; font-size:11px; margin-top:10px;">Detected Speed: <b>{res['wpm']} WPM</b></p>
        </div>
        """
        return html, res['behavior_score'], f"⏱️ Duration: {res['duration']}"
    except Exception as e:
        return f"Interview Error: {str(e)}", 0, "00:00"

# --- STAGE 3: FINAL ---
def generate_final_report(name, ats, behavior):
    if ats <= 0 or behavior <= 0:
        return "<div style='color:#f87171;'>⚠️ Complete Stage 1 & 2 first.</div>"
    final = round((ats * 0.6) + (behavior * 0.4), 2)
    rec = "✅ RECOMMENDED" if final > 0.75 else "⏳ REVIEW"
    return f"""
    <div style="background:#1e293b; padding:30px; border-radius:15px; border:2px solid #38bdf8; text-align:center; max-width:500px; margin:auto;">
        <h2 style="color:white;">{name.upper()}</h2>
        <h1 style="color:#38bdf8; font-size:60px; margin:15px 0;">{final}</h1>
        <h2 style="color:#4ade80;">{rec}</h2>
    </div>
    """

# --- UI LAYOUT ---
compact_css = ".gradio-container { max-width: 900px !important; margin: auto !important; }"

with gr.Blocks(theme=gr.themes.Soft(), css=compact_css, title="HireSync AI") as demo:
    ats_score = gr.State(0); int_score = gr.State(0)
    gr.Markdown("<h1 style='text-align:center; color:#38bdf8;'>🤖 HireSync AI Pipeline</h1>")
    
    with gr.Tabs():
        with gr.TabItem("📄 Stage 1: Resume"):
            with gr.Row():
                with gr.Column():
                    name_in = gr.Textbox(label="Full Name")
                    res_in = gr.File(label="Resume (PDF)")
                    jd_in = gr.File(label="JD (TXT)")
                    btn1 = gr.Button("Analyze Resume", variant="primary")
                with gr.Column():
                    out1 = gr.HTML()
        
        with gr.TabItem("🎥 Stage 2: Interview"):
            with gr.Row():
                with gr.Column():
                    vid_in = gr.Video(label="Record/Upload", sources=["webcam", "upload"])
                    dur_label = gr.Markdown("⏱️ 00:00")
                    btn2 = gr.Button("Analyze Behavior", variant="primary")
                with gr.Column():
                    out2 = gr.HTML()

        with gr.TabItem("📊 Stage 3: Report"):
            btn3 = gr.Button("🏆 GENERATE FINAL SCORECARD", variant="primary")
            out3 = gr.HTML()

    btn1.click(run_ats_stage, [res_in, jd_in, name_in], [out1, ats_score])
    btn2.click(run_interview_stage, [vid_in], [out2, int_score, dur_label])
    btn3.click(generate_final_report, [name_in, ats_score, int_score], [out3])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, inline=False, share=True)