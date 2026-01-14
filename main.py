import gradio as gr
import pandas as pd
import os
import PyPDF2
from features import extract_features
from src.ats_matcher import ATSMatcher

matcher = ATSMatcher()

# --- STAGE 1: ATS ANALYSIS ---
def run_ats_stage(resume_pdf, jd_file, name_text):
    if not resume_pdf or not jd_file or not name_text:
        return "<div style='color:#f87171;'>⚠️ All inputs (Name, Resume, JD) are required.</div>", -1
    try:
        is_name_v, name_e = matcher.validate_name(name_text)
        if not is_name_v: return f"<div style='background:#450a0a; padding:15px; border-radius:10px; color:#f87171;'>{name_e}</div>", -1
        
        reader = PyPDF2.PdfReader(resume_pdf.name)
        resume_text = " ".join([page.extract_text() for page in reader.pages])
        
        is_res_v, res_e = matcher.validate_resume(resume_text)
        if not is_res_v: return f"<div style='background:#450a0a; padding:15px; border-radius:10px; color:#f87171;'>{res_e}</div>", -1

        score, matching_skills = matcher.analyze_resume(resume_text, jd_file.name)
        skills_html = "".join([f"<span style='background:#1e293b; color:#38bdf8; padding:4px 8px; margin:2px; border-radius:5px; display:inline-block; border:1px solid #38bdf8; font-size:11px;'>{s.upper()}</span>" for s in matching_skills])
        
        html_report = f"""
        <div style="background:#0f172a; padding:15px; border-radius:12px; border:1px solid #38bdf8;">
            <h4 style="color:#38bdf8; margin:0;">Phase 1: Resume Alignment</h4>
            <h1 style="font-size:40px; color:white; margin:10px 0;">{score*100:.1f}%</h1>
            <div>{skills_html}</div>
            <p style="color:#22c55e; font-size:12px; margin-top:10px; font-weight:bold;">✅ Data Validated & Saved.</p>
        </div>
        """
        return html_report, score
    except Exception as e:
        return f"<div style='color:red;'>ATS Error: {str(e)}</div>", -1

# --- STAGE 2: INTERVIEW ANALYSIS ---
def run_interview_stage(video_path):
    if not video_path: 
        return "<div style='color:#f87171;'>⚠️ Please provide a video recording.</div>", 0, "⏱️ Duration: 00:00"
    try:
        res = extract_features(None, None, video_path, skip_ats=True)
        html_report = f"""
        <div style="background:#0f172a; padding:15px; border-radius:12px; border:1px solid #a855f7; max-width:420px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h4 style="color:#a855f7; margin:0;">Phase 2: Behavioral Analysis</h4>
                <span style="background:#a855f7; color:white; padding:2px 8px; border-radius:8px; font-size:11px;">⏱️ {res['duration']}</span>
            </div>
            <div style="text-align:center; margin:10px 0; background:#1e293b; padding:10px; border-radius:10px;">
                <h2 style="color:#4ade80; margin:2px 0;">{res.get('behavior_grade', 'B')}</h2>
                <p style="color:white; font-size:14px; margin:0;">AI Score: {res['behavior_score']}</p>
            </div>
            <div style="background:#000; color:#4ade80; padding:10px; border-radius:6px; height:120px; overflow-y:auto; font-family:monospace; font-size:11px;">{res['transcript']}</div>
        </div>
        """
        return html_report, res['behavior_score'], f"⏱️ Duration: {res['duration']}"
    except Exception as e: 
        return f"<div style='color:red;'>Interview Error: {str(e)}</div>", 0, "Error"

# --- STAGE 3: FINAL EVALUATION ---
def generate_final_report(name, ats_score, int_score):
    if ats_score <= 0 or int_score <= 0:
        return "<div style='color:#f87171; text-align:center; padding:20px;'>⚠️ Complete Phase 1 & 2 first.</div>"
    final_suitability = round((ats_score * 0.6) + (int_score * 0.4), 2)
    return f"""
    <div style="background:#1e293b; padding:30px; border-radius:15px; border:2px solid #38bdf8; text-align:center;">
        <h3 style="color:#38bdf8; margin:0;">FINAL SCORECARD: {name.upper()}</h3>
        <h1 style="color:white; font-size:60px; margin:15px 0;">{final_suitability}</h1>
        <h2 style="color:#4ade80;">{"✅ RECOMMENDED" if final_suitability > 0.7 else "⏳ REVIEW NEEDED"}</h2>
    </div>
    """

# --- UI LAYOUT ---
compact_css = ".gradio-container { max-width: 900px !important; margin: auto !important; }"

with gr.Blocks(title="HireSync AI") as demo:
    ats_score_state = gr.State(0)
    int_score_state = gr.State(0)

    gr.Markdown("<h1 style='text-align:center; color:#38bdf8;'>🤖 HireSync AI Pipeline</h1>")
    
    with gr.Tabs():
        with gr.TabItem("📄 Phase 1: ATS"):
            with gr.Row():
                with gr.Column():
                    name_in = gr.Textbox(label="Candidate Name")
                    res_in = gr.File(label="Resume (PDF)")
                    jd_in = gr.File(label="JD (TXT)")
                    ats_btn = gr.Button("Analyze Resume", variant="primary")
                with gr.Column():
                    # FIXED: Variable name here matches logic below
                    ats_output_display = gr.HTML("<div style='padding-top:50px; color:#666;'>Awaiting upload...</div>")

        with gr.TabItem("🎥 Phase 2: Interview"):
            with gr.Row():
                with gr.Column():
                    vid_in = gr.Video(label="Interview Video", sources=["webcam", "upload"], format="mp4")
                    dur_label = gr.Markdown("⏱️ **Duration: 00:00**")
                    int_btn = gr.Button("Analyze Behavior", variant="primary")
                with gr.Column():
                    int_output_display = gr.HTML("<div style='padding-top:50px; color:#666;'>Awaiting interview...</div>")

        with gr.TabItem("📊 Final Report"):
            final_btn = gr.Button("🏆 GENERATE FINAL SCORECARD", variant="primary", size="lg")
            final_output_display = gr.HTML("<div style='text-align:center; padding:50px;'>Complete previous stages.</div>")

    # Wire up logic
    ats_btn.click(run_ats_stage, [res_in, jd_in, name_in], [ats_output_display, ats_score_state])
    int_btn.click(run_interview_stage, [vid_in], [int_output_display, int_score_state, dur_label])
    final_btn.click(generate_final_report, [name_in, ats_score_state, int_score_state], [final_output_display])

if __name__ == "__main__":
    # CRITICAL: show_api=False avoids the Python 3.14/3.12 boolean error
    demo.queue().launch(
    inbrowser=True, 
    share=False,
    server_name="0.0.0.0",
    server_port=7860,
    theme=gr.themes.Soft(),
    css=compact_css
)


