import gradio_client.utils as client_utils

# --- 1. THE NUCLEAR PATCH (Python 3.14/3.12 Stability) ---
def patched_get_type(schema):
    if not isinstance(schema, dict): return "unknown"
    return "const" if "const" in schema else "enum" if "enum" in schema else "unknown"
client_utils.get_type = patched_get_type
client_utils.json_schema_to_python_type = lambda *args, **kwargs: "Any"

import gradio as gr
import pandas as pd
import PyPDF2
import os
import json
import time
import threading
from features import extract_features, upload_to_cloud, save_candidate_to_supabase
from src.ats_matcher import ATSMatcher
import database as db
from static_ffmpeg import add_paths

# Initialize backend tools
add_paths()
matcher = ATSMatcher()

# --- CSS FOR PROFESSIONAL INTERFACE ---
custom_css = """
footer { display: none !important; }
.gradio-container { background-color: #0b0f19 !important; }
.candidate-box { border: 1px solid #10b981 !important; padding: 25px; border-radius: 15px; background: #111827; }
.hr-box { border: 1px solid #38bdf8 !important; padding: 20px; border-radius: 15px; background: #111827; }
.agent-box { background: #1e293b; border-left: 5px solid #10b981; padding: 15px; border-radius: 10px; margin: 10px 0; }
.submit-btn { background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important; color: white !important; }

@keyframes pulse-red {
    0% { border-color: #ef4444; box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
    70% { border-color: #7f1d1d; box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
    100% { border-color: #ef4444; box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}
.wait-warning { 
    background: #450a0a; border: 2px solid #ef4444; padding: 15px; border-radius: 10px; 
    animation: pulse-red 2s infinite; margin-top: 10px;
}
"""

# --- HR LOGIC ---
def hr_create_job(title, jd_text, manual_q):
    if not title or not jd_text: return "⚠️ Error: Title/JD required.", gr.update(), gr.update(), gr.update()
    questions = manual_q if manual_q.strip() else matcher.generate_questions(jd_text)
    db.add_job(title, jd_text, questions)
    new_choices = db.get_job_titles()
    return f"✅ Position '{title}' published!", gr.update(choices=new_choices), gr.update(choices=new_choices), gr.update(choices=new_choices)

def hr_delete_job(title):
    if not title or title == "No Positions Available": return "⚠️ Select a valid position.", gr.update(), gr.update(), gr.update()
    msg = db.delete_job(title)
    new_choices = db.get_job_titles()
    return msg, gr.update(choices=new_choices), gr.update(choices=new_choices), gr.update(choices=new_choices)

# --- CANDIDATE FLOW LOGIC ---
def go_to_step_2(name, role):
    if not name or not role: return gr.update(visible=True), gr.update(visible=False), "⚠️ Name and Role required."
    is_v, err = matcher.validate_name(name)
    if not is_v: return gr.update(visible=True), gr.update(visible=False), err
    return gr.update(visible=False), gr.update(visible=True), "Next step unlocked."

def go_to_step_3(resume, role_title):
    if not resume: return gr.update(visible=True), gr.update(visible=False), "⚠️ Upload Resume.", "", ""
    try:
        reader = PyPDF2.PdfReader(resume.name)
        resume_text = " ".join([page.extract_text() for page in reader.pages])
        target_jd = db.get_jd_by_title(role_title)
        ai_questions = matcher.generate_questions_from_resume(resume_text, target_jd)
        
        q_html = f"""<div class='agent-box'><h3 style='color:#10b981; margin:0 0 10px 0;'>📋 Interview Assessment</h3>
        <p style='color:#94a3b8; font-size:12px; margin-bottom:15px;'>Please answer the following questions clearly:</p>
        <p style='color:white; white-space: pre-wrap; font-size:16px; line-height:1.6;'>{ai_questions}</p></div>"""
        return gr.update(visible=False), gr.update(visible=True), "✅ Interview Ready", q_html, resume_text
    except Exception as e: return gr.update(visible=True), gr.update(visible=False), f"❌ {e}", "", ""

def background_analysis_task(name, role, email, phone, res_path, vid_path, res_text, q_html):
    try:
        raw_q = q_html.replace("<div class='agent-box'>", "").replace("</div>", "").replace("<h3 style='color:#10b981; margin:0 0 10px 0;'>📋 Interview Assessment</h3>", "").replace("<p style='color:#94a3b8; font-size:12px; margin-bottom:15px;'>Please answer the following questions clearly:</p>", "").replace("<p style='color:white; white-space: pre-wrap; font-size:16px; line-height:1.6;'>", "").replace("</p>", "")
        target_jd = db.get_jd_by_title(role)
        ats_data = matcher.analyze_resume_llm(res_text, target_jd)
        behavior = extract_features(res_path, target_jd, vid_path, skip_ats=True, questions=raw_q)
        res_url = upload_to_cloud(res_path, "image")
        vid_url = upload_to_cloud(vid_path, "video")
        
        ats_val = ats_data.get('final_score', 0) if ats_data else 0
        record = {
            "name": name, "role": role, "email": email, "phone": phone,
            "ats_score": ats_val, "behavior_score": behavior.get('behavior_score', 0) * 100,
            "final_score": round(((ats_val/100)*0.6) + (behavior.get('behavior_score', 0)*0.4), 2)*100,
            "transcript": behavior.get('transcript', ''), "resume_url": res_url, "video_url": vid_url,
            "behavior_grade": behavior.get('behavior_grade', 'B'),
            "integrity_status": behavior.get('integrity_status', 'Safe'),
            "conflict_report": behavior.get('conflict_report', 'Clean'),
            "details": ats_data.get('explanation', {"strengths": "N/A", "weaknesses": "N/A"})
        }
        save_candidate_to_supabase(record)
        print(f"✅ [CLOUD SYNC] Complete for {name}")
    except Exception as e: print(f"❌ Background Error: {e}")

def final_candidate_submit(name, role, email, phone, resume, video, res_text, q_html):
    if not video: return "⚠️ Record Video", gr.update(visible=True), name, email, phone, resume, video
    threading.Thread(target=background_analysis_task, args=(name, role, email, phone, resume.name, video, res_text, q_html)).start()
    return "✅ **Registration Successful!** AI processing in background.", gr.update(visible=True), "", "", "", None, None

def show_processing_warning(video_data):
    if video_data is not None:
        return gr.update(value='<div class="wait-warning">⚠️ Finalizing Recording... Please wait for the Play icon before clicking Submit.</div>', visible=True)
    return gr.update(visible=False)

# --- HR LOGIC: DATA FETCHING ---
def load_full_candidate_info(name):
    print(f"📡 HR Fetching: {name}")
    data = db.get_candidate_details(name)
    if not data: return [None]*7 + ["N/A", "### ❌ Record not yet found."]
    
    details = data.get('details') or {}
    if isinstance(details, str): details = json.loads(details)
    
    xai = f"<div style='background:#1e293b; padding:10px; border-radius:8px;'><b>🧠 AI REASONING</b><br>Strengths: {details.get('strengths', 'N/A')}<br>Gaps: {details.get('weaknesses', 'N/A')}</div>"
    raw_status = str(data.get('integrity_status', 'Safe')).upper()
    color = "#ef4444" if raw_status == "HIGH" else "#22c55e"
    integ = f"<div style='border:2px solid {color}; padding:10px;'>🛡️ <b>INTEGRITY: {raw_status}</b></div>"
    stats = f"**Contact:** {data.get('email', 'N/A')} | {data.get('phone', 'N/A')}\n- **Grade:** {data.get('behavior_grade', 'N/A')}\n- **Overall Fit:** {data.get('final_score', 0)}%"
    
    return (data.get('video_url'), f"### 📄 [View Resume]({data.get('resume_url', '#')})", data.get('transcript', ''), f"{data.get('ats_score', 0)}%", stats, integ, xai, f"## 👤 Reviewing: {name}")

def handle_applicant_selection(evt: gr.SelectData, df):
    if evt.index[1] != 4: return [gr.update()]*11
    name = df.iloc[evt.index[0], 0]
    return [None, "### ⏳ Loading...", "⏳ Fetching...", "...", "...", "...", "...", f"## 👤 Accessing: {name}", gr.update(visible=True), gr.Tabs(selected=4), name]

# --- GUI LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="HireSync AI Pro") as demo:
    res_txt_state = gr.State(""); hr_real_name_state = gr.State("")

    with gr.Tabs() as main_tabs:
        # --- CANDIDATE ---
        with gr.TabItem("👤 CANDIDATE PORTAL"):
            with gr.Column(visible=True, elem_classes="candidate-box") as step_1_ui:
                gr.Markdown("<p class='step-header'>Step 1: Login</p>")
                c_name = gr.Textbox(label="Full Name"); c_role = gr.Dropdown(label="Position", choices=db.get_job_titles())
                login_btn = gr.Button("Next ➡️", variant="primary"); c_status = gr.Markdown("Ready")
            with gr.Column(visible=False, elem_classes="candidate-box") as step_2_ui:
                gr.Markdown("<p class='step-header'>Step 2: Upload Details</p>")
                c_email = gr.Textbox(label="Email"); c_phone = gr.Textbox(label="Phone"); c_res = gr.File(label="Resume (PDF)")
                upload_btn = gr.Button("Next ➡️", variant="primary")
            with gr.Column(visible=False, elem_classes="candidate-box") as step_3_ui:
                gr.Markdown("<p class='step-header'>Step 3: AI Interview</p>")
                c_q_area = gr.HTML(); c_vid = gr.Video(label="Record Response", sources=["webcam", "upload"], interactive=True)
                c_wait_msg = gr.HTML(visible=False); submit_btn = gr.Button("🚀 FINISH & SUBMIT", variant="primary")

        # --- HR ADMIN ---
        with gr.TabItem("🏢 HR ADMIN PORTAL"):
            with gr.Tabs() as hr_tabs:
                with gr.TabItem("➕ Publish Job", id=1):
                    with gr.Column(elem_classes="hr-box"):
                        h_title = gr.Textbox(label="Job Title"); h_jd = gr.Textbox(label="JD", lines=4); h_q = gr.Textbox(label="Questions", lines=2)
                        h_btn = gr.Button("Publish Position", variant="primary"); h_status = gr.Markdown("")
                with gr.TabItem("📋 Manage Vacancies", id=2):
                    with gr.Column(elem_classes="hr-box"):
                        hr_job_refresh = gr.Button("🔄 REFRESH"); hr_job_table = gr.Dataframe(); hr_delete_select = gr.Dropdown(label="Delete", choices=db.get_job_titles()); hr_delete_btn = gr.Button("DELETE", variant="stop"); hr_delete_status = gr.Markdown("")
                with gr.TabItem("📊 Applicant List", id=3):
                    with gr.Row():
                        hr_view_selector = gr.Dropdown(label="Filter Role", choices=db.get_job_titles()); hr_refresh_list = gr.Button("🔄 LOAD")
                    hr_main_table = gr.Dataframe(headers=["Name", "Role", "Score", "Date", "Analysis"], interactive=False)
                with gr.TabItem("🔍 Deep Review", id=4, visible=False) as deep_review_tab:
                    hr_back_btn = gr.Button("⬅️ BACK")
                    with gr.Row():
                        with gr.Column(scale=1):
                            hr_name_display = gr.Markdown(); hr_ats_label = gr.Label(label="ATS Score"); hr_integrity_box = gr.HTML(); hr_xai_box = gr.HTML(); hr_extra_stats = gr.Markdown(); hr_pdf_display = gr.Markdown(); hr_reload_btn = gr.Button("🔄 RE-LOAD DATA")
                        with gr.Column(scale=2):
                            hr_video_display = gr.Video(label="Cloud Playback"); hr_transcript_display = gr.Textbox(label="Transcription", lines=10)

    # --- ACTION MAPPINGS ---
    h_btn.click(hr_create_job, [h_title, h_jd, h_q], [h_status, c_role, hr_view_selector, hr_delete_select])
    hr_job_refresh.click(db.get_all_positions_df, outputs=[hr_job_table])
    hr_delete_btn.click(hr_delete_job, [hr_delete_select], [hr_delete_status, c_role, hr_delete_select, hr_view_selector])
    
    login_btn.click(go_to_step_2, [c_name, c_role], [step_1_ui, step_2_ui, c_status])
    upload_btn.click(go_to_step_3, [c_res, c_role], [step_2_ui, step_3_ui, c_status, c_q_area, res_txt_state])
    c_vid.change(show_processing_warning, [c_vid], [c_wait_msg])
    submit_btn.click(final_candidate_submit, [c_name, c_role, c_email, c_phone, c_res, c_vid, res_txt_state, c_q_area], [c_status, step_1_ui, c_name, c_email, c_phone, c_res, c_vid]).then(lambda: [gr.update(visible=False), gr.update(visible=False)], None, [step_3_ui, c_wait_msg])

    hr_refresh_list.click(lambda r: db.get_candidates_by_role(r), [hr_view_selector], [hr_main_table])
    
    # Selecting triggers Jump + Loading Screen, THEN fetches real data (Total 11 arguments fixed)
    hr_main_table.select(
        handle_applicant_selection, 
        [hr_main_table], 
        [hr_video_display, hr_pdf_display, hr_transcript_display, hr_ats_label, hr_extra_stats, hr_integrity_box, hr_xai_box, hr_name_display, deep_review_tab, hr_tabs, hr_real_name_state]
    ).then(
        load_full_candidate_info, 
        [hr_real_name_state], 
        [hr_video_display, hr_pdf_display, hr_transcript_display, hr_ats_label, hr_extra_stats, hr_integrity_box, hr_xai_box, hr_name_display]
    )
    
    hr_reload_btn.click(load_full_candidate_info, [hr_real_name_state], [hr_video_display, hr_pdf_display, hr_transcript_display, hr_ats_label, hr_extra_stats, hr_integrity_box, hr_xai_box, hr_name_display])
    hr_back_btn.click(lambda: [gr.update(visible=False), gr.Tabs(selected=3)], outputs=[deep_review_tab, hr_tabs])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, share=True, inline=False)