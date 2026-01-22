import gradio_client.utils as client_utils

# --- 1. THE NUCLEAR STABILITY PATCH (Crucial for Python 3.14/3.12) ---
def patched_get_type(schema):
    if not isinstance(schema, dict): return "unknown"
    return "const" if "const" in schema else "enum" if "enum" in schema else "unknown"

client_utils.get_type = patched_get_type
client_utils.json_schema_to_python_type = lambda *args, **kwargs: "Any"
client_utils._json_schema_to_python_type = lambda *args, **kwargs: "Any"

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

# --- CANDIDATE FLOW LOGIC (3-STEP) ---
def go_to_step_2(name, role):
    is_v, err = matcher.validate_name(name)
    if not is_v: return gr.update(visible=True), gr.update(visible=False), err
    return gr.update(visible=False), gr.update(visible=True), ""

def go_to_step_3(resume, role_title):
    if not resume: return gr.update(visible=True), gr.update(visible=False), "⚠️ Upload Resume.", "", ""
    try:
        reader = PyPDF2.PdfReader(resume.name)
        resume_text = " ".join([page.extract_text() for page in reader.pages])
        target_jd = db.get_jd_by_title(role_title)
        
        # Adaptive Phase 8: Generate questions based on JD
        ai_questions = matcher.generate_questions_from_jd(target_jd)
        
        q_html = f"""<div class='agent-box'><h3 style='color:#10b981; margin:0 0 10px 0;'>📋 AI Interview Assessment</h3>
        <p style='color:white; white-space: pre-wrap; font-size:16px; line-height:1.6;'>{ai_questions}</p></div>"""
        
        return gr.update(visible=False), gr.update(visible=True), "✅ Interview Ready", q_html, resume_text
    except Exception as e: return gr.update(visible=True), gr.update(visible=False), f"❌ {e}", "", ""

def background_analysis_task(name, role, email, phone, res_path, vid_path, res_text, q_html):
    """Heavy background processing to avoid UI lag."""
    try:
        raw_q = q_html.replace("<div class='agent-box'>", "").replace("</div>", "").replace("AI Interview Assessment", "")
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

def final_candidate_submit(name, role, email, phone, resume, video, resume_text, questions_html):
    if not video: return "⚠️ Record Video", gr.update(visible=True), name, email, phone, resume, video
    
    # Start thread for zero-lag
    threading.Thread(target=background_analysis_task, args=(name, role, email, phone, resume.name, video, resume_text, questions_html)).start()
    
    return "✅ **Registration Successful!** You can now close this tab.", gr.update(visible=True), "", "", "", None, None

def show_processing_warning(video_data):
    if video_data is not None:
        return gr.update(value='<div class="wait-warning">⚠️ Finalizing Recording... Please wait for the Play icon before clicking Submit.</div>', visible=True)
    return gr.update(visible=False)

# --- HR LOGIC ---
def load_full_candidate_info(name):
    data = db.get_candidate_details(name)
    if not data: return ["No Data"]*6 + [f"## ❌ Record not yet found for {name}"]
    
    details = data.get('details') or {}
    if isinstance(details, str): details = json.loads(details)
    
    xai = f"<div style='background:#1e293b; border-left: 5px solid #38bdf8; padding:15px; border-radius:10px;'><b>🧠 AI REASONING (XAI)</b><br>Strengths: {details.get('strengths', 'N/A')}<br>Gaps: {details.get('weaknesses', 'N/A')}</div>"
    
    status = str(data.get('integrity_status', 'Safe')).upper()
    color = "#ef4444" if status == "HIGH" else "#22c55e"
    integ = f"<div style='background:#111827; border: 2px solid {color}; padding:15px; border-radius:10px;'>🛡️ <b>BEHAVIORAL INTEGRITY: {status}</b><br>{data.get('conflict_report', 'Cues aligned.')}</div>"
    
    final_score = int(data.get('final_score', 0))
    s_color = "#22c55e" if final_score > 75 else "#f59e0b"
    suit_html = f"<div style='background:#1e293b; padding:25px; border-radius:15px; border: 3px solid {s_color}; text-align:center;'><h3>SUITABILITY INDEX</h3><h1>{final_score}/100</h1></div>"
    
    contact = f"**Email:** {data.get('email', 'N/A')} | **Phone:** {data.get('phone', 'N/A')}\n\n[📄 Open Resume PDF]({data.get('resume_url', '#')})\n[📹 Watch Interview Video]({data.get('video_url', '#')})"
    
    return (data.get('transcript', ''), integ, gr.HTML(f"<b>Match: {data.get('ats_score', 0)}%</b>"), gr.HTML(f"<b>Grade: {data.get('behavior_grade', 'B')}</b>"), suit_html, contact, f"## 👤 Full Report: {name}")

def handle_applicant_selection(evt: gr.SelectData, df):
    if evt.index[1] != 4: return [gr.update()]*10
    name = df.iloc[evt.index[0], 0]
    # Return loading indicators immediately
    return ["⏳ Loading..."]*6 + [f"## 👤 Accessing: {name}", gr.update(visible=True), gr.Tabs(selected=4), name]

# --- GUI LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="HireSync AI Suite") as demo:
    res_txt_state = gr.State(""); hr_real_name_state = gr.State("")

    with gr.Tabs() as main_tabs:
        # --- CANDIDATE PORTAL ---
        with gr.TabItem("👤 CANDIDATE PORTAL"):
            with gr.Column(visible=True, elem_classes="candidate-box") as step_1_ui:
                gr.Markdown("<p class='step-header'>Step 1: Welcome & Login</p>")
                c_name = gr.Textbox(label="Full Name"); c_role = gr.Dropdown(label="Apply for Position", choices=db.get_job_titles())
                login_btn = gr.Button("Next ➡️", variant="primary"); c_status = gr.Markdown("Ready")
            with gr.Column(visible=False, elem_classes="candidate-box") as step_2_ui:
                gr.Markdown("<p class='step-header'>Step 2: Upload Profile</p>")
                c_email = gr.Textbox(label="Email Address"); c_phone = gr.Textbox(label="Phone Number"); c_res = gr.File(label="Resume (PDF)")
                upload_btn = gr.Button("Generate AI Interview ➡️", variant="primary")
            with gr.Column(visible=False, elem_classes="candidate-box") as step_3_ui:
                gr.Markdown("<p class='step-header'>Step 3: AI Interview</p>")
                c_q_area = gr.HTML(); c_vid = gr.Video(label="Record Your Response", sources=["webcam", "upload"],mirror_webcam=False,interactive=True)
                c_wait_msg = gr.HTML(visible=False); submit_btn = gr.Button("🚀 FINISH & SUBMIT", variant="primary")

        # --- HR ADMIN PORTAL ---
        with gr.TabItem("🏢 HR ADMIN PORTAL"):
            with gr.Tabs() as hr_tabs:
                with gr.TabItem("➕ Publish Job", id=1):
                    h_title = gr.Textbox(label="Job Title"); h_jd = gr.Textbox(label="Job Description", lines=4); h_q = gr.Textbox(label="Optional Q")
                    h_btn = gr.Button("Publish Position", variant="primary"); h_status = gr.Markdown("")
                with gr.TabItem("📋 Vacancies", id=2):
                    hr_j_refresh = gr.Button("🔄 REFRESH"); hr_j_table = gr.Dataframe(); hr_j_del = gr.Dropdown(label="Delete", choices=db.get_job_titles()); hr_j_del_btn = gr.Button("DELETE", variant="stop"); hr_j_del_status = gr.Markdown("")
                with gr.TabItem("📊 Applicant List", id=3):
                    hr_view_selector = gr.Dropdown(label="Filter Role", choices=db.get_job_titles()); hr_refresh_list = gr.Button("🔄 LOAD")
                    hr_main_table = gr.Dataframe(headers=["Name", "Role", "Score", "Date", "Analysis"], interactive=False)
                with gr.TabItem("🔍 Deep Review", id=4, visible=False) as deep_review_tab:
                    hr_back_btn = gr.Button("⬅️ BACK TO LIST")
                    with gr.Row():
                        with gr.Column(scale=1):
                            hr_name_display = gr.Markdown(); hr_integrity_box = gr.HTML(); hr_ats_display = gr.HTML()
                            hr_interview_display = gr.HTML(); hr_suitability_display = gr.HTML(); hr_contact_info = gr.Markdown()
                            hr_reload_btn = gr.Button("🔄 RE-LOAD DATA") 
                        with gr.Column(scale=1):
                            hr_xai_box = gr.HTML(); hr_transcript_display = gr.Textbox(label="Full Transcription", lines=12)

    # --- ACTIONS ---
    h_btn.click(hr_create_job, [h_title, h_jd, h_q], [h_status, c_role, hr_view_selector, hr_j_del])
    hr_j_refresh.click(db.get_all_positions_df, outputs=[hr_j_table])
    hr_j_del_btn.click(hr_delete_job, [hr_j_del], [hr_j_del_status, c_role, hr_j_del, hr_view_selector])
    
    login_btn.click(go_to_step_2, [c_name, c_role], [step_1_ui, step_2_ui, c_status])
    upload_btn.click(go_to_step_3, [c_res, c_role], [step_2_ui, step_3_ui, c_status, c_q_area, res_txt_state])
    c_vid.change(show_processing_warning, [c_vid], [c_wait_msg])
    submit_btn.click(final_candidate_submit, [c_name, c_role, c_email, c_phone, c_res, c_vid, res_txt_state, c_q_area], [c_status, step_1_ui, c_name, c_email, c_phone, c_res, c_vid]).then(lambda: [gr.update(visible=False), gr.update(visible=False)], None, [step_3_ui, c_wait_msg])

    hr_refresh_list.click(lambda r: db.get_candidates_by_role(r), [hr_view_selector], [hr_main_table])
    hr_main_table.select(handle_applicant_selection, [hr_main_table], [hr_transcript_display, hr_integrity_box, hr_ats_display, hr_interview_display, hr_suitability_display, hr_contact_info, hr_name_display, deep_review_tab, hr_tabs, hr_real_name_state]).then(load_full_candidate_info, [hr_real_name_state], [hr_transcript_display, hr_integrity_box, hr_ats_display, hr_interview_display, hr_suitability_display, hr_contact_info, hr_name_display])
    hr_reload_btn.click(load_full_candidate_info, [hr_real_name_state], [hr_transcript_display, hr_integrity_box, hr_ats_display, hr_interview_display, hr_suitability_display, hr_contact_info, hr_name_display])
    hr_back_btn.click(lambda: [gr.update(visible=False), gr.Tabs(selected=3)], outputs=[deep_review_tab, hr_tabs])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, share=False, inline=False)