import gradio_client.utils as client_utils
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

# --- 1. THE NUCLEAR PATCH (Python 3.14/3.12 Stability) ---
def patched_get_type(schema):
    if not isinstance(schema, dict): 
        return "unknown"
    return "const" if "const" in schema else "enum" if "enum" in schema else "unknown"

client_utils.get_type = patched_get_type
client_utils.json_schema_to_python_type = lambda *args, **kwargs: "Any"

# Initialize
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

# --- HR LOGIC: VACANCY MANAGEMENT ---
def hr_create_job(title, jd_text, manual_q):
    if not title or not jd_text: 
        return "⚠️ Error: Title/JD required.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    questions = manual_q if manual_q.strip() else matcher.generate_questions(jd_text)
    db.add_job(title, jd_text, questions)
    new_choices = db.get_job_titles()
    return (f"✅ Position '{title}' published!", "", "", "", gr.update(choices=new_choices), gr.update(choices=new_choices), gr.update(choices=new_choices))

def hr_delete_job(title):
    if not title or title == "No Positions Available": 
        return "⚠️ Select a valid position.", gr.update(), gr.update(), gr.update()
    msg = db.delete_job(title)
    new_choices = db.get_job_titles()
    return msg, gr.update(choices=new_choices), gr.update(choices=new_choices), gr.update(choices=new_choices)

# --- CANDIDATE FLOW LOGIC ---
def go_to_step_2(name, role):
    if not name or not role: 
        return gr.update(visible=True), gr.update(visible=False), "⚠️ Name and Role required."
    is_v, err = matcher.validate_name(name)
    if not is_v: 
        return gr.update(visible=True), gr.update(visible=False), err
    return gr.update(visible=False), gr.update(visible=True), ""

def go_to_step_3(resume, role_title):
    if not resume: 
        return gr.update(visible=True), gr.update(visible=False), "⚠️ Upload Resume.", "", ""
    try:
        reader = PyPDF2.PdfReader(resume.name)
        resume_text = " ".join([page.extract_text() for page in reader.pages])
        target_jd = db.get_jd_by_title(role_title)
        ai_questions = matcher.generate_questions_from_jd(target_jd)
        q_html = f"<div class='agent-box'><h3 style='color:#10b981; margin:0 0 10px 0;'>📋 AI Interview Assessment</h3><p style='color:white; white-space: pre-wrap; font-size:16px; line-height:1.6;'>{ai_questions}</p></div>"
        return gr.update(visible=False), gr.update(visible=True), "✅ Interview Ready", q_html, resume_text
    except Exception as e: 
        return gr.update(visible=True), gr.update(visible=False), f"❌ {e}", "", ""

# --- UPDATED BACKGROUND TASK ---
def background_analysis_task(name, role, email, phone, resume_path, video_path, resume_text, questions_html):
    try:
        print(f"⚙️ [BACKGROUND] Starting AI processing for {name}...")
        time.sleep(10) # Standard file stability wait
        
        raw_q = questions_html.replace("<div class='agent-box'>", "").replace("</div>", "").replace("AI Interview Assessment", "")
        target_jd = db.get_jd_by_title(role)
        
        # 1. Run AI Engines
        ats_data = matcher.analyze_resume_llm(resume_text, target_jd)
        behavior = extract_features(resume_path, target_jd, video_path, skip_ats=True, questions=raw_q)
        
        # 2. Extract specific values for the Verdict
        ats_val = ats_data.get('final_score', 0)
        beh_val = behavior.get('behavior_score', 0) * 100
        
        # 3. GENERATE THE XAI VERDICT (Fixed the "Missing Arguments" error)
        print(f"🧠 Generating Explainable AI Verdict for {name}...")
        final_verdict = matcher.generate_final_verdict_xai(
            name=name,
            ats_score=ats_val,
            behavior_score=beh_val,
            behavior_grade=behavior.get('behavior_grade', 'B'),
            strengths=", ".join(ats_data.get('matched_keywords', [])),
            gaps=ats_data.get('explanation', {}).get('weaknesses', 'N/A'),
            integrity=behavior.get('integrity_status', 'Safe'),
            transcript=behavior.get('transcript', '')
        )

        # 4. Final suitability calculation (60/40 weighted)
        weighted_base = (ats_val * 0.6) + (beh_val * 0.4)
        
        # If the candidate fails the interview (beh_val < 40), 
        # we apply a "Disqualification Multiplier"
        if beh_val < 40:
            # The lower the behavior, the more the technical score is "destroyed"
            penalty_factor = beh_val / 40  # e.g., 25 / 40 = 0.625
            final_val = round(weighted_base * penalty_factor, 1)
        else:
            final_val = round(weighted_base, 1)
            
        print(f"🎯 Aggressive Scoring Applied: {final_val}")


        # 5. Cloud Uploads
        res_url = upload_to_cloud(resume_path, resource_type="image")
        vid_url = upload_to_cloud(video_path, resource_type="video")

        # 6. Save to Supabase
        record = {
            "name": name, "role": role, "email": email, "phone": phone,
            "ats_score": ats_val, 
            "behavior_score": beh_val, 
            "final_score": final_val,
            "transcript": behavior.get('transcript', ''), 
            "resume_url": res_url, "video_url": vid_url,
            "behavior_grade": behavior.get('behavior_grade', 'B'),
            "hume_confidence": behavior.get('hume_confidence', 0),
            "attention": behavior.get('attention', 0),
            "integrity_status": behavior.get('integrity_status', 'Safe'),
            "conflict_report": behavior.get('conflict_report', 'Clean'),
            "details": ats_data.get('explanation', {}),
            "final_reasoning_text": final_verdict # Now populated with real AI reasoning
        }
        save_candidate_to_supabase(record)
        print(f"✅ [SUCCESS] Application for {name} saved to Cloud.")
    except Exception as e:
        import traceback
        print(f"❌ [CRITICAL BACKGROUND ERROR] {e}")
        traceback.print_exc()
        
def final_candidate_submit(name, role, email, phone, resume, video, resume_text, questions_html):
    if not video: 
        return "⚠️ Record Video", gr.update(visible=True), name, email, phone, resume, video
    threading.Thread(target=background_analysis_task, args=(name, role, email, phone, resume.name, video, resume_text, questions_html)).start()
    return "✅ **Registration Successful!** You can now close this tab.", gr.update(visible=True), "", "", "", None, None

def show_processing_warning(video_data):
    if video_data is not None:
        return gr.update(value='<div class="wait-warning">⚠️ Finalizing Recording... Please wait for the Play icon before clicking Submit.</div>', visible=True)
    return gr.update(visible=False)

# --- HR LOGIC: ADVANCED FORENSIC DASHBOARD ---
def load_full_candidate_info(name):
    print(f"📡 HR Generating Detailed Report for: {name}")
    data = db.get_candidate_details(name)
    if not data: 
        return ["No Data"] * 6 + [f"## ❌ Record not found for {name}"]

    details = data.get('details') or {}
    if isinstance(details, str): 
        details = json.loads(details)

    # 1. Integrity Narrative
    raw_status = str(data.get('integrity_status', 'Safe')).upper()
    color = "#ef4444" if raw_status == "HIGH" else "#f59e0b" if raw_status == "MEDIUM" else "#22c55e"
    integrity_html = f"""<div style="background:#111827; border-left: 5px solid {color}; padding:15px; border-radius:10px; margin-bottom:15px;">
        <h3 style="color:{color}; margin:0;">🛡️ INTEGRITY AUDIT: {raw_status}</h3>
        <p style="color:white; font-size:13px; line-height:1.4;"><b>AI Observation:</b> {data.get('conflict_report', 'Cues aligned.')}</p>
    </div>"""

    # 2. ATS Narrative
    ats_score = data.get('ats_score', 0)
    ats_html = f"""<div style="background:#0f172a; border-left: 5px solid #38bdf8; padding:15px; border-radius:10px; margin-bottom:15px;">
        <h3 style="color:#38bdf8; margin:0;">📄 DOCUMENT INTELLIGENCE (ATS)</h3>
        <p style="color:white; font-size:14px; margin:10px 0;"><b>Score: {ats_score}% Match</b></p>
        <p style="color:white; font-size:12px;"><b>Strengths:</b> {details.get('strengths', 'N/A')}<br><b>Gaps:</b> {details.get('weaknesses', 'N/A')}</p>
    </div>"""

    # 3. Behavioral Narrative
    beh_score = data.get('behavior_score', 0)
    conf_v = round(float(data.get('hume_confidence', 0)), 2)
    attn_v = round(float(data.get('attention', 0)), 2)
    
    interview_html = f"""<div style="background:#0f172a; border-left: 5px solid #a855f7; padding:15px; border-radius:10px; margin-bottom:15px;">
        <h3 style="color:#a855f7; margin:0;">🎤 PERFORMANCE ANALYSIS</h3>
        <p style="color:white; font-size:13px; line-height:1.4;">Candidate achieved a <b>Grade {data.get('behavior_grade', 'B')}</b> with a behavior index of <b>{beh_score}%</b>.</p>
        <p style="color:#94a3b8; font-size:11px;">Confidence: {conf_v} | Attention: {attn_v}</p>
    </div>"""

    # 4. Suitability Index & Final Verdict
    final_score = int(data.get('final_score', 0))
    s_color = "#22c55e" if final_score > 75 else "#f59e0b" if final_score > 50 else "#ef4444"
    verdict_text = data.get('final_reasoning_text', 'Analysis based on multimodal inputs.')

    suit_html = f"""<div style="background:#1e293b; padding:25px; border-radius:15px; border: 3px solid {s_color}; text-align:center;">
        <h3 style="color:#94a3b8; margin:0; font-size:12px;">OVERALL SUITABILITY INDEX</h3>
        <h1 style="color:white; font-size:60px; margin:10px 0;">{final_score}/100</h1>
        <p style="color:white; font-size:13px; line-height:1.6; max-width:480px; margin:auto;"><b>Final AI Verdict:</b> {verdict_text}</p>
    </div>"""

    # 5. Contacts & Asset Dossier
    contact = f"Email: {data.get('email', 'N/A')} | Phone: {data.get('phone', 'N/A')}\n\n[📄 Open Resume PDF]({data.get('resume_url', '#')})\n[📹 Watch Interview Video]({data.get('video_url', '#')})"
    
    return (data.get('transcript', ''), integrity_html, ats_html, interview_html, suit_html, contact, f"## 👤 Full Report: {name}")

def handle_applicant_selection(evt: gr.SelectData, df):
    if evt.index[1] != 4: 
        return [gr.update()]*10
    name = df.iloc[evt.index[0], 0]
    return ["⏳ Loading Transcript..."] + ["### ⏳ Fetching data..."]*5 + [f"## 👤 Accessing: {name}", gr.update(visible=True), gr.Tabs(selected=4), name]

# --- GUI LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="HireSync AI Pro") as demo:
    res_txt_state = gr.State(""); hr_real_name_state = gr.State("")

    with gr.Tabs() as main_tabs:
        # PORTAL 1: CANDIDATE
        with gr.TabItem("👤 CANDIDATE PORTAL"):
            with gr.Column(visible=True, elem_classes="candidate-box") as step_1_ui:
                gr.Markdown("<p class='step-header'>Step 1: Login</p>")
                c_name = gr.Textbox(label="Full Name")
                c_role = gr.Dropdown(label="Position", choices=db.get_job_titles())
                login_btn = gr.Button("Next ➡️", variant="primary")
                c_status = gr.Markdown("Ready")
            with gr.Column(visible=False, elem_classes="candidate-box") as step_2_ui:
                gr.Markdown("<p class='step-header'>Step 2: Upload Resume</p>")
                c_email = gr.Textbox(label="Email")
                c_phone = gr.Textbox(label="Phone")
                c_res = gr.File(label="Resume (PDF)")
                upload_btn = gr.Button("Next ➡️", variant="primary")
            with gr.Column(visible=False, elem_classes="candidate-box") as step_3_ui:
                gr.Markdown("<p class='step-header'>Step 3: AI Interview</p>")
                c_q_area = gr.HTML()
                c_vid = gr.Video(label="Record Response", sources=["webcam", "upload"], interactive=True, autoplay=False)
                c_wait_msg = gr.HTML(visible=False)
                submit_btn = gr.Button("🚀 FINISH & SUBMIT", variant="primary")

        # PORTAL 2: HR ADMIN
        with gr.TabItem("🏢 HR ADMIN PORTAL"):
            with gr.Tabs() as hr_tabs:
                with gr.TabItem("➕ Publish Job", id=1):
                    with gr.Column(elem_classes="hr-box"):
                        h_title = gr.Textbox(label="Job Title")
                        h_jd = gr.Textbox(label="JD", lines=4)
                        h_q = gr.Textbox(label="Questions", lines=1)
                        h_btn = gr.Button("Publish")
                        h_status = gr.Markdown("")
                with gr.TabItem("📋 Manage Vacancies", id=2):
                    with gr.Column(elem_classes="hr-box"):
                        hr_job_refresh = gr.Button("🔄 REFRESH")
                        hr_job_table = gr.Dataframe()
                        hr_delete_select = gr.Dropdown(label="Delete Position", choices=db.get_job_titles())
                        hr_delete_btn = gr.Button("DELETE", variant="stop")
                        hr_delete_status = gr.Markdown("")
                with gr.TabItem("📊 Applicant List", id=3):
                    hr_view_selector = gr.Dropdown(label="Filter Role", choices=db.get_job_titles())
                    hr_refresh_list = gr.Button("🔄 LOAD")
                    hr_main_table = gr.Dataframe(headers=["Name", "Role", "Score", "Date", "Analysis"], interactive=False)
                with gr.TabItem("🔍 Deep Review", id=4, visible=False) as deep_review_tab:
                    with gr.Column(elem_classes="hr-box"):
                        hr_back_btn = gr.Button("⬅️ BACK TO LIST")
                        with gr.Row():
                            with gr.Column(scale=1):
                                hr_name_display = gr.Markdown()
                                hr_integrity_box = gr.HTML()
                                hr_ats_display = gr.HTML()
                                hr_interview_display = gr.HTML()
                                hr_suitability_display = gr.HTML()
                                hr_contact_info = gr.Markdown()
                                hr_reload_btn = gr.Button("🔄 RE-LOAD DATA") 
                            with gr.Column(scale=1):
                                hr_xai_box = gr.HTML()
                                hr_transcript_display = gr.Textbox(label="Full Transcription", lines=12)

    # --- ACTION MAPPINGS ---
    h_btn.click(hr_create_job, [h_title, h_jd, h_q], [h_status, h_title, h_jd, h_q, c_role, hr_view_selector, hr_delete_select])
    hr_job_refresh.click(db.get_all_positions_df, outputs=[hr_job_table])
    hr_delete_btn.click(hr_delete_job, [hr_delete_select], [hr_delete_status, c_role, hr_delete_select, hr_view_selector])
    
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