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
from features import extract_features, upload_to_cloud, save_candidate_to_supabase
from src.ats_matcher import ATSMatcher
import database as db
from static_ffmpeg import add_paths

# Initialize
add_paths()
matcher = ATSMatcher()

# --- CSS FOR PROFESSIONAL DASHBOARD ---
custom_css = """
footer { display: none !important; }
.gradio-container { background-color: #0b0f19 !important; }
.candidate-box { border: 1px solid #10b981 !important; padding: 20px; border-radius: 15px; background: #111827; }
.hr-box { border: 1px solid #38bdf8 !important; padding: 20px; border-radius: 15px; background: #111827; }
.submit-btn { background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important; color: white !important; }
"""

# --- HR LOGIC: POSITION MANAGEMENT ---
def hr_create_job(title, jd_text, manual_q):
    if not title or not jd_text:
        return "⚠️ Error: Title and JD are required.", gr.update(), gr.update(), gr.update()
    questions = manual_q if manual_q.strip() else matcher.generate_questions(jd_text)
    db.add_job(title, jd_text, questions)
    new_choices = db.get_job_titles()
    return f"✅ Position '{title}' published!", gr.update(choices=new_choices), gr.update(choices=new_choices), gr.update(choices=new_choices)

def hr_delete_job(title):
    if not title or title == "No Positions Available":
        return "⚠️ Select a valid position.", gr.update(), gr.update(), gr.update()
    msg = db.delete_job(title)
    new_choices = db.get_job_titles()
    return msg, gr.update(choices=new_choices), gr.update(choices=new_choices), gr.update(choices=new_choices)

# --- CANDIDATE LOGIC: SUBMISSION ---
def candidate_submission(name, role, resume, video):
    if not all([name, role, resume, video]):
        return "⚠️ All fields (Name, Role, Resume, Video) are required.", name, role, resume, video
    
    try:
        # 1. Validation
        is_v, err = matcher.validate_name(name)
        if not is_v: return f"❌ {err}", name, role, resume, video

        # 2. Extract context
        target_jd = db.get_jd_by_title(role)
        reader = PyPDF2.PdfReader(resume.name)
        resume_text = " ".join([p.extract_text() for p in reader.pages])
        
        # 3. Phase 1 AI Analysis (XAI Enabled)
        # This now returns a JSON with 'explanation' -> 'strengths', 'weaknesses', etc.
        ats_data = matcher.analyze_resume_llm(resume_text, target_jd)
        if not ats_data:
            return "❌ ATS Analysis failed. Please check Gemini API connection.", name, role, resume, video

        # 4. Phase 2 & Phase 5 Analysis (Behavioral + Integrity)
        behavior = extract_features(None, None, video, skip_ats=True)
        
        # 5. Scoring Math
        ats_val = ats_data['final_score']
        beh_val = behavior['behavior_score'] * 100
        final_val = round((ats_val * 0.6) + (beh_val * 0.4), 2)
        
        # 6. Cloud Upload
        # resource_type="image" for PDFs ensures they are viewable in browser
        res_url = upload_to_cloud(resume.name, resource_type="image")
        vid_url = upload_to_cloud(video, resource_type="video")

        # 7. BUILD EXPLAINABLE CANDIDATE RECORD
        candidate_record = {
            "name": name,
            "role": role,
            "ats_score": ats_val,
            "behavior_score": beh_val,
            "final_score": final_val,
            "transcript": behavior['transcript'],
            "resume_url": res_url,
            "video_url": vid_url,
            "behavior_grade": behavior.get('behavior_grade', 'B'),
            "hume_confidence": behavior.get('hume_confidence', 0.5),
            "hume_anxiety": behavior.get('hume_anxiety', 0.2),
            "integrity_status": behavior.get('integrity_status', 'Safe'),
            "conflict_report": behavior.get('conflict_report', 'Clean'),
            # --- XAI DATA INTEGRATION ---
            "details": {
                "strengths": ats_data.get('explanation', {}).get('strengths', 'N/A'),
                "weaknesses": ats_data.get('explanation', {}).get('weaknesses', 'N/A'),
                "verdict": ats_data.get('explanation', {}).get('verdict', 'N/A'),
                "matched": ats_data.get('matched_keywords', [])
            }
        }
        
        # 8. SAVE TO SUPABASE
        save_candidate_to_supabase(candidate_record)
        
        print(f"✅ Full Dossier saved for {name} with XAI Reasoning.")
        
        return f"✅ Application submitted successfully! Thank you, {name}.", "", None, None, None

    except Exception as e:
        print(f"Submission Error: {e}")
        return f"❌ Error: {str(e)}", name, role, resume, video
# --- HR LOGIC: DATA FETCHING & DYNAMIC DASHBOARD ---
def load_full_candidate_info(name):
    """MSc Logic: Safely retrieves dossier and handles XAI fallbacks for old records."""
    if not name:
        return [None]*7 + ["### ⚠️ Select a name"]
        
    data = db.get_candidate_details(name)
    if not data: 
        return [None]*7 + ["### ❌ Candidate not found"]
    
    # --- 1. SAFE XAI DATA PARSING ---
    details = data.get('details')
    # If details is a string (from database), parse it; if None, use empty dict
    if details is None:
        details = {}
    elif isinstance(details, str):
        try:
            import json
            details = json.loads(details)
        except:
            details = {}

    # Build the XAI HTML safely
    xai_html = f"""
    <div style="background:#1e293b; border-left: 5px solid #38bdf8; padding:15px; border-radius:10px; margin-top:10px;">
        <p style="color:#38bdf8; font-weight:bold; margin:0; font-size:14px;">🧠 AI REASONING (XAI)</p>
        <p style="color:white; font-size:13px; margin:5px 0;"><b>Strengths:</b> {details.get('strengths', 'Not available for older records')}</p>
        <p style="color:white; font-size:13px; margin:5px 0;"><b>Gaps:</b> {details.get('weaknesses', 'Not available for older records')}</p>
        <p style="color:#10b981; font-size:13px; margin:5px 0;"><b>Verdict:</b> {details.get('verdict', 'Data pending...')}</p>
    </div>
    """

    # --- 2. INTEGRITY CHECK (Phase 5) ---
    status = data.get('integrity_status', 'Safe')
    color = "#ef4444" if status == "High" else "#f59e0b" if status == "Medium" else "#22c55e"
    integrity_html = f"""
    <div style="background:#111827; border: 2px solid {color}; padding:15px; border-radius:10px; margin-top:10px;">
        <p style="color:{color}; font-weight:bold; margin:0; font-size:12px;">🛡️ BEHAVIORAL INTEGRITY CHECK: {status.upper()}</p>
        <p style="color:white; font-size:13px; margin:5px 0;">{data.get('conflict_report', 'No significant behavioral discrepancies detected.')}</p>
    </div>
    """

    # --- 3. RESUME & METRICS ---
    resume_link = f"### 📄 [Click to View Resume PDF]({data['resume_url']})"
    analysis_text = f"""
    - **Behavior Grade:** {data.get('behavior_grade', 'N/A')}
    - **Candidate Confidence:** {data.get('hume_confidence', 0)}
    - **Anxiety Level:** {data.get('hume_anxiety', 0)}
    """
    
    # Returns 8 values to populate the UI components
    return (
        data['video_url'], 
        resume_link, 
        data['transcript'], 
        f"{data['ats_score']}%", 
        analysis_text,
        integrity_html,
        xai_html,
        f"## 👤 Reviewing: {name}"
    )

def handle_applicant_selection(evt: gr.SelectData, df):
    """Triggered on table click. Logic: Column 4 is the 'Deep Analysis' button."""
    if evt.index[1] != 4:  # If the user didn't click the 'Deep Analysis' column
        return [gr.update()]*10

    candidate_name = df.iloc[evt.index[0], 0]
    results = load_full_candidate_info(candidate_name)
    
    # Combine the 8 data results with the 2 Tab-control commands
    return results + (gr.update(visible=True), gr.Tabs(selected=4))

def close_deep_review():
    return gr.update(visible=False), gr.Tabs(selected=3)

def update_filtered_table(selected_role):
    if not selected_role or selected_role == "No Positions Available":
        return pd.DataFrame(columns=["Status"], data=[["Select a position above"]])
    df = db.get_candidates_by_role(selected_role)
    if df.empty: return pd.DataFrame(columns=["Status"], data=[["No applicants yet"]])
    return df

# --- GUI LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="HireSync AI Pro") as demo:
    gr.Markdown("<h1 style='text-align:center; color:#38bdf8;'>🤖 HireSync AI Suite</h1>")

    with gr.Tabs() as main_tabs:
        with gr.TabItem("👤 CANDIDATE PORTAL"):
            with gr.Column(elem_classes="candidate-box"):
                gr.Markdown("### 📝 Job Application Form")
                c_name = gr.Textbox(label="Full Name")
                c_role = gr.Dropdown(label="Select Position", choices=db.get_job_titles())
                with gr.Row():
                    c_res = gr.File(label="Upload Resume (PDF)")
                    c_vid = gr.Video(label="Interview Session")
                c_btn = gr.Button("🚀 SUBMIT APPLICATION", variant="primary", elem_classes="submit-btn")
                c_status = gr.Markdown("Ready.")

        with gr.TabItem("🏢 HR ADMIN PORTAL"):
            with gr.Tabs() as hr_tabs:
                with gr.TabItem("➕ Publish Job", id=1):
                    with gr.Column(elem_classes="hr-box"):
                        h_title = gr.Textbox(label="Job Title"); h_jd = gr.Textbox(label="JD", lines=4)
                        h_q = gr.Textbox(label="Custom Questions", lines=2)
                        h_btn = gr.Button("Publish Position", variant="primary"); h_status = gr.Markdown("")

                with gr.TabItem("📋 Manage Vacancies", id=2):
                    with gr.Column(elem_classes="hr-box"):
                        hr_job_refresh = gr.Button("🔄 REFRESH VACANCY LIST")
                        hr_job_table = gr.Dataframe(label="Active Openings")
                        hr_delete_select = gr.Dropdown(label="Delete Position", choices=db.get_job_titles())
                        hr_delete_btn = gr.Button("DELETE POSITION", variant="stop"); hr_delete_status = gr.Markdown("")

                with gr.TabItem("📊 Applicant List", id=3):
                    gr.Markdown("### 🔍 Filter Applicants by Role")
                    with gr.Row():
                        hr_view_selector = gr.Dropdown(label="Select Role", choices=db.get_job_titles())
                        hr_refresh_list = gr.Button("🔄 REFRESH", variant="secondary")
                    hr_main_table = gr.Dataframe(interactive=False)

                with gr.TabItem("🔍 Deep Review", id=4, visible=False) as deep_review_tab:
                    with gr.Column(elem_classes="hr-box"):
                        hr_back_btn = gr.Button("⬅️ BACK TO LIST", variant="secondary")
                        with gr.Row():
                            with gr.Column(scale=1):
                                hr_name_display = gr.Markdown("## 👤 Reviewing: ...") 
                                hr_ats_label = gr.Label(label="ATS Score")
                                hr_integrity_box = gr.HTML() # Phase 5 Box
                                hr_xai_box = gr.HTML() # <--- Add this component
                                hr_extra_stats = gr.Markdown("### Metrics")
                                hr_pdf_display = gr.Markdown("### Resume Link")
                            with gr.Column(scale=2):
                                hr_video_display = gr.Video(label="Interview Cloud Playback")
                                hr_transcript_display = gr.Textbox(label="Transcription", lines=10)

    # --- EVENT MAPPING ---
    h_btn.click(hr_create_job, [h_title, h_jd, h_q], [h_status, c_role, hr_delete_select, hr_view_selector])
    hr_job_refresh.click(db.get_all_positions_df, outputs=[hr_job_table])
    hr_delete_btn.click(hr_delete_job, [hr_delete_select], [hr_delete_status, c_role, hr_delete_select, hr_view_selector])
    c_btn.click(candidate_submission, [c_name, c_role, c_res, c_vid], [c_status, c_name, c_role, c_res, c_vid])
    hr_view_selector.change(update_filtered_table, [hr_view_selector], [hr_main_table])
    hr_refresh_list.click(update_filtered_table, [hr_view_selector], [hr_main_table])

    # Find this line in the 'EVENT MAPPING' section of main.py
    hr_main_table.select(
        fn=handle_applicant_selection,
        inputs=[hr_main_table],
        outputs=[
            hr_video_display,     # 1
            hr_pdf_display,       # 2
            hr_transcript_display, # 3
            hr_ats_label,          # 4
            hr_extra_stats,        # 5
            hr_integrity_box,      # 6
            hr_xai_box,            # 7
            hr_name_display,       # 8
            deep_review_tab,       # 9 (Tab Visibility)
            hr_tabs                # 10 (Tab Switch)
        ]
    )
    hr_back_btn.click(fn=close_deep_review, outputs=[deep_review_tab, hr_tabs])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, share=True, inline=False)