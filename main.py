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
        return "⚠️ Error: Title and JD are required.", gr.update(), gr.update()
    
    questions = manual_q if manual_q.strip() else matcher.generate_questions(jd_text)
    db.add_job(title, jd_text, questions)
    
    new_choices = db.get_job_titles()
    # Returns status, updates candidate dropdown, and updates the delete dropdown
    return f"✅ Position '{title}' published!", gr.update(choices=new_choices), gr.update(choices=new_choices)

def hr_delete_job(title):
    if not title or title == "No Positions Available":
        return "⚠️ Please select a valid position.", gr.update(), gr.update()
    
    msg = db.delete_job(title)
    new_choices = db.get_job_titles()
    return msg, gr.update(choices=new_choices), gr.update(choices=new_choices)

# --- CANDIDATE LOGIC: SUBMISSION ---
def candidate_submission(name, role, resume, video):
    if not all([name, role, resume, video]):
        return "⚠️ All fields required.", name, role, resume, video
    
    try:
        is_v, err = matcher.validate_name(name)
        if not is_v: return f"❌ {err}", name, role, resume, video

        target_jd = db.get_jd_by_title(role)
        reader = PyPDF2.PdfReader(resume.name)
        resume_text = " ".join([p.extract_text() for p in reader.pages])
        
        ats_data = matcher.analyze_resume_llm(resume_text, target_jd)
        behavior = extract_features(None, None, video, skip_ats=True)
        
        ats_val = ats_data['final_score']
        beh_val = behavior['behavior_score'] * 100
        final_val = round((ats_val * 0.6) + (beh_val * 0.4), 2)
        
        res_url = upload_to_cloud(resume.name, resource_type="raw")
        vid_url = upload_to_cloud(video, resource_type="video")

        candidate_record = {
            "name": name, "role": role, "ats_score": ats_val,
            "behavior_score": beh_val, "final_score": final_val,
            "transcript": behavior['transcript'], "resume_url": res_url,
            "video_url": vid_url, "behavior_grade": behavior.get('behavior_grade', 'B'),
            "hume_confidence": behavior.get('hume_confidence', 0.5),
            "hume_anxiety": behavior.get('hume_anxiety', 0.2)
        }
        save_candidate_to_supabase(candidate_record)

        return f"✅ Application for '{role}' submitted successfully.", "", None, None, None
    except Exception as e:
        return f"❌ Error: {str(e)}", name, role, resume, video

# --- HR LOGIC: DATA FETCHING ---
def load_full_candidate_info(name):
    if not name: return None, "### ⚠️ Select a name", "No data", "0%", "N/A"
    data = db.get_candidate_details(name)
    if not data: return None, "### ❌ Not found", "No data", "0%", "N/A"
    
    resume_link = f"### 📄 [Click to View Resume PDF]({data['resume_url']})"
    analysis_text = f"- **Grade:** {data.get('behavior_grade', 'N/A')}\n- **Confidence:** {data.get('hume_confidence', 0)}\n- **Anxiety:** {data.get('hume_anxiety', 0)}"
    
    return data['video_url'], resume_link, data['transcript'], f"{data['ats_score']}%", analysis_text

def refresh_hr_view():
    df = db.get_cloud_candidates_df()
    names = df["Name"].tolist() if not df.empty else []
    return df, gr.update(choices=names)

# --- GUI LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="HireSync AI Pro") as demo:
    gr.Markdown("<h1 style='text-align:center; color:#38bdf8;'>🤖 HireSync AI Suite</h1>")

    with gr.Tabs():
        # --- PORTAL 1: CANDIDATE ---
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

        # --- PORTAL 2: HR ADMIN ---
        with gr.TabItem("🏢 HR ADMIN PORTAL"):
            with gr.Tabs():
                # Tab 1: Publish Job
                with gr.TabItem("➕ Publish Job"):
                    with gr.Column(elem_classes="hr-box"):
                        h_title = gr.Textbox(label="Job Title")
                        h_jd = gr.Textbox(label="Job Description", lines=4)
                        h_q = gr.Textbox(label="Questions (Optional)", lines=2)
                        h_btn = gr.Button("Publish Position", variant="primary")
                        h_status = gr.Markdown("")

                # Tab 2: Manage Vacancies (NEW)
                with gr.TabItem("📋 Manage Vacancies"):
                    with gr.Column(elem_classes="hr-box"):
                        gr.Markdown("### Active Job Openings")
                        hr_job_refresh = gr.Button("🔄 REFRESH VACANCY LIST")
                        hr_job_table = gr.Dataframe(interactive=False)
                        gr.Markdown("---")
                        gr.Markdown("### 🗑️ Remove a Position")
                        hr_delete_select = gr.Dropdown(label="Select Position to Delete", choices=db.get_job_titles())
                        hr_delete_btn = gr.Button("DELETE POSITION", variant="stop")
                        hr_delete_status = gr.Markdown("")

                # Tab 3: Applicant List
                with gr.TabItem("📊 Applicant List"):
                    hr_refresh_btn = gr.Button("🔄 REFRESH FROM CLOUD")
                    hr_main_table = gr.Dataframe(interactive=False)

                # Tab 4: Deep Review
                with gr.TabItem("🔍 Candidate Deep Review"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            hr_selector = gr.Dropdown(label="Select Candidate", choices=[])
                            hr_load_btn = gr.Button("📂 LOAD FULL DETAILS", variant="primary")
                            hr_ats_label = gr.Label(label="ATS Match")
                            hr_extra_stats = gr.Markdown("Behavioral Metrics...")
                            hr_pdf_display = gr.Markdown("Resume Link...")
                        with gr.Column(scale=2):
                            hr_video_display = gr.Video(label="Interview Cloud Playback")
                            hr_transcript_display = gr.Textbox(label="AI Transcription", lines=10)

    # --- EVENT MAPPING ---
    
    # HR: Publish Job
    h_btn.click(hr_create_job, [h_title, h_jd, h_q], [h_status, c_role, hr_delete_select])
    
    # HR: Manage Jobs
    hr_job_refresh.click(db.get_all_positions_df, outputs=[hr_job_table])
    hr_delete_btn.click(hr_delete_job, [hr_delete_select], [hr_delete_status, c_role, hr_delete_select])
    
    # Candidate: Submit
    c_btn.click(candidate_submission, [c_name, c_role, c_res, c_vid], [c_status, c_name, c_role, c_res, c_vid])
    
    # HR: Applicant Data
    hr_refresh_btn.click(refresh_hr_view, outputs=[hr_main_table, hr_selector])
    hr_load_btn.click(load_full_candidate_info, [hr_selector], [hr_video_display, hr_pdf_display, hr_transcript_display, hr_ats_label, hr_extra_stats])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, share=True, inline=False)