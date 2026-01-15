import gradio_client.utils as client_utils

# --- 1. THE NUCLEAR PATCH (Python 3.14/3.12 Fix) ---
def patched_get_type(schema):
    if not isinstance(schema, dict): return "unknown"
    return "const" if "const" in schema else "enum" if "enum" in schema else "unknown"
client_utils.get_type = patched_get_type
client_utils.json_schema_to_python_type = lambda *args, **kwargs: "Any"

import gradio as gr
import pandas as pd
import PyPDF2
import os
from features import extract_features
from src.ats_matcher import ATSMatcher
import database as db
from static_ffmpeg import add_paths

add_paths()
matcher = ATSMatcher()

# --- HR: CREATE POSITION ---
def hr_create_job(title, jd_text, manual_q):
    if not title or not jd_text: return "⚠️ Missing info"
    questions = manual_q if manual_q.strip() else matcher.generate_questions(jd_text)
    db.add_job(title, jd_text, questions)
    return f"✅ Position '{title}' published!"

# --- CANDIDATE: SUBMIT APPLICATION ---
def submit_application(name, role, resume, video):
    if not all([name, role, resume, video]):
        return "⚠️ All fields required!", name, role, resume, video
    
    try:
        # 1. Fetch JD for the selected role
        target_jd = db.get_jd_by_title(role)
        
        # 2. Process Resume (Gemini)
        reader = PyPDF2.PdfReader(resume.name)
        resume_text = " ".join([p.extract_text() for p in reader.pages])
        ats_data = matcher.analyze_resume_llm(resume_text, target_jd)
        
        # 3. Process Video (Whisper/Hume)
        behavior = extract_features(None, None, video, skip_ats=True)
        
        # 4. Calculate Scores
        ats_val = ats_data['final_score']
        beh_val = behavior['behavior_score'] * 100 # Convert to 0-100
        final_val = round((ats_val * 0.6) + (beh_val * 0.4), 2)
        
        # 5. ACTUAL SAVE TO DATABASE
        db.save_candidate(
            name=name,
            role=role,
            ats=ats_val,
            behavior=beh_val,
            final=final_val,
            transcript=behavior['transcript']
        )

        
        return f"✅ Success! Application for {role} submitted.", "", None, None, None
    except Exception as e:
        print(f"Error: {e}")
        return f"❌ Error: {str(e)}", name, role, resume, video

# --- GUI ---
with gr.Blocks(theme=gr.themes.Soft(), css="footer {display:none !important;}") as demo:
    gr.Markdown("# 🤖 **HireSync AI Enterprise**")

    with gr.Tabs():
        # HR TAB
        with gr.TabItem("🏢 HR ADMIN PORTAL"):
            with gr.Tabs():
                with gr.TabItem("Create Job"):
                    title = gr.Textbox(label="Job Title")
                    jd = gr.Textbox(label="JD", lines=3)
                    q = gr.Textbox(label="Questions (Optional)")
                    btn_create = gr.Button("Publish")
                    status_create = gr.Markdown("")
                    btn_create.click(hr_create_job, [title, jd, q], [status_create])

                with gr.TabItem("View Applicants"):
                    btn_refresh = gr.Button("🔄 REFRESH CANDIDATE LIST")
                    # Dataframe component
                    hr_table = gr.Dataframe(label="Applicant Database")
                    # Trigger the database query
                    btn_refresh.click(db.get_all_candidates_df, outputs=[hr_table])

        # CANDIDATE TAB
        with gr.TabItem("👤 CANDIDATE PORTAL"):
            c_name = gr.Textbox(label="Name")
            # Update choices manually or refresh app to see new jobs
            c_role = gr.Dropdown(label="Select Job", choices=db.get_job_titles())
            c_res = gr.File(label="Resume")
            c_vid = gr.Video(label="Interview")
            c_btn = gr.Button("Submit")
            c_status = gr.Markdown("")
            
            c_btn.click(submit_application, [c_name, c_role, c_res, c_vid], [c_status, c_name, c_role, c_res, c_vid])

if __name__ == "__main__":
    demo.queue().launch(share=True)