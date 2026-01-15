import gradio_client.utils as client_utils

# --- 1. STABILITY PATCH FOR PYTHON 3.14/3.12 ---
def total_bypass_parser(*args, **kwargs): return "Any"
client_utils.json_schema_to_python_type = total_bypass_parser
client_utils._json_schema_to_python_type = total_bypass_parser
def patched_get_type(schema):
    if not isinstance(schema, dict): return "unknown"
    return "const" if "const" in schema else "enum" if "enum" in schema else "unknown"
client_utils.get_type = patched_get_type

import gradio as gr
import pandas as pd
import PyPDF2
import os
from features import extract_features
from src.ats_matcher import ATSMatcher
from database import save_candidate, get_all_candidates
from static_ffmpeg import add_paths

add_paths()
matcher = ATSMatcher()

# --- CUSTOM CSS ---
custom_css = """
footer { display: none !important; }
.gradio-container { background-color: #0b0f19 !important; }
.candidate-form { background: #111827; padding: 30px; border-radius: 15px; border: 1px solid #10b981; }
"""

# --- LOGIC: CANDIDATE PORTAL (Clear after Submit) ---
def candidate_submission(name, resume, jd, video):
    if not all([name, resume, jd, video]):
        # Return the error message and keep the current values in the inputs
        return "⚠️ Error: Please complete all fields before submitting.", name, resume, jd, video
    
    try:
        # 1. Validation
        is_v, err = matcher.validate_name(name)
        if not is_v: return err, name, resume, jd, video

        # 2. Extract and Process (Background)
        reader = PyPDF2.PdfReader(resume.name)
        resume_text = " ".join([p.extract_text() for p in reader.pages])
        with open(jd.name, 'r') as f: jd_text = f.read()
        
        ats_data = matcher.analyze_resume_llm(resume_text, jd_text)
        if not ats_data:
            return "❌ AI Engine Error. Try again.", name, resume, jd, video

        behavior_results = extract_features(None, None, video, skip_ats=True)
        
        ats_score = ats_data['final_score'] / 100
        final_score = round((ats_score * 0.6) + (behavior_results['behavior_score'] * 0.4), 2)
        
        # 3. SAVE TO DATABASE
        save_candidate(name, ats_score, behavior_results['behavior_score'], final_score, {
            "skills": ats_data['matched_skills'],
            "transcript": behavior_results['transcript'],
            "confidence": behavior_results.get('hume_confidence', 0),
            "anxiety": behavior_results.get('hume_anxiety', 0)
        })

        success_msg = f"✅ **SUCCESS!** Thank you {name}, your application has been securely submitted to HR."
        
        # 4. RETURN SUCCESS + CLEAR ALL INPUTS (Returning None/Empty string clears the UI)
        return success_msg, "", None, None, None

    except Exception as e:
        return f"❌ Submission Error: {str(e)}", name, resume, jd, video

# --- LOGIC: HR PORTAL ---
def refresh_hr_table():
    data = get_all_candidates()
    return pd.DataFrame(data, columns=["Name", "ATS Match", "Behavior Score", "Final Score", "Timestamp"])

# --- GUI LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="HireSync AI Pro") as demo:
    gr.Markdown("<h1 style='text-align:center; color:#38bdf8;'>🤖 HireSync AI: Recruitment System</h1>")

    with gr.Tabs():
        # --- CANDIDATE PORTAL ---
        with gr.TabItem("👤 CANDIDATE PORTAL"):
            with gr.Column(elem_classes="candidate-form"):
                gr.Markdown("### 📝 Application Submission\nFill the form and record your interview. Results are private to HR.")
                
                c_name = gr.Textbox(label="Full Name", placeholder="First and Last Name")
                with gr.Row():
                    c_res = gr.File(label="Upload Resume (PDF)")
                    c_jd = gr.File(label="Job Description Context (TXT)")
                c_vid = gr.Video(label="Record Interview Session")
                
                c_btn = gr.Button("🚀 SUBMIT APPLICATION", variant="primary", size="lg")
                c_status = gr.Markdown("Ready for upload.")
            
            # THE KEY CHANGE: Map the 5 return values to the 5 UI components
            c_btn.click(
                fn=candidate_submission, 
                inputs=[c_name, c_res, c_jd, c_vid], 
                outputs=[c_status, c_name, c_res, c_jd, c_vid]
            )

        # --- HR ADMIN PORTAL ---
        with gr.TabItem("🏢 HR ADMIN PORTAL"):
            gr.Markdown("### 📊 Applicant Tracking Database")
            hr_refresh = gr.Button("🔄 REFRESH AND VIEW APPLICANTS")
            hr_table = gr.Dataframe(label="Stored Applications")

    hr_refresh.click(refresh_hr_table, outputs=[hr_table])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, inline=False, share=True)