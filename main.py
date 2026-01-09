import gradio as gr
import pandas as pd
from features import extract_features

def hiresync_ui_logic(resume_pdf, jd_file, video_input, candidate_name):
    if not all([resume_pdf, jd_file, video_input, candidate_name]):
        return "⚠️ Error: Please provide Name, Resume, JD, and Video.", None
    
    # Process everything
    results = extract_features(resume_pdf.name, jd_file.name, video_input)
    
    # Professional HTML Dashboard
    html = f"""
    <div style="background:#0f172a; color:white; padding:25px; border-radius:15px; border:2px solid #38bdf8; font-family:sans-serif;">
        <h2 style="margin:0; color:#38bdf8;">Report Card: {candidate_name}</h2>
        <hr style="border:0.1px solid #334155; margin:15px 0;">
        <div style="display:flex; justify-content:space-around; margin-bottom:20px;">
            <div style="text-align:center;">
                <p style="margin:0; color:#94a3b8; font-size:12px;">ATS MATCH</p>
                <h1 style="margin:0; color:#38bdf8;">{results['suitability']*100:.1f}%</h1>
            </div>
            <div style="text-align:center;">
                <p style="margin:0; color:#94a3b8; font-size:12px;">BEHAVIOR SCORE</p>
                <h1 style="margin:0; color:#38bdf8;">{results['final_score']:.2f}</h1>
            </div>
        </div>
        <p><b>Analysis:</b> {results['transcript'][:150]}...</p>
    </div>
    """
    
    history = pd.DataFrame([[candidate_name, results['final_score']]], columns=["Candidate", "Score"])
    return html, history

# --- GRADIO INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft(), title="HireSync AI") as demo:
    gr.Markdown("# 🤖 **HireSync AI: Multimodal Dashboard**")
    
    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="Candidate Name")
            res = gr.File(label="Resume (PDF)")
            jd = gr.File(label="Job Description (TXT)")
            vid = gr.Video(label="Record or Upload Video", sources=["webcam", "upload"])
            btn = gr.Button("🚀 RUN FULL ANALYSIS", variant="primary")
            
        with gr.Column():
            out_html = gr.HTML("Awaiting Input...")
            out_df = gr.Dataframe(label="Evaluation History")

    btn.click(hiresync_ui_logic, [res, jd, vid, name], [out_html, out_df])

if __name__ == "__main__":
    demo.queue().launch(
        share=False,      # Set to False to keep it on local secure context
        inline=False,     # DISABLLES the VS Code internal browser
        inbrowser=True,   # FORCES Chrome or Edge to open
        debug=True
    )