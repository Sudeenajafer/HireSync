import gradio as gr
import PyPDF2
import plotly.graph_objects as go

def create_radar_chart(data):
    fig = go.Figure()
    categories = ['Education', 'Skills', 'Experience']
    scores = [data['education_score'], data['skills_score'], data['experience_score']]
    fig.add_trace(go.Scatterpolar(r=scores + [scores[0]], theta=categories + [categories[0]], fill='toself', line_color='#10b981'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, paper_bgcolor='rgba(0,0,0,0)', font_color="white")
    return fig

def run_candidate_eval(name, res_file, jd_file, matcher):
    if not all([name, res_file, jd_file]): return "⚠️ Inputs missing", None
    
    is_v, err = matcher.validate_name(name)
    if not is_v: return f"<div>{err}</div>", None

    reader = PyPDF2.PdfReader(res_file.name)
    resume_text = " ".join([p.extract_text() for p in reader.pages])
    with open(jd_file.name, 'r') as f: jd_text = f.read()

    res = matcher.analyze_resume_llm(resume_text, jd_text)
    if not res: return "❌ AI Analysis Failed", None

    chart = create_radar_chart(res)
    html = f"""<div class='report-card' style='border-color:#10b981;'>
        <h3 style='color:#10b981; margin:0;'>Candidate Self-Check: {res['final_score']}%</h3>
        <p style='color:white; font-size:14px;'>{res['reasoning']}</p>
        <p style='color:#94a3b8; font-size:12px;'><b>MSc Tip:</b> {res['suggestions']}</p>
    </div>"""
    return html, chart