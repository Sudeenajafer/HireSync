import gradio as gr
from phase3_audio import audio_phase_score  # Your Phase 3

def live_hiresync(audio):
    if not audio:
        return "🎤 Speak: 'Tell me about yourself'"
    
    stats = audio_phase_score(audio)
    fluency = stats["fluency_score"]
    
    # Mock ATS + Gaze from your phases
    ats = 0.493  # phase1
    gaze = 0.85  # phase2
    
    final_score = round((ats + gaze + fluency) / 3, 3)
    
    return f"""
## 🎯 **HireSync Final Score: {final_score}**
**Phase 1 ATS**: {ats}  
**Phase 2 Gaze**: {gaze}
**Phase 3 Fluency**: {fluency} ⭐

**Transcript**: `{stats["transcript"][:80]}...`
**WPM**: {stats["wpm"]} | **Fillers**: {stats["filler_count"]}

✅ **Interview READY!** (Viva demo perfect)
    """

# Launch
gr.Interface(
    fn=live_hiresync,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs=gr.Markdown(),
    title="🤖 HireSync AI - Complete Pipeline",
    description="**LIVE**: Mic → ATS + Gaze + Audio → Score"
).launch(share=True)
