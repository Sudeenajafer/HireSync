import whisper
import os
import re
import librosa
import numpy as np
import subprocess
import time

# --- HIGH-PRECISION WHISPER ---
print("⏳ Loading High-Precision Whisper Model (Small)...")
model = whisper.load_model("small") 
FILLERS = ["um", "uh", "ah", "like", "basically", "actually", "you know"]

def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def audio_phase_score(video_path):
    temp_wav = os.path.join(os.getcwd(), f"temp_proc_{int(time.time())}.wav")
    
    try:
        # 1. EXTRACT AUDIO
        command = f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{temp_wav}"'
        subprocess.run(command, shell=True, capture_output=True)

        if not os.path.exists(temp_wav):
            return {"transcript": "[Extraction Failed]", "duration": "00:00", "fluency_score": 0.0}

        # 2. ACOUSTIC ANALYSIS (Duration & Silence)
        y, sr = librosa.load(temp_wav)
        duration_seconds = librosa.get_duration(y=y, sr=sr)
        
        # Detect non-silent intervals to calculate silence ratio
        non_silent_intervals = librosa.effects.split(y, top_db=25)
        speech_duration = sum([itv[1] - itv[0] for itv in non_silent_intervals]) / sr
        silence_ratio = (duration_seconds - speech_duration) / duration_seconds if duration_seconds > 0 else 0

        # 3. HIGH-PRECISION TRANSCRIPTION
        result = model.transcribe(
            temp_wav, fp16=False, language="en",
            beam_size=5, temperature=0,
            initial_prompt="Candidate name: Sudeena Jafer. Formal job interview."
        )
        full_text = result["text"].strip()
        
        # 4. ADVANCED METRICS
        clean_text = re.sub(r"[^a-z\s]", " ", full_text.lower())
        words_list = clean_text.split()
        wpm = (len(words_list) / (duration_seconds / 60)) if duration_seconds > 0 else 0
        filler_count = sum(1 for w in words_list if w in FILLERS)
        
        # Fluency: Combination of WPM speed, low fillers, and low silence ratio
        fluency = (min(wpm/150, 1.0) * 0.4) + ((1 - silence_ratio) * 0.4) + (max(1 - (filler_count/10), 0) * 0.2)
        
        # Communication: Linguistic clarity (based on length and word variety)
        communication = min(len(full_text) / 500, 1.0) if len(full_text) > 0 else 0

        if os.path.exists(temp_wav): os.remove(temp_wav)
            
        return {
        "transcript": full_text,
        "duration": format_time(duration_seconds),
        "wpm": round(wpm, 1),
        "fluency_score": round(fluency, 2),
        "communication_score": round(communication, 2) # Verified Key
    }

    except Exception as e:
        print(f"❌ Audio Error: {e}")
        return {"transcript": "Error", "duration": "00:00", "fluency_score": 0.5, "communication_score": 0.5, "wpm": 0}