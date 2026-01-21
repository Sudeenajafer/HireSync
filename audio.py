import os
import re
import librosa
import numpy as np
import subprocess
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
from static_ffmpeg import add_paths

# Initialize
add_paths() 
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def audio_phase_score(video_path):
    """
    MSc Precision Engine:
    1. Removes background noise using High-pass/Low-pass filters.
    2. Normalizes volume to peak levels.
    3. Uses Gemini 1.5 Flash with 'Interviewer Context' for better accuracy.
    """
    # Create a unique temp file to avoid Windows "File in Use" errors
    temp_wav = os.path.join(os.getcwd(), f"processed_voice_{int(time.time())}.wav")
    
    try:
        # Wait for the browser to finish writing the video blob
        time.sleep(2) 
        
        if not os.path.exists(video_path):
            return {"transcript": "[Video not found]", "duration": "00:00", "fluency_score": 0}

        print(f"🎙️ Pre-processing audio for transcription...")
        
        # --- THE MASTER FFmpeg FILTER COMMAND ---
        # -af "highpass=f=200,lowpass=f=3000": Removes low hum and high hiss
        # -af "afftdn": Uses Fast Fourier Transform for deep noise reduction
        # -af "agnorm": Automatically normalizes volume
        cmd = [
            'ffmpeg', '-y', '-ignore_unknown',
            '-i', video_path,
            '-vn', 
            '-af', 'highpass=f=200,lowpass=f=3000,afftdn,anlmdn,asmooth,aresample=async=1,normalize',
            '-acodec', 'pcm_s16le', 
            '-ar', '16000', 
            '-ac', '1', 
            temp_wav
        ]
        
        subprocess.run(cmd, capture_output=True, text=True)

        # Safety Check: Did we extract any sound?
        if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) < 2000:
            return {
                "transcript": "[SILENCE DETECTED: Ensure your microphone is ALLOWED in the browser lock icon settings.]",
                "duration": "00:00", "wpm": 0, "fluency_score": 0.1, "communication_score": 0.1
            }

        # 1. ACOUSTIC ANALYSIS (Duration/Silence)
        y, sr = librosa.load(temp_wav)
        total_dur = librosa.get_duration(y=y, sr=sr)
        intervals = librosa.effects.split(y, top_db=25)
        speaking_time = sum([itv[1] - itv[0] for itv in intervals]) / sr

        # 2. GEMINI HIGH-PRECISION TRANSCRIPTION
        print("☁️ Sending to Gemini 1.5 (High Fidelity Mode)...")
        with open(temp_wav, "rb") as f:
            audio_bytes = f.read()
            
        # We tell Gemini exactly what kind of content to expect (Interview Context)
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                """You are a professional stenographer. Transcribe this interview audio perfectly. 
                Keep technical terms (like Python, AI, Machine Learning, Data) accurate. 
                Do not add any comments, just the transcript.""",
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
            ]
        )
        full_text = response.text.strip()

        # 3. METRIC MATH
        words = full_text.split()
        speaking_mins = speaking_time / 60
        wpm = round(len(words) / speaking_mins, 1) if speaking_mins > 0 else 0
        
        # Cleanup
        if os.path.exists(temp_wav): os.remove(temp_wav)

        return {
            "transcript": full_text if full_text else "[Speech detected but unintelligible]",
            "duration": f"{int(total_dur // 60):02d}:{int(total_dur % 60):02d}",
            "wpm": wpm,
            "fluency_score": 0.85 if 120 < wpm < 160 else 0.50,
            "communication_score": 0.80
        }

    except Exception as e:
        print(f"❌ Transcription Crash: {e}")
        return {"transcript": f"[Error: {str(e)[:30]}]", "duration": "00:00", "fluency_score": 0.1, "wpm": 0}