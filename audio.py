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

add_paths() 
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

TEMP_DIR = os.path.join(os.getcwd(), "temp_audio_cache")
os.makedirs(TEMP_DIR, exist_ok=True)

def audio_phase_score(video_path):
    file_id = int(time.time())
    temp_wav = os.path.join(TEMP_DIR, f"extract_{file_id}.wav")
    
    try:
        # 1. Wait for file to be ready (Crucial for Windows)
        time.sleep(2) 
        
        if not os.path.exists(video_path):
            return {"transcript": "[Error: File not found on server]", "duration": "00:00", "fluency_score": 0}

        print(f"🎤 Extracting audio from uploaded file: {os.path.basename(video_path)}")
        
        # 2. UNIVERSAL EXTRACTION COMMAND
        # We removed -map 0:a? and used a simpler approach that works for 99% of MP4/WebM files
        cmd = [
            'ffmpeg', '-y', 
            '-i', video_path,
            '-vn',                    # Disable video
            '-ac', '1',               # Force Mono
            '-ar', '16000',           # 16kHz
            '-acodec', 'pcm_s16le',   # Standard WAV codec
            temp_wav
        ]
        
        # Run FFmpeg and capture logs for debugging
        result = subprocess.run(cmd, capture_output=True, text=True)

        # 3. CONTENT VALIDATION
        if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) < 1000:
            print(f"❌ FFmpeg Log: {result.stderr}")
            return {
                "transcript": "[AUDIO NOT FOUND: The uploaded video has no sound track, or the format is incompatible. Please try an MP4 with AAC/MP3 audio.]",
                "duration": "00:00", "wpm": 0, "fluency_score": 0.1, "communication_score": 0.1
            }

        # 4. TRANSCRIPTION (Gemini 1.5)
        print("☁️ Sending to Gemini for Cloud Transcription...")
        with open(temp_wav, "rb") as f:
            audio_data = f.read()
            
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                "Provide a verbatim transcript of this technical interview response.",
                types.Part.from_bytes(data=audio_data, mime_type="audio/wav")
            ]
        )
        full_text = response.text.strip()
        
        # 5. LINGUISTIC ANALYSIS
        y, sr = librosa.load(temp_wav)
        total_dur = librosa.get_duration(y=y, sr=sr)
        
        words = full_text.split()
        # WPM Calculation: (Total Words / Duration in Minutes)
        wpm = round(len(words) / (total_dur / 60), 1) if total_dur > 0 else 0

        print(f"✅ Success: {len(words)} words transcribed.")

        return {
            "transcript": full_text if full_text else "[Silence detected in file]",
            "duration": f"{int(total_dur // 60):02d}:{int(total_dur % 60):02d}",
            "wpm": wpm,
            "fluency_score": 0.85 if 120 < wpm < 170 else 0.60,
            "communication_score": 0.80
        }

    except Exception as e:
        print(f"❌ Audio Processing Error: {e}")
        return {"transcript": f"Error: {str(e)}", "duration": "00:00", "fluency_score": 0.1, "wpm": 0}

    finally:
        # Always clean up the temp WAV file
        if os.path.exists(temp_wav):
            try: os.remove(temp_wav)
            except: pass