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

        print(f"🎤 Extracting audio from: {os.path.basename(video_path)}")
        
        # 2. UNIVERSAL EXTRACTION COMMAND
        cmd = [
            'ffmpeg', '-y', 
            '-err_detect', 'ignore_err', 
            '-ignore_unknown',
            '-i', video_path,
            '-vn', '-map', '0:a:0', 
            '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 
            temp_wav
        ]

        
        subprocess.run(cmd, capture_output=True, text=True)

        # 3. CONTENT VALIDATION
        if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) < 1000:
            return {
                "transcript": "[AUDIO NOT FOUND: Ensure your microphone is active.]",
                "duration": "00:00", "wpm": 0, "fluency_score": 0.1, "communication_score": 0.1
            }

        # 4. TRANSCRIPTION WITH 404 & 503 PROTECTION
        print("☁️ Sending to Gemini for Cloud Transcription...")
        with open(temp_wav, "rb") as f:
            audio_data = f.read()
            
        full_text = ""
        # The list of models to try in case one 404s
        model_options = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
        
        for attempt in range(3):
            try:
                # Use the first model in the list
                current_model = model_options[0] if attempt < 2 else model_options[1]
                
                response = client.models.generate_content(
                    model=current_model,
                    contents=[
                        types.Part.from_bytes(data=audio_data, mime_type="audio/wav"),
                        "Provide a verbatim transcript of this audio. Technical accuracy is required."
                    ]
                )
                full_text = response.text.strip()
                if full_text:
                    break 
            except Exception as e:
                err_str = str(e).lower()
                if "503" in err_str or "overloaded" in err_str:
                    print(f"⚠️ Gemini Busy (Attempt {attempt+1}/3). Retrying...")
                    time.sleep(4)
                elif "404" in err_str or "not found" in err_str:
                    print(f"🔄 Model string 404. Trying alternative alias...")
                    # Switch to the 'latest' alias if the standard name 404s
                    model_options.reverse() 
                else:
                    print(f"❌ Gemini Error: {e}")
                    full_text = "[Transcription Error]"
                    break

        # If after 3 attempts we still have no text
        if not full_text:
            full_text = "[Transcription Service Temporarily Unavailable]"
            
        # 5. LINGUISTIC ANALYSIS
        y, sr = librosa.load(temp_wav)
        total_dur = librosa.get_duration(y=y, sr=sr)
        
        words = full_text.split()
        # WPM Calculation
        wpm = round(len(words) / (total_dur / 60), 1) if total_dur > 0 else 0

        print(f"✅ Transcription Complete: {len(words)} words detected.")

        return {
            "transcript": full_text if full_text else "[Silence detected]",
            "duration": f"{int(total_dur // 60):02d}:{int(total_dur % 60):02d}",
            "wpm": wpm,
            "fluency_score": 0.85 if 120 < wpm < 170 else 0.60,
            "communication_score": 0.80
        }

    except Exception as e:
        print(f"❌ Audio Processing Error: {e}")
        return {"transcript": f"Error: {str(e)}", "duration": "00:00", "fluency_score": 0.1, "wpm": 0}

    finally:
        # Cleanup
        if os.path.exists(temp_wav):
            try: os.remove(temp_wav)
            except: pass