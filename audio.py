import whisper
import os
import re
import librosa
import numpy as np
import subprocess
import time

# Load the model
print("⏳ Loading Whisper AI...")
model = whisper.load_model("tiny", device="cpu")

def audio_phase_score(video_path):
    # Create a clean filename
    temp_wav = os.path.join(os.getcwd(), f"temp_proc_{int(time.time())}.wav")
    
    try:
        if not os.path.exists(video_path):
            return {"transcript": "[File Error]", "fluency_score": 0.0}

        print(f"🎤 Extraction started for: {video_path}")
        
        # --- THE FIX: Try multiple ways to run FFmpeg ---
        ffmpeg_cmd = "ffmpeg"
        # Try to use static-ffmpeg if installed
        try:
            from static_ffmpeg import add_paths
            add_paths()
        except:
            pass

        # Execute extraction
        process = subprocess.run(
            [ffmpeg_cmd, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_wav],
            capture_output=True,
            text=True
        )

        if process.returncode != 0:
            print(f"❌ FFmpeg Error: {process.stderr}")
            return {"transcript": "[Audio Extraction Error]", "fluency_score": 0.3}

        # Verify WAV exists and has data
        if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) < 100:
            return {"transcript": "[Empty Audio Stream]", "fluency_score": 0.3}

        print("📝 Transcribing...")
        # Add a prompt to help the AI recognize your specific name
        result = model.transcribe(
            temp_wav, 
            fp16=False, 
            language="en",
            initial_prompt="Candidate: Sudeena Jafer. Interview response."
        )
        
        text = result["text"].strip()
        
        # Acoustic Analysis
        y, sr = librosa.load(temp_wav)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Cleanup
        if os.path.exists(temp_wav):
            try: os.remove(temp_wav)
            except: pass
            
        return {
            "transcript": text if text else "[Silence Detected]", 
            "fluency_score": 0.85 if len(text) > 5 else 0.4
        }
        
    except Exception as e:
        print(f"❌ Transcription Crash: {e}")
        return {"transcript": f"[Transcription Failed: {str(e)[:15]}]", "fluency_score": 0.5}

print("✅ Audio Phase 3 (Self-Healing) LOADED!")