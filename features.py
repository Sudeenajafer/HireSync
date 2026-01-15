import cv2
import numpy as np
import os
import PyPDF2
import time
from audio import audio_phase_score
from src.ats_matcher import ATSMatcher
from dotenv import load_dotenv # New Import
load_dotenv()

# --- MODERN MEDIAPIPE ---
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- SAFE HUME AI INTEGRATION ---
HUME_AVAILABLE = False
try:
    from hume import HumeClient
    from hume.models.config import FaceConfig
    HUME_AVAILABLE = True
    print("✅ Hume AI SDK detected.")
except ImportError:
    print("⚠️ Hume AI SDK missing. Running in Local-Only mode.")

# Ensure 'face_landmarker.task' is in your project folder
MODEL_PATH = 'face_landmarker.task'

# --- CONFIGURATION ---
HUME_API_KEY = os.getenv("HUME_API_KEY") 

def get_grade(score):
    if score >= 0.82: return "🏆 A+ (Exceptional)"
    if score >= 0.72: return "✅ A (Strong Match)"
    if score >= 0.55: return "⚠️ B (Average)"
    return "❌ C (Unsuitable)"

def analyze_behavior_with_hume(video_path):
    """MSc Logic: Affective Computing via Hume AI"""
    # 1. Fallback if SDK is missing or Key is default
    if not HUME_AVAILABLE or HUME_API_KEY == os.getenv("HUME_API_KEY"):
        return {"confidence": 0.75, "anxiety": 0.25}

    try:
        client = HumeClient(api_key=HUME_API_KEY)
        configs = [FaceConfig(identify_faces=True)]
        
        print("☁️ Requesting Hume AI Cloud Analysis...")
        job = client.submit_job([], configs, files=[video_path])
        job.await_complete()
        results = job.get_predictions()

        # Access the emotion predictions
        predictions = results[0]['results']['predictions'][0]['models']['face']['grouped_predictions'][0]['predictions'][0]['emotions']
        emo_map = {e['name']: e['score'] for e in predictions}
        
        anxiety = emo_map.get("Anxiety", 0)
        calm = emo_map.get("Calm", 0)
        joy = emo_map.get("Joy", 0)
        
        return {"confidence": round((calm + joy)/2, 2), "anxiety": round(anxiety, 2)}
    except Exception as e:
        print(f"⚠️ Hume API Error: {e}")
        return {"confidence": 0.70, "anxiety": 0.30}

def analyze_video_vision(video_path):
    """Efficient 1-FPS Visual Sampling"""
    if not os.path.exists(MODEL_PATH):
        return 0.80, 0.80 
    try:
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = int(total_frames / fps) if fps > 0 else 0
            hits, samples = 0, 0
            for sec in range(0, duration_sec):
                cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                ret, frame = cap.read()
                if not ret: break
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = landmarker.detect_for_video(mp_image, sec * 1000)
                samples += 1
                if result.face_landmarks:
                    nose_x = result.face_landmarks[0][1].x
                    if 0.35 < nose_x < 0.65: hits += 1
            cap.release()
        return (hits / samples)**2 if samples > 0 else 0.5, 0.85
    except: return 0.5, 0.5

def extract_features(resume_pdf_path, jd_path, video_path, skip_ats=False):
    print(f"🚀 HireSync AI: Starting Multimodal Engine...")
    
    # 1. Local Analysis
    audio = audio_phase_score(video_path)
    attention, gaze = analyze_video_vision(video_path)

    # 2. Hume Cloud Analysis
    hume_data = analyze_behavior_with_hume(video_path)

    # 3. Geometric Mean Scoring (MSc Stricter Logic)
    f = max(0.1, audio.get('fluency_score', 0.5))
    c = max(0.1, audio.get('communication_score', 0.5))
    a = max(0.1, attention)
    h_conf = max(0.1, hume_data['confidence'])
    
    # Calculate penalty based on cloud anxiety
    anxiety_penalty = max(0, hume_data['anxiety'] - 0.4)
    raw_behavior = (f * c * a * h_conf) ** (1/4)
    behavior_score = round(raw_behavior - anxiety_penalty, 2)
    
    res_data = {
        "behavior_score": max(0, behavior_score),
        "behavior_grade": get_grade(behavior_score),
        "fluency": round(f, 2),
        "communication": round(c, 2),
        "attention": round(a, 2),
        "hume_confidence": hume_data['confidence'],
        "hume_anxiety": hume_data['anxiety'],
        "duration": audio.get('duration', '00:00'),
        "transcript": audio.get('transcript', 'No speech detected'),
        "wpm": audio.get('wpm', 0)
    }
    if skip_ats: return res_data

    # Phase 1 Logic
    suitability = 0.0
    strengths = []
    if resume_pdf_path and jd_path:
        matcher = ATSMatcher()
        try:
            reader = PyPDF2.PdfReader(resume_pdf_path)
            text = " ".join([p.extract_text() for p in reader.pages])
            suitability, strengths = matcher.analyze_resume(text, jd_path)
        except: suitability = 0.5

    final_score = round((suitability * 0.6) + (behavior_score * 0.4), 2)
    res_data.update({"suitability": suitability, "final_score": final_score, "strengths": strengths})
    return res_data

print("✅ features.py: High-Stability Engine Loaded.")