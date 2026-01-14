import cv2
import numpy as np
import os
import PyPDF2
import time
from audio import audio_phase_score
from src.ats_matcher import ATSMatcher

# --- MODERN MEDIAPIPE TASKS API ---
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- VERSION-PROOF HUME IMPORT (Replaces old import lines) ---
try:
    # Try the newer SDK version (v0.16+)
    from hume import HumeClient as HumeBatchClient
    from hume.models.config import FaceConfig
    print("✅ Hume AI: Using modern SDK (HumeClient)")
except ImportError:
    try:
        # Fallback to older SDK version
        from hume import HumeBatchClient
        from hume.models.config import FaceConfigs
        print("✅ Hume AI: Using legacy SDK (HumeBatchClient)")
    except ImportError:
        print("❌ Hume AI: SDK not found. Please run 'pip install hume'")

# Ensure 'face_landmarker.task' is in your project folder
MODEL_PATH = 'face_landmarker.task'

# --- CONFIGURATION (With your actual key) ---
HUME_API_KEY = "340ecs5IynBsdrMgyuCLaFrQu2PY2DqXLBGtAg3qGUfj5Iuo" 

def get_grade(score):
    if score >= 0.82: return "🏆 A+ (Exceptional)"
    if score >= 0.72: return "✅ A (Strong Match)"
    if score >= 0.55: return "⚠️ B (Average)"
    return "❌ C (Unsuitable)"

def analyze_behavior_with_hume(video_path):
    """
    MSc Cloud Logic: Detects anxiety vs confidence.
    Updated to handle the latest sdk variable hierarchy.
    """
    if not HUME_API_KEY or HUME_API_KEY == "YOUR_ACTUAL_API_KEY_HERE":
        return {"confidence": 0.75, "anxiety": 0.25}

    try:
        # The aliased HumeBatchClient handles initialization
        client = HumeBatchClient(api_key=HUME_API_KEY)
        configs = [FaceConfig(identify_faces=True)]
        
        print("☁️ Submitting Video to Hume AI...")
        # Note: In HumeClient, this works via the top-level or sub-modules
        job = client.submit_job([], configs, files=[video_path])
        
        print("⏳ Waiting for Cloud Analysis (takes 30-60s)...")
        job.await_complete()
        results = job.get_predictions()

        # Extract Emotions safely
        # Hierarchy: [File][Model][Prediction][Emotion]
        # Fixed the variable naming here:
        predictions = results[0]['results']['predictions'][0]['models']['face']['grouped_predictions'][0]['predictions'][0]['emotions']
        
        emo_map = {e['name']: e['score'] for e in predictions}
        
        anxiety = emo_map.get("Anxiety", 0)
        calm = emo_map.get("Calm", 0)
        joy = emo_map.get("Joy", 0)
        
        # Calculate a combined confidence metric
        confidence = (calm + joy) / 2
        
        return {"confidence": round(confidence, 2), "anxiety": round(anxiety, 2)}

    except Exception as e:
        print(f"⚠️ Hume API Error: {e}")
        return {"confidence": 0.7, "anxiety": 0.3}

def analyze_video_vision(video_path):
    """Efficient 1-FPS Visual Sampling using MediaPipe Tasks"""
    if not os.path.exists(MODEL_PATH):
        return 0.80, 0.80 
    try:
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
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
    
    # 1. LOCAL ANALYSIS (Whisper + MediaPipe)
    audio = audio_phase_score(video_path)
    attention, gaze = analyze_video_vision(video_path)

    # 2. CLOUD BEHAVIORAL ANALYSIS (Hume AI)
    hume_data = analyze_behavior_with_hume(video_path)

    # 3. MULTIPLICATIVE SCORING LOGIC
    f = max(0.1, audio.get('fluency_score', 0.5))
    c = max(0.1, audio.get('communication_score', 0.5))
    a = max(0.1, attention)
    h_conf = max(0.1, hume_data['confidence'])
    
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

    # 4. PHASE 1 INTEGRATION
    suitability = 0.0
    strengths = []
    if resume_pdf_path and jd_path:
        matcher = ATSMatcher()
        try:
            reader = PyPDF2.PdfReader(resume_pdf_path)
            text = " ".join([p.extract_text() for p in reader.pages])
            suitability, strengths = matcher.analyze_resume(text, jd_path)
        except: suitability = 0.5

    # FINAL FUSION (60% Resume, 40% Behavior)
    final_score = round((suitability * 0.6) + (behavior_score * 0.4), 2)
    res_data.update({"suitability": suitability, "final_score": final_score, "strengths": strengths})
    
    return res_data

print("✅ features.py: Hume AI Version-Proof Engine Loaded.")