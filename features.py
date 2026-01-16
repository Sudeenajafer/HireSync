import cv2
import numpy as np
import os
import PyPDF2
import time
import cloudinary
import cloudinary.uploader
from supabase import create_client
from audio import audio_phase_score
from src.ats_matcher import ATSMatcher
from dotenv import load_dotenv

# --- 1. ENVIRONMENT & CLOUD CONFIGURATION ---
load_dotenv()

# Explicit Cloudinary Configuration (Fixes "Must supply api_key" error)
cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
    secure = True
)

# Supabase Initialization
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None

# --- 2. AI ENGINE INITIALIZATION ---
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from hume import HumeClient
    from hume.models.config import FaceConfig
    HUME_AVAILABLE = True
except ImportError:
    HUME_AVAILABLE = False

MODEL_PATH = 'face_landmarker.task'
HUME_API_KEY = os.getenv("HUME_API_KEY")

# --- 3. CLOUD UTILITY FUNCTIONS ---

def upload_to_cloud(file_path, resource_type="auto"):
    """
    MSc Robustness: Uploads assets to Cloudinary with explicit error handling.
    resource_type should be 'video' for recordings and 'raw' for PDFs.
    """
    try:
        if not file_path or not os.path.exists(file_path):
            print(f"❌ File not found for upload: {file_path}")
            return None
            
        print(f"☁️ Cloudinary: Uploading {resource_type} from {os.path.basename(file_path)}...")
        
        response = cloudinary.uploader.upload(
            file_path, 
            resource_type=resource_type,
            chunk_size=6000000 # 6MB chunks for video stability
        )
        
        url = response.get('secure_url')
        print(f"✅ Cloudinary Success: {url}")
        return url
        
    except Exception as e:
        print(f"❌ CRITICAL CLOUDINARY ERROR: {e}")
        return None

def save_candidate_to_supabase(candidate_data):
    """
    Persists evaluation metadata and asset links to the PostgreSQL database.
    """
    if not supabase:
        print("⚠️ Supabase not configured. Skipping database save.")
        return
    try:
        print(f"💾 Supabase: Saving record for {candidate_data.get('name')}...")
        supabase.table("candidates").insert(candidate_data).execute()
        print("✅ Database Synchronized.")
    except Exception as e:
        print(f"❌ SUPABASE SAVE ERROR: {e}")

# --- 4. BEHAVIORAL ANALYSIS HELPERS ---

def get_grade(score):
    if score >= 0.80: return "🏆 A+ (Elite Match)"
    if score >= 0.70: return "✅ A (Strong)"
    if score >= 0.50: return "⚠️ B (Average)"
    return "❌ F (Rejected)"

def analyze_behavior_with_hume(video_path):
    """MSc Logic: Affective computing to determine stress and confidence."""
    if not HUME_AVAILABLE or not HUME_API_KEY:
        return {"confidence": 0.75, "anxiety": 0.25}
    try:
        client = HumeClient(api_key=HUME_API_KEY)
        configs = [FaceConfig(identify_faces=True)]
        job = client.submit_job([], configs, files=[video_path])
        job.await_complete()
        results = job.get_predictions()
        
        # Fixed hierarchy parsing
        predictions = results[0]['results']['predictions'][0]['models']['face']['grouped_predictions'][0]['predictions'][0]['emotions']
        emo_map = {e['name']: e['score'] for e in predictions}
        
        confidence = (emo_map.get("Calm", 0) + emo_map.get("Joy", 0)) / 2
        return {"confidence": round(confidence, 2), "anxiety": round(emo_map.get("Anxiety", 0), 2)}
    except Exception as e:
        print(f"⚠️ Hume API Error: {e}")
        return {"confidence": 0.60, "anxiety": 0.40}

def analyze_video_vision(video_path):
    """Efficient Temporal Sampling (1 FPS) using MediaPipe."""
    if not os.path.exists(MODEL_PATH): return 0.5 
    try:
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps) if fps > 0 else 0
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
        attention = (hits / samples)**2 if samples > 0 else 0.1
        return round(float(attention), 2)
    except: return 0.2

# --- 5. MAIN MULTIMODAL ENGINE ---

def extract_features(resume_path, jd_text, video_path, skip_ats=False):
    """
    Main Entry Point: Fuses Local & Cloud AI signals.
    """
    print(f"🚀 HireSync Engine: Analyzing Behavioral Biometrics...")
    
    # Run Local and Cloud Streams
    audio = audio_phase_score(video_path)
    attention = analyze_video_vision(video_path)
    hume_data = analyze_behavior_with_hume(video_path)

    # Multiplicative Scoring Logic (MSc Standard)
    f = max(0.01, audio.get('fluency_score', 0.5))
    c = max(0.01, audio.get('communication_score', 0.5))
    a = max(0.01, attention)
    h = max(0.01, hume_data['confidence'])
    
    # Calculate Anxiety Penalty
    anx_p = max(0, hume_data['anxiety'] - 0.4)
    behavior_score = round(((f * c * a * h) ** (1/4)) - anx_p, 2)
    behavior_score = max(0.05, behavior_score)

    res_data = {
        "behavior_score": behavior_score,
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

    # Resume Stage (Phase 1)
    suitability = 0.0
    strengths = []
    if resume_path and jd_text:
        matcher = ATSMatcher()
        try:
            reader = PyPDF2.PdfReader(resume_path)
            resume_text = " ".join([p.extract_text() for p in reader.pages])
            ats_results = matcher.analyze_resume_llm(resume_text, jd_text)
            suitability = ats_results['final_score'] / 100
            strengths = ats_results['matched_skills']
        except Exception as e:
            print(f"⚠️ ATS processing error: {e}")
            suitability = 0.5

    # Final Overall Score (Weighted 60/40)
    final_score = round((suitability * 0.6) + (behavior_score * 0.4), 2)
    res_data.update({
        "suitability": suitability, 
        "final_score": final_score, 
        "strengths": strengths
    })
    
    return res_data

print("✅ features.py: High-Stability Cloud Engine LOADED.")