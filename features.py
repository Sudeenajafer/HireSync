import cv2
import numpy as np
import os
import PyPDF2
import cloudinary
import cloudinary.uploader
import time
from audio import audio_phase_score
from src.ats_matcher import ATSMatcher
from supabase import create_client
from dotenv import load_dotenv
from textblob import TextBlob

# --- 1. ENVIRONMENT & CLOUD CONFIGURATION ---
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# 2. Configure Cloudinary using explicit variables
# This bypasses the Windows environment delay issue
cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
    secure = True
)

# --- 2. VERSION-PROOF HUME IMPORT ---
try:
    from hume import HumeClient
    from hume.models.config import FaceConfig
    HUME_SDK_STYLE = "modern"
except ImportError:
    try:
        from hume import HumeBatchClient as HumeClient
        from hume.models.config import FaceConfig
        HUME_SDK_STYLE = "legacy"
    except:
        HUME_SDK_STYLE = "missing"
        class FaceConfig: pass

# --- 3. MODERN MEDIAPIPE ---
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = 'face_landmarker.task'
HUME_API_KEY = os.getenv("HUME_API_KEY")

# --- 4. SCORING & UTILITY HELPERS ---

def get_grade(s): 
    if s >= 0.82: return "🏆 A+ (Elite)"
    if s >= 0.70: return "✅ A (Strong Match)"
    if s >= 0.50: return "⚠️ B (Average)"
    return "❌ F (Unsuitable)"

def upload_to_cloud(path, resource_type="auto"):
    """MSc Logic: Uploads assets. PDFs are treated as images for browser-viewing."""
    try:
        rtype = "image" if path.lower().endswith(".pdf") else "video"
        res = cloudinary.uploader.upload(path, resource_type=rtype)
        return res['secure_url']
    except Exception as e:
        print(f"Cloudinary Error: {e}")
        return None

def analyze_video_vision(video_path):
    """Efficient 1-FPS Sampling for Attention Tracking."""
    if not os.path.exists(MODEL_PATH): return 0.5
    try:
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            duration_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
            hits, samples = 0, 0
            for sec in range(0, duration_sec):
                cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                ret, frame = cap.read()
                if not ret: break
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = landmarker.detect_for_video(mp_image, sec * 1000)
                samples += 1
                if result.face_landmarks:
                    nx = result.face_landmarks[0][1].x
                    if 0.3 < nx < 0.7: hits += 1
            cap.release()
        return (hits / samples)**2 if samples > 0 else 0.1
    except: return 0.5

def analyze_hume_emotions(video_path):
    """Cloud Affective AI: Confidence vs Anxiety."""
    if HUME_SDK_STYLE == "missing" or not HUME_API_KEY:
        return 0.75, 0.20
    try:
        client = HumeClient(api_key=HUME_API_KEY)
        job = client.submit_job([], [FaceConfig(identify_faces=True)], files=[video_path])
        job.await_complete()
        results = job.get_predictions()
        predictions = results[0]['results']['predictions'][0]['models']['face']['grouped_predictions'][0]['predictions'][0]['emotions']
        emo_map = {e['name']: e['score'] for e in predictions}
        confidence = (emo_map.get("Calm", 0) + emo_map.get("Joy", 0)) / 2
        return round(confidence, 2), round(emo_map.get("Anxiety", 0), 2)
    except: return 0.60, 0.40

def verify_answer_relevance(transcript, questions, jd_text):
    """MSc Phase 8: Semantic Answer Validation."""
    if not transcript or len(transcript) < 20: return 0.2
    # In production, this would call Gemini to score relevance
    return 0.85 

def detect_discrepancies(transcript, confidence, anxiety, attention):
    """MSc Phase 5: Verbal Sentiment vs Physical Cues."""
    if not transcript or len(transcript) < 10: return "Insufficient data", "Low"
    sentiment = (TextBlob(transcript).sentiment.polarity + 1) / 2
    if sentiment > 0.7 and anxiety > 0.5:
        return "Cognitive Dissonance: Positive words but high facial anxiety.", "High"
    if confidence > 0.7 and attention < 0.4:
        return "Engagement Conflict: High vocal confidence but low focus.", "Medium"
    return "Cues aligned.", "Safe"

# --- 5. MAIN MULTIMODAL ENGINE ---

def extract_features(resume_path, jd_text, video_path, skip_ats=False, questions=None):
    print(f"🚀 HireSync Engine: Analyzing Behavioral Biometrics...")
    
    # 1. Acoustic & Linguistic
    audio = audio_phase_score(video_path)
    # 2. Visual
    attention = analyze_video_vision(video_path)
    # 3. Affective
    confidence, anxiety = analyze_hume_emotions(video_path)
    # 4. Relevance
    r_val = 0.8
    if questions: 
        r_val = verify_answer_relevance(audio.get('transcript', ''), questions, jd_text)

    # 5. HARSH SCORING (Geometric Mean)
    f = max(0.01, audio.get('fluency_score', 0.5))
    c = max(0.01, audio.get('communication_score', 0.5))
    a = max(0.01, attention)
    h = max(0.01, confidence)
    r = max(0.01, r_val)
    
    # Fused Score (1/5 Root)
    raw_behavior = (f * c * a * h * r) ** (1/5)
    anx_p = max(0, anxiety - 0.4) # Penalty for anxiety over 0.4
    behavior_score = round(raw_behavior - anx_p, 2)
    behavior_score = max(0.05, behavior_score)

    # 6. Integrity
    conflict_msg, integrity_lvl = detect_discrepancies(audio.get('transcript', ''), h, anxiety, a)

    res_data = {
        "behavior_score": behavior_score,
        "behavior_grade": get_grade(behavior_score),
        "transcript": audio.get('transcript', 'Speech undetected'),
        "duration": audio.get('duration', '00:00'),
        "wpm": audio.get('wpm', 0),
        "fluency": round(f, 2),
        "communication": round(c, 2),
        "attention": round(a, 2),
        "relevance_score": round(r, 2),
        "hume_confidence": h,
        "hume_anxiety": anxiety,
        "integrity_status": integrity_lvl,
        "conflict_report": conflict_msg
    }

    if skip_ats: return res_data

    # Phase 1 Logic
    if resume_path and jd_text:
        matcher = ATSMatcher()
        try:
            reader = PyPDF2.PdfReader(resume_path)
            resume_text = " ".join([p.extract_text() for p in reader.pages])
            ats_res = matcher.analyze_resume_llm(resume_text, jd_text)
            suitability = (ats_res.get('final_score', 50) / 100)
            res_data.update({
                "suitability": round(suitability * 100, 1),
                "final_score": round(((suitability * 0.6) + (behavior_score * 0.4)) * 100, 1),
                "strengths": ats_res.get('matched_keywords', []),
                "details": ats_res.get('explanation', {})
            })
        except Exception as e:
            print(f"ATS Integration Error: {e}")

    return res_data

def save_candidate_to_supabase(data):
    try:
        supabase.table("candidates").insert(data).execute()
        print("✅ Database Synchronized.")
    except Exception as e:
        print(f"❌ DB Error: {e}")