import cv2
import numpy as np
import os
import PyPDF2
import cloudinary.uploader
import time
import json
from audio import audio_phase_score
from src.ats_matcher import ATSMatcher
from supabase import create_client
from dotenv import load_dotenv
from textblob import TextBlob

# --- 1. ENVIRONMENT & CLOUD CONFIGURATION ---
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Explicit Cloudinary Configuration
import cloudinary
cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
    secure = True
)

# --- 2. VERSION-PROOF HUME IMPORT ---
try:
    from hume import HumeClient
    from hume.models.config import FaceConfig # type: ignore
    HUME_SDK_STYLE = "modern"
except ImportError:
    try:
        from hume import HumeBatchClient as HumeClient # type: ignore
        from hume.models.config import FaceConfig # type: ignore
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

def analyze_video_vision(video_path):
    """Sequential Frame Scanning for real Attention metrics."""
    if not os.path.exists(MODEL_PATH): return 0.5
    try:
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(video_path)
            hits, total_samples = 0, 0
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or total_samples >= 60: break
                if frame_count % 10 == 0:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    result = landmarker.detect_for_video(mp_image, frame_count * 33)
                    total_samples += 1
                    if result.face_landmarks:
                        nx = result.face_landmarks[0][1].x
                        if 0.35 < nx < 0.65: hits += 1
                frame_count += 1
            cap.release()
        return (hits / total_samples)**2 if total_samples > 0 else 0.3
    except: return 0.4

def analyze_behavior_with_hume(video_path):
    """Cloud Affective AI: Confidence vs Anxiety."""
    if HUME_SDK_STYLE == "missing" or not HUME_API_KEY:
        return {"confidence": 0.75, "anxiety": 0.20}
    try:
        client = HumeClient(api_key=HUME_API_KEY)
        job = client.submit_job([], [FaceConfig(identify_faces=True)], files=[video_path])
        job.await_complete()
        results = job.get_predictions()
        predictions = results[0]['results']['predictions'][0]['models']['face']['grouped_predictions'][0]['predictions'][0]['emotions']
        emo_map = {e['name']: e['score'] for e in predictions}
        confidence = (emo_map.get("Calm", 0) + emo_map.get("Joy", 0)) / 2
        return {"confidence": round(confidence, 2), "anxiety": round(emo_map.get("Anxiety", 0), 2)}
    except: return {"confidence": 0.6, "anxiety": 0.4}

def verify_technical_knowledge(transcript, questions, jd_text):
    """
    Core Innovation: Uses Gemini to verify if the candidate's answers 
    are technically accurate for the specific field mentioned in the JD.
    """
    if len(transcript.split()) < 10: return 0.1
    
    prompt = f"""
    Compare the Candidate's Answer against the Interview Questions and Job Description.
    QUESTIONS: {questions}
    TRANSCRIPT: {transcript}
    JOB DESCRIPTION: {jd_text[:1000]}

    SCORE CRITERIA (0.0 to 1.0):
    - 1.0: Precise technical knowledge of the field.
    - 0.5: Generic answers with no specific field expertise.
    - 0.1: Wrong field or irrelevant content (e.g., talking about ML for a BA role).
    
    Return ONLY the numeric float.
    """
    try:
        from src.ats_matcher import ATSMatcher
        m = ATSMatcher()
        response = m.client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
        return float(response.text.strip())
    except: return 0.2

def detect_discrepancies(transcript, confidence, anxiety, attention):
    if not transcript or len(transcript) < 10: return "Insufficient data", "Low"
    sentiment = (TextBlob(transcript).sentiment.polarity + 1) / 2
    if sentiment > 0.75 and anxiety > 0.5:
        return "Cognitive Dissonance: Verbal sentiment vs facial anxiety.", "High"
    return "Cues aligned.", "Safe"

# --- 5. MAIN MULTIMODAL ENGINE ---

def extract_features(resume_path, jd_text, video_path, skip_ats=False, questions=None):
    print(f"🚀 HireSync Engine: Analyzing Behavioral Biometrics & Knowledge...")
    
    # 1. Processing Streams
    audio = audio_phase_score(video_path)
    attention = analyze_video_vision(video_path)
    hume_results = analyze_behavior_with_hume(video_path)
    
    confidence = hume_results.get('confidence', 0.5)
    anxiety = hume_results.get('anxiety', 0.5)

    # 2. Knowledge Analysis (The "Strict" Check)
    knowledge_score = 0.8 # Default baseline
    if questions:
        knowledge_score = verify_technical_knowledge(audio['transcript'], questions, jd_text)
        print(f"🎯 Field Knowledge Score: {knowledge_score}")

    # 3. Multimodal Fusion (Geometric Mean)
    # f=Fluency, c=Communication, a=Attention, h=Confidence, k=Knowledge
    f = max(0.01, audio.get('fluency_score', 0.5))
    c = max(0.01, audio.get('communication_score', 0.5))
    a = max(0.01, attention)
    h = max(0.01, confidence)
    k = max(0.01, knowledge_score)
    
    # Fusing all 5 modes (1/5 root)
    behavior_score = round((f * c * a * h * k) ** (1/5), 2)
    
    # 4. DISQUALIFICATION GATE
    # If knowledge or attention is very low, force a failing grade
    if k < 0.35 or a < 0.3:
        behavior_grade = "❌ F (Rejected - Lack of Focus/Knowledge)"
        behavior_score = min(behavior_score, 0.25) # Force crash score
    else:
        behavior_grade = get_grade(behavior_score)

    conflict_msg, integrity_lvl = detect_discrepancies(audio['transcript'], h, anxiety, a)

    res_data = {
        "behavior_score": behavior_score,
        "behavior_grade": behavior_grade,
        "relevance_score": knowledge_score,
        "hume_confidence": confidence,
        "hume_anxiety": anxiety,
        "attention": attention,
        "transcript": audio['transcript'],
        "duration": audio['duration'],
        "integrity_status": integrity_lvl,
        "conflict_report": conflict_msg,
        "wpm": audio.get('wpm', 0)
    }

    if skip_ats: return res_data

    # 5. Document AI (Phase 1)
    if resume_path and jd_text:
        from src.ats_matcher import ATSMatcher
        matcher = ATSMatcher()
        try:
            reader = PyPDF2.PdfReader(resume_path)
            resume_text = " ".join([p.extract_text() for p in reader.pages])
            ats_res = matcher.analyze_resume_llm(resume_text, jd_text)
            suitability = (ats_res.get('final_score', 40) / 100)
            
            # Final 100-point calculation
            final_overall = round(((suitability * 0.6) + (behavior_score * 0.4)) * 100, 1)

            res_data.update({
                "suitability": round(suitability * 100, 1),
                "final_score": final_overall,
                "strengths": ats_res.get('matched_keywords', []),
                "details": ats_res.get('explanation', {}),
                "final_reasoning_text": ats_res.get('explanation', {}).get('verdict', "Analysis complete.")
            })
        except: pass

    return res_data

def save_candidate_to_supabase(data):
    try:
        supabase.table("candidates").insert(data).execute()
        print("✅ Database Synchronized.")
    except Exception as e: print(f"❌ DB Error: {e}")

def upload_to_cloud(path, resource_type="auto"):
    try:
        rtype = "image" if path.lower().endswith(".pdf") else "video"
        res = cloudinary.uploader.upload(path, resource_type=rtype)
        return res['secure_url']
    except: return None