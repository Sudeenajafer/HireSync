import cv2
import numpy as np
import os
import PyPDF2
import time
import cloudinary
import json
import cloudinary.uploader
from supabase import create_client
from audio import audio_phase_score
from src.ats_matcher import ATSMatcher
from dotenv import load_dotenv
from textblob import TextBlob 

# --- 1. ENVIRONMENT & CLOUD CONFIGURATION ---
load_dotenv()

cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
    secure = True
)

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

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

# --- 3. PHASE 8: ANSWER RELEVANCE HELPER ---
def verify_answer_relevance(transcript, questions, jd_text):
    """
    MSc Phase 8 Logic: Semantic Answer Validation.
    Uses Gemini to check if the candidate actually answered the generated questions.
    """
    if not transcript or len(transcript.split()) < 5: 
        return 0.1 # Automatic low score for no speech

    prompt = f"""
    You are a Technical Interview Evaluator.
    INTERVIEW QUESTIONS ASKED: {questions}
    CANDIDATE RESPONSE TRANSCRIPT: {transcript}
    JOB REQUIREMENTS: {jd_text[:1000]}

    TASK:
    Rate the 'Technical Relevance' of the candidate's answer from 0.0 to 1.0.
    - 1.0: Addressed all questions with technical accuracy.
    - 0.5: Vague or partial answers.
    - 0.1: Avoided the questions or spoke about unrelated topics.

    Return ONLY the numeric score as a float.
    """
    try:
        from src.ats_matcher import ATSMatcher
        temp_matcher = ATSMatcher()
        response = temp_matcher.client.models.generate_content(
            model="gemini-2.0-flash-lite", contents=prompt
        )
        score = float(response.text.strip())
        return min(max(score, 0.1), 1.0)
    except:
        return 0.5 # Default fallback

# --- 4. PHASE 5: DISCREPANCY DETECTION LOGIC ---
def detect_discrepancies(transcript, confidence, anxiety, attention):
    if not transcript or transcript == "No speech detected":
        return "Not enough data", "Low"
    sentiment = (TextBlob(transcript).sentiment.polarity + 1) / 2
    warnings = []
    severity = "Safe"
    if sentiment > 0.75 and anxiety > 0.55:
        warnings.append("Cognitive Dissonance: Verbal sentiment vs Facial Anxiety.")
        severity = "High"
    if confidence > 0.70 and attention < 0.45:
        warnings.append("Engagement Conflict: Vocal confidence vs Visual distraction.")
        severity = "Medium"
    return (" | ".join(warnings) if warnings else "Cues aligned.", severity)

# --- 5. CORE ANALYSIS HELPERS ---
def get_grade(score):
    if score >= 0.80: return "🏆 A+ (Elite)"
    if score >= 0.70: return "✅ A (Strong)"
    if score >= 0.50: return "⚠️ B (Average)"
    return "❌ F (Unsuitable)"

def analyze_behavior_with_hume(video_path):
    if not HUME_AVAILABLE or not HUME_API_KEY:
        return {"confidence": 0.75, "anxiety": 0.25}
    try:
        client = HumeClient(api_key=HUME_API_KEY)
        configs = [FaceConfig(identify_faces=True)]
        job = client.submit_job([], configs, files=[video_path])
        job.await_complete()
        results = job.get_predictions()
        predictions = results[0]['results']['predictions'][0]['models']['face']['grouped_predictions'][0]['predictions'][0]['emotions']
        emo_map = {e['name']: e['score'] for e in predictions}
        return {"confidence": round((emo_map.get("Calm", 0) + emo_map.get("Joy", 0))/2, 2), "anxiety": round(emo_map.get("Anxiety", 0), 2)}
    except: return {"confidence": 0.60, "anxiety": 0.40}

def analyze_video_vision(video_path):
    if not os.path.exists(MODEL_PATH): return 0.5 
    try:
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(video_path)
            duration_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30))
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
                    if 0.35 < nx < 0.65: hits += 1
            cap.release()
        return (hits / samples)**2 if samples > 0 else 0.1
    except: return 0.2

# --- 6. MAIN MULTIMODAL ENGINE ---
def extract_features(resume_path, jd_text, video_path, skip_ats=False, questions=None):
    print(f"🚀 HireSync Engine: Analyzing Behavioral Biometrics & Answer Relevance...")
    
    # 1. Run Local and Cloud Streams
    audio = audio_phase_score(video_path)
    attention = analyze_video_vision(video_path)
    hume_data = analyze_behavior_with_hume(video_path)

    # 2. Answer Relevance Check (Phase 8)
    relevance_score = 0.8 
    if questions:
        relevance_score = verify_answer_relevance(audio.get('transcript', ''), questions, jd_text)
    
    # 3. HARSH SCORING (Geometric Mean)
    f = max(0.01, audio.get('fluency_score', 0.5))
    c = max(0.01, audio.get('communication_score', 0.5))
    a = max(0.01, attention if attention is not None else 0.5)
    h = max(0.01, hume_data.get('confidence', 0.5))
    r = max(0.01, relevance_score)

    # Calculate Geometric Mean of 5 signals
    raw_behavior = (f * c * a * h * r) ** (1/5)
    
    # FIX: Explicitly define anx_p to avoid NameError
    anxiety_val = hume_data.get('anxiety', 0.2)
    anx_p = max(0, anxiety_val - 0.4) # Penalty if anxiety > 0.4
    
    behavior_score = round(raw_behavior - anx_p, 2)
    behavior_score = max(0.05, behavior_score)

    # 4. Phase 5 Discrepancy
    conflict_msg, conflict_level = detect_discrepancies(audio.get('transcript', ''), h, anxiety_val, a)

    res_data = {
        "behavior_score": behavior_score,
        "behavior_grade": get_grade(behavior_score),
        "fluency": round(f, 2),
        "communication": round(c, 2),
        "attention": round(a, 2),
        "relevance_score": round(r, 2),
        "hume_confidence": h,
        "hume_anxiety": anxiety_val,
        "duration": audio.get('duration', '00:00'),
        "transcript": audio.get('transcript', 'No speech detected'),
        "wpm": audio.get('wpm', 0),
        "conflict_report": conflict_msg,
        "integrity_status": conflict_level
    }

    if skip_ats: return res_data

    # 5. Document AI (Stage 1)
    # ... (Keep your existing ATS logic here)
    

    # 5. STAGE 1: DOCUMENT AI (Resume Matching)
    suitability = 0.0
    strengths = []
    if resume_path and jd_text:
        matcher = ATSMatcher()
        try:
            # Handle if resume_path is a file object (Gradio) or a string
            path = resume_path.name if hasattr(resume_path, 'name') else resume_path
            reader = PyPDF2.PdfReader(path)
            resume_text = " ".join([p.extract_text() for p in reader.pages])
            
            # Deep Gemini Analysis
            ats_res = matcher.analyze_resume_llm(resume_text, jd_text)
            
            suitability = ats_res.get('final_score', 0) / 100
            strengths = ats_res.get('matched_keywords', [])
            
            # Capture the XAI explanation (Strengths/Weaknesses)
            res_data["details"] = ats_res.get('explanation', {"strengths": "N/A", "weaknesses": "N/A"})
        except Exception as e:
            print(f"⚠️ ATS processing error: {e}")
            suitability = 0.5
            res_data["details"] = {"strengths": "Manual review required", "weaknesses": "Parsing error"}

    # Final Combined Score: 60% Technical Suitability + 40% Behavioral Performance
    final_score = round((suitability * 0.6) + (behavior_score * 0.4), 2)
    
    # Convert back to 0-100 scale for database consistency
    res_data.update({
        "suitability": round(suitability * 100, 1), 
        "final_score": round(final_score * 100, 1), 
        "strengths": strengths
    })
    
    return res_data


def upload_to_cloud(path, resource_type="auto"):
    try:
        target = "image" if path.lower().endswith(".pdf") else "video"
        res = cloudinary.uploader.upload(path, resource_type=target)
        return res.get('secure_url')
    except: return None

def save_candidate_to_supabase(data):
    try: supabase.table("candidates").insert(data).execute()
    except Exception as e: print(f"DB Error: {e}")