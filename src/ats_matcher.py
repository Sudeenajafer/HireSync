import os, re, json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class ATSMatcher:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def validate_name(self, name):
        name = name.strip()
        if len(name.split()) < 2:
            return False, "🛑 **Name Error**: Please enter your **First AND Last name**."
        if not re.match(r"^[a-zA-Z\s]+$", name):
            return False, "🛑 **Name Error**: Numbers and symbols are not allowed in names."
        return True, ""

    def validate_email(self, email):
        email = email.strip()
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            return False, "🛑 **Email Error**: Please enter a valid email address (e.g., name@gmail.com)."
        return True, ""

    def validate_phone(self, phone):
        clean_phone = "".join(filter(str.isdigit, phone))
        if len(clean_phone) < 10 or len(clean_phone) > 13:
            return False, "🛑 **Phone Error**: Please enter a valid 10-13 digit phone number."
        return True, ""

    def validate_resume_dna(self, file_path):
        if not file_path or not file_path.lower().endswith('.pdf'):
            return False, "🛑 **File Error**: The resume must be a **PDF document**."
        return True, ""

    def analyze_resume_llm(self, resume_text, jd_text):
        """Generates high-detail XAI reasoning with strict category matching."""
        print("🚀 [API] Calling Gemini for Strict ATS Match...")
        prompt = f"""
        Act as a Critical Technical Recruiter. Analyze the Resume against the Job Description.
        
        STRICT RULES:
        1. CATEGORY CHECK: If the Job is 'Business Analyst' and the Resume is 'ML Engineer', apply a 40% penalty.
        2. EVIDENCE CHECK: Only give points for skills explicitly stated with context.
        3. SCORING: 0-40 (Poor), 41-70 (Average), 71-100 (Exceptional).

        Return ONLY a JSON object:
        {{
            "final_score": int,
            "matched_keywords": ["list"],
            "explanation": {{
                "strengths": "Technical assets found",
                "weaknesses": "CRITICAL GAPS: Why this candidate is a mismatch...",
                "verdict": "A blunt professional summary of the alignment."
            }}
        }}
        Resume: {resume_text[:4000]} | JD: {jd_text[:2000]}
        """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite", contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return json.loads(response.text)
        except:
            return {"final_score": 30, "explanation": {"verdict": "System fallback due to limit."}, "matched_keywords": []}

    def generate_final_verdict_xai(self, name, ats_data, behavior_data):
        """Synthesizes all multimodal data into a forensic final decision."""
        prompt = f"""
        Act as a Senior Hiring Director. Provide a final decision for {name}.
        
        DATA:
        - ATS Match: {ats_data.get('final_score')}%
        - Interview Knowledge Accuracy: {behavior_data.get('relevance_score')*100}%
        - Confidence Level: {behavior_data.get('hume_confidence')*100}%
        - Attention/Focus: {behavior_data.get('attention')*100}%
        - Transcript: {behavior_data.get('transcript')}

        TASK:
        1. Compare the Transcript against the Job Field. If the candidate talked about unrelated topics (e.g., talked about ML for a BA job), REJECT THEM.
        2. Explain the discrepancy between technical match and behavioral performance.
        3. Write a 3-sentence verdict.
        """
        try:
            response = self.client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
            return response.text.strip()
        except:
            return "Manual review required: Candidate signals are inconsistent with the job role."
    def generate_questions_from_jd(self, jd_text):
        """MSc Phase 8: Standardized Technical Interview Generation."""
        prompt = f"Act as an Interviewer. Generate 3 technical questions for this JD: {jd_text[:1500]}. Question 1 must be an intro. Format as numbered list."
        try:
            response = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return response.text.strip()
        except: return "1. Please introduce yourself.\n2. What are your key technical strengths?\n3. Why do you want this role?"
        
        
    def generate_final_verdict_xai(self, name, ats_score, behavior_score, behavior_grade, strengths, gaps, integrity, transcript):
        """
        Fuses all multimodal data into a human-like final hiring verdict.
        """
        prompt = f"""
        Act as a Senior Recruitment Director. Write a 3-sentence 'Final Hiring Verdict' for {name}.
        
        CANDIDATE DATA:
        - Technical Match (ATS): {ats_score}%
        - Behavioral Score: {behavior_score}% (Grade: {behavior_grade})
        - Integrity Audit: {integrity}
        - Identified Strengths: {strengths}
        - Critical Technical Gaps: {gaps}
        - Interview Speech Excerpt: {transcript[:400]}...

        TASK:
        Explain the suitability of the candidate. 
        - If technical match is low but behavior is high, mention 'potential but needs training'.
        - If technical knowledge (relevance) was low during the interview, be critical.
        - If attention was low, mention 'lack of focus'.
        
        OUTPUT: Return only the 3-sentence professional paragraph.
        """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"Verdict Error: {e}")
            return f"Candidate {name} shows a {behavior_grade} level of interview performance. Technical alignment is at {ats_score}%. A secondary technical review is suggested to address identified gaps."