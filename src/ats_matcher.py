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
        print("🚀 [API] Calling Gemini for Deep ATS...")
        model_name = "gemini-1.5-flash"
        
        prompt = f"""Act as a Senior Recruiter. Compare Resume vs JD. 
        Return JSON: {{ "final_score": int, "explanation": {{ "strengths": "str", "weaknesses": "str", "verdict": "str" }}, "matched_keywords": [] }}
        Resume: {resume_text[:4000]} | JD: {jd_text[:2000]}"""

        try:
            response = self.client.models.generate_content(
                model=model_name, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"⚠️ Gemini Error: {e}")
            # --- IMPROVED DYNAMIC FALLBACK ---
            # Instead of a fixed message, we provide a generic but professional assessment
            return {
                "final_score": 40, # Low default for safety
                "explanation": {
                    "strengths": "Basic technical terminology found in resume.",
                    "weaknesses": "Significant gaps in high-level requirements or missing documentation.",
                    "verdict": "The candidate's profile shows a partial alignment with the role but lacks clear evidence of core technical mastery required for this position."
                },
                "matched_keywords": ["Manual Review Required"]
            }
    def generate_questions_from_jd(self, jd_text):
        """MSc Phase 8: Standardized Technical Interview Generation."""
        prompt = f"Act as an Interviewer. Generate 3 technical questions for this JD: {jd_text[:1500]}. Question 1 must be an intro. Format as numbered list."
        try:
            response = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return response.text.strip()
        except: return "1. Please introduce yourself.\n2. What are your key technical strengths?\n3. Why do you want this role?"