import os, re, json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class ATSMatcher:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def validate_name(self, name):
        if not name or len(name.split()) < 2: return False, "🛑 Enter First and Last Name."
        return True, ""

    def validate_email(self, email):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return (True, "") if re.match(pattern, email) else (False, "🛑 Invalid Email format.")

    def validate_phone(self, phone):
        clean = "".join(filter(str.isdigit, phone))
        return (True, "") if 10 <= len(clean) <= 13 else (False, "🛑 Phone must be 10-13 digits.")

    def analyze_resume_llm(self, resume_text, jd_text):
        prompt = f"""Act as an ATS. Compare Resume vs JD. Return JSON ONLY: 
        {{ "final_score": int, "explanation": {{ "strengths": "str", "weaknesses": "str", "verdict": "str" }}, "matched_keywords": [] }}
        Resume: {resume_text[:4000]} | JD: {jd_text[:2000]}"""
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite", contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return json.loads(response.text)
        except: return {"final_score": 50, "explanation": {"strengths": "Manual review needed", "weaknesses": "API Limit"}, "matched_keywords": []}

    def generate_questions_from_jd(self, jd_text):
        """MSc Phase 8: Standardized Technical Interview Generation."""
        prompt = f"Act as an Interviewer. Generate 3 technical questions for this JD: {jd_text[:1500]}. Question 1 must be an intro. Format as numbered list."
        try:
            response = self.client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
            return response.text.strip()
        except: return "1. Please introduce yourself.\n2. What are your key technical strengths?\n3. Why do you want this role?"