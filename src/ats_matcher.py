import os
import re
import json
from google import genai # Modern 2026 SDK
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class ATSMatcher:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate_questions(self, jd_text):
        """MSc Logic: Gemini-driven adaptive interview question generation"""
        prompt = f"""
        Based on the following Job Description, generate 3 professional behavioral interview questions 
        that will test the candidate's technical and soft skills.
        Format the output as a simple numbered list.
        JD: {jd_text[:2000]}
        """
        try:
            response = self.client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
            return response.text.strip()
        except:
            return "1. Tell us about your experience.\n2. How do you handle challenges?\n3. Why do you want this role?"

    def validate_name(self, name):
        if not name or len(name.split()) < 2: 
            return False, "🛑 Please enter both First and Last name."
        return True, ""

    def validate_resume(self, text):
        markers = ['education', 'experience', 'skills']
        if not any(m in text.lower() for m in markers):
            return False, "🛑 This PDF doesn't look like a resume."
        return True, ""


    def analyze_resume_llm(self, resume_text, jd_text):
        print("🚀 [API DEBUG] Calling Modern Google GenAI (Gemini 1.5)...")
        
        prompt = f"""
        Act as an expert ATS. Analyze the Resume against the JD.
        Return ONLY a JSON object:
        {{
            "education_score": int, "skills_score": int, "experience_score": int,
            "final_score": int, "matched_skills": [], "missing_skills": [],
            "reasoning": "str", "suggestions": "str"
        }}
        Resume: {resume_text[:5000]} | JD: {jd_text[:3000]}
        """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Gemini SDK Error: {e}")
            return None
        
        
        