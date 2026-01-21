import os, re, json, time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class ATSMatcher:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate_questions_from_resume(self, resume_text, jd_text):
        """Standardized technical questions."""
        try:
            prompt = f"Generate 3 technical questions for this JD: {jd_text[:1000]} and Resume: {resume_text[:1000]}"
            response = self.client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
            return response.text.strip()
        except:
            return "1. Please introduce yourself.\n2. Describe your most complex technical project.\n3. How do you handle project deadlines?"

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
        """MSc Logic: Deep ATS with Quota Fallback."""
        print("🚀 [API] Calling Gemini 1.5 Flash...")
        # Use 'gemini-1.5-flash' - it has higher limits than 2.0-lite
        model_name = "gemini-2.5-flash-lite"
        
        prompt = f"""Act as an ATS. Compare Resume vs JD. Return JSON: 
        {{ "final_score": int, "explanation": {{ "strengths": "str", "weaknesses": "str" }}, "matched_keywords": [] }}
        Resume: {resume_text[:4000]} | JD: {jd_text[:2000]}"""

        try:
            response = self.client.models.generate_content(
                model=model_name, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"⚠️ Gemini Quota/Error: {e}")
            # FALLBACK: If API is exhausted, return a generic but valid response
            return {
                "final_score": 50, 
                "explanation": {"strengths": "Technical background detected", "weaknesses": "Detailed AI analysis unavailable due to API limits."},
                "matched_keywords": ["Analysis Pending"]
            }   
        
        
def validate_name(self, name):
        """Letters only, at least two words (First and Last name)."""
        name = name.strip()
        if len(name.split()) < 2:
            return False, "🛑 Please enter both your **First and Last name**."
        if not re.match(r"^[a-zA-Z\s]+$", name):
            return False, "🛑 Name should only contain **letters and spaces**."
        return True, ""

def validate_email(self, email):
        """Standard email regex check."""
        email = email.strip()
        # Pattern: letters/numbers + @ + domain + .extension
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            return False, "🛑 Please enter a **valid email address** (e.g., name@domain.com)."
        return True, ""
def validate_phone(self, phone):
        """Checks for a valid 10-digit phone number (standard for India/Global)."""
        phone = "".join(filter(str.isdigit, phone)) # Remove spaces, dashes, or (+)
        if len(phone) < 10 or len(phone) > 13:
            return False, "🛑 Phone number must be between **10 to 13 digits**."
        return True, ""

def validate_resume_dna(self, file_path):
        """Checks if the file is a PDF and has enough content."""
        if not file_path.lower().endswith('.pdf'):
            return False, "🛑 Only **PDF files** are allowed for resumes."
        
        # Check file size (should be > 10KB to be a real resume)
        if os.path.getsize(file_path) < 10000:
            return False, "🛑 The uploaded file is **too small** to be a valid resume."
        return True, ""
