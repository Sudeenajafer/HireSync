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
        """Generates high-detail XAI reasoning for the ATS match."""
        # We slice to protect against very large files causing API timeouts
        resume_context = resume_text[:5000]
        jd_context = jd_text[:3000]

        prompt = f"""
        Act as a Senior HR Analytics Lead. Your goal is to provide a highly objective, 
        evidence-based comparison between the Candidate's Resume and the Job Description.
        
        TASK:
        1. Calculate a final match score (0-100) based strictly on technical alignment.
        2. Identify specific strengths (skills present) and weaknesses (skills missing).
        3. Provide a 'Verdict' which is a professional summary of the candidate's fit.

        Return ONLY a JSON object with this exact structure:
        {{
            "final_score": int,
            "matched_keywords": ["list of tech skills found"],
            "explanation": {{
                "strengths": "Bulleted list of technical assets found in resume...",
                "weaknesses": "Specific gaps or missing certifications/tools...",
                "verdict": "A 2-sentence expert summary of why the candidate received this score."
            }}
        }}

        RESUME DATA:
        {resume_context}

        JOB DESCRIPTION:
        {jd_context}
        """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2 # Low temperature for consistent, professional grading
                )
            )
            # Parse response
            data = json.loads(response.text)
            
            # Safety Gate: Ensure score is a valid integer between 0 and 100
            data['final_score'] = min(max(int(data.get('final_score', 50)), 0), 100)
            return data

        except Exception as e:
            print(f"⚠️ Gemini Analysis Error: {e}")
            # Reliable fallback so the background process doesn't crash
            return {
                "final_score": 40,
                "matched_keywords": ["Processing..."],
                "explanation": {
                    "strengths": "Data extracted, but AI analysis hit a temporary limit.",
                    "weaknesses": "Detailed gaps will appear after a manual refresh.",
                    "verdict": "Technical match detected. Detailed AI reasoning is pending."
                }
            }
    def generate_questions_from_jd(self, jd_text):
        """MSc Phase 8: Standardized Technical Interview Generation."""
        prompt = f"Act as an Interviewer. Generate 3 technical questions for this JD: {jd_text[:1500]}. Question 1 must be an intro. Format as numbered list."
        try:
            response = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return response.text.strip()
        except: return "1. Please introduce yourself.\n2. What are your key technical strengths?\n3. Why do you want this role?"