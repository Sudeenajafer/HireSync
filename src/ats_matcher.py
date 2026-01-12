import fitz
from sentence_transformers import SentenceTransformer, util
import numpy as np
import nltk
import re

try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    pass

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

class ATSMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.stop_words = set(stopwords.words('english'))
        # Core AI/Tech terms that trigger "Bonus Points"
        self.tech_dictionary = {
            'python', 'machine learning', 'ai', 'artificial intelligence', 
            'sql', 'flutter', 'firebase', 'java', 'cpp', 'c++', 'algorithms', 
            'data structures', 'nlp', 'natural language processing', 'tensorflow', 
            'pytorch', 'computer vision', 'git', 'deep learning'
        }

    def clean_text(self, text):
        text = text.replace('\n', ' ')
        # Remove contact info so it doesn't dilute the score
        text = re.sub(r'\S*@\S*\s?', '', text) 
        text = re.sub(r'\+?\d[\d -]{8,12}\d', '', text)
        return re.sub(r'\s+', ' ', text).strip().lower()

    def analyze_resume(self, resume_text, jd_path):
        with open(jd_path, 'r', encoding='utf-8') as f:
            jd_text = f.read()

        clean_resume = self.clean_text(resume_text)
        clean_jd = self.clean_text(jd_text)

        # 1. SEGMENTATION & CORE REQUIREMENTS
        # We split JD into small logical chunks (Requirements)
        jd_reqs = [s for s in sent_tokenize(jd_text) if len(s.split()) > 4]
        # We focus on the most important technical sentences in the resume
        res_sents = [s for s in sent_tokenize(resume_text) if len(s.split()) > 3]

        jd_emb = self.model.encode(jd_reqs, convert_to_numpy=True)
        res_emb = self.model.encode(res_sents, convert_to_numpy=True)

        # 2. REQUIREMENT COVERAGE (The MSc Edge)
        # For every requirement in the JD, find the best match in the Resume
        cos_scores = util.cos_sim(jd_emb, res_emb)
        best_matches = np.max(cos_scores.cpu().detach().numpy(), axis=1)

        # Logic: If a candidate meets a requirement at > 0.55, it's a "Full Match"
        # If they meet it at > 0.40, it's a "Partial Match"
        full_matches = np.sum(best_matches > 0.55)
        partial_matches = np.sum((best_matches <= 0.55) & (best_matches > 0.35))
        
        coverage_ratio = (full_matches + (partial_matches * 0.5)) / len(jd_reqs)

        # 3. KEYWORD SPECIFICITY
        # Check for exact matches of high-value technical terms
        found_skills = [skill for skill in self.tech_dictionary if skill in clean_resume]
        jd_skills = [skill for skill in self.tech_dictionary if skill in clean_jd]
        
        skill_score = len(found_skills) / len(jd_skills) if jd_skills else 0.5

        # 4. FINAL FUSION & SCALING
        # We weight Coverage at 70% and Skill-Density at 30%
        raw_score = (coverage_ratio * 0.7) + (skill_score * 0.3)
        
        # --- THE "PROFESSIONAL" RESCALER ---
        # In scientific AI, 0.45 is 'Good'. In Recruitment, 'Good' must be 80%.
        # We use a non-linear boost to lift high-quality matches into the 80s.
        if raw_score > 0.40:
            final_display_score = 0.65 + (raw_score * 0.35)
        else:
            final_display_score = raw_score * 1.2
            
        # Ensure we don't exceed 98% unless it's a literal copy-paste
        final_display_score = min(final_display_score, 0.98)

        # Identify top 8 skills for the badges
        top_skills = [s.upper() for s in found_skills if s in jd_skills][:10]

        return float(final_display_score), top_skills
    
    def validate_resume(self, text):
        """
        High-Efficiency Structural Validator
        Checks for: 1. Header Density, 2. Timeline (Years), 3. Prohibited Patterns (Academic Papers)
        """
        text_lower = text.lower()
        score = 0
        
        # 1. Check for Timeline/Years (Standard in Resumes)
        # Matches years like 2018, 2022, 2024-Present
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        if len(years) >= 2: score += 40  # Good sign: contains a timeline
        
        # 2. Check for Professional Section Headers
        headers = ['education', 'experience', 'skills', 'projects', 'employment', 'internship', 'languages']
        found_headers = [h for h in headers if h in text_lower]
        score += (len(found_headers) * 15) # Each header adds weight
        
        # 3. PROHIBITED PATTERNS (Things found in books/papers but NOT resumes)
        invalid_patterns = ['abstract', 'table of contents', 'chapter', 'references', 'bibliography', 'introduction']
        for pattern in invalid_patterns:
            if re.search(rf'\b{pattern}\b', text_lower):
                score -= 60 # Heavy penalty for academic paper structure
        
        # 4. Length Check (Words)
        word_count = len(text.split())
        
        # FINAL LOGIC
        # A valid resume must score at least 50 points and be a reasonable length
        if score < 50:
            return False, "🛑 **Invalid Document Structure**: This does not appear to be a professional resume. It is missing a timeline (years) or standard section headers."
        
        if word_count < 40:
            return False, "🛑 **Document Too Short**: A valid resume must be more than 40 words."
            
        if word_count > 2500:
            return False, "🛑 **Document Too Long**: Resumes are typically 1-3 pages. This document is too large."

        return True, "Valid"
    def validate_name(self, name):
        """Checks if the name is realistic (Letters only, at least 2 words)"""
        name = name.strip()
        if len(name) < 3:
            return False, "🛑 **Invalid Name**: Name is too short."
        # Check if it contains only letters and spaces
        if not re.match(r"^[a-zA-Z\s]+$", name):
            return False, "🛑 **Invalid Name**: Name should only contain letters."
        # Check if it has at least a First and Last name
        if len(name.split()) < 2:
            return False, "🛑 **Invalid Name**: Please enter both First and Last name."
        return True, "Valid"

    def validate_jd(self, text):
        """Checks if the JD contains hiring-related language"""
        text_lower = text.lower()
        # Job Description 'DNA' keywords
        jd_markers = [
            'requirement', 'responsibility', 'qualification', 'experience', 
            'looking for', 'benefits', 'candidate', 'skills', 'role', 'apply'
        ]
        found_markers = [m for m in jd_markers if m in text_lower]
        
        word_count = len(text.split())

        if len(found_markers) < 2:
            return False, "🛑 **Invalid Job Description**: This file is missing standard job requirements or responsibilities."
        
        if word_count < 30:
            return False, "🛑 **Job Description Too Short**: Please provide more details about the role."
            
        return True, "Valid"