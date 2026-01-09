import fitz
from sentence_transformers import SentenceTransformer, util
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import os

class ATSMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def extract_pdf_text(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    
    def compute_match_score(self, resume_text, jd_text):
        resume_sents = sent_tokenize(resume_text)
        jd_sents = sent_tokenize(jd_text)
        
        if not resume_sents or not jd_sents:
            return 0.0, []
        
        # BULLETPROOF: Always NumPy
        resume_emb = self.model.encode(resume_sents, convert_to_numpy=True)
        jd_emb = self.model.encode(jd_sents, convert_to_numpy=True)
        
        cos_scores = util.cos_sim(resume_emb, jd_emb)
        max_scores = np.max(cos_scores.cpu().detach().numpy(), axis=1)
        avg_score = float(np.mean(max_scores))
        
        return avg_score, max_scores.tolist()
    
    def analyze_resume(self, resume_input, jd_path, output_csv='ats_report.csv'):
        if isinstance(resume_input, str) and os.path.isfile(resume_input) and resume_input.endswith('.pdf'):
            resume_text = self.extract_pdf_text(resume_input)
        else:
            resume_text = resume_input
        
        with open(jd_path, 'r', encoding='utf-8') as f:
            jd_text = f.read()
        
        score, scores = self.compute_match_score(resume_text, jd_text)
        
        df = pd.DataFrame({
            'Resume Sentence': sent_tokenize(resume_text),
            'Score': [f"{s:.3f}" for s in scores]
        })
        df.to_csv(output_csv, index=False)
        
        print(f"🎯 ATS Score: {score:.3f}")
        return score, df
