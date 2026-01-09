from sentence_transformers import SentenceTransformer, util
import numpy as np
from nltk.tokenize import sent_tokenize
import pandas as pd

print("🚀 HireSync AI - Phase 1 ATS ✅ PRODUCTION READY")

model = SentenceTransformer('all-MiniLM-L6-v2')

resume_text = """
AI/ML Student Kerala - poco pie
ASL recognition ResNet/MobileNet/EfficientNet Colab T4 GPU
TensorFlow/Keras PPO RL Gemini Nightbot chatbot streaming
"""

jd_text = """
AI/ML Engineer: Deep learning TensorFlow/Keras, CV ResNet/MobileNet
Colab deployment Gemini API integration real-time systems
"""

resume_sents = sent_tokenize(resume_text)
jd_sents = sent_tokenize(jd_text)

# Safe NumPy pipeline
resume_emb = model.encode(resume_sents, convert_to_numpy=True)
jd_emb = model.encode(jd_sents, convert_to_numpy=True)

cos_scores = util.cos_sim(resume_emb, jd_emb)

# FORCE NumPy conversion
cos_scores = cos_scores if isinstance(cos_scores, np.ndarray) else cos_scores.numpy()
max_scores = np.max(cos_scores, axis=1)
ats_score = np.mean(max_scores)

# FIXED: Numeric column for nlargest
df = pd.DataFrame({
    'Resume Sentence': resume_sents,
    'JD Match Score': pd.to_numeric(max_scores)  # Numeric dtype
})

print(f"📄 Resume chunks: {len(resume_sents)}")
print(f"🎯 ATS SCORE: {ats_score:.3f}")
print("\n📊 BEST MATCHES:")
print(df.nlargest(3, 'JD Match Score')[['Resume Sentence', 'JD Match Score']].round(3).to_string(index=False))

df.to_csv('hiresync_report.csv', index=False)
print("\n✅ Report saved: hiresync_report.csv")
print("🎓 MSc DEMO LIVE!")
