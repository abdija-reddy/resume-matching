import streamlit as st
import fitz
import docx2txt
import os
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set custom NLTK data path (still needed for stopwords)
nltk.download('stopwords')

# --- Functions from backend ---

def extract_text(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".pdf":
        text = ""
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif ext == ".docx":
        return docx2txt.process(file)
    else:
        return ""

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]
    return ' '.join(tokens)

def match_resumes(jd_text, resumes):
    all_texts = [jd_text] + resumes
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    jd_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]
    similarity_scores = cosine_similarity(jd_vector, resume_vectors).flatten()
    return similarity_scores

# --- Streamlit App ---
st.set_page_config(page_title="Resume Matcher", layout="centered")
st.title("📄 Automated Resume Matcher")

st.sidebar.header("Upload Files")

jd_file = st.sidebar.file_uploader("Upload Job Description (PDF/DOCX)", type=['pdf', 'docx'])
resume_files = st.sidebar.file_uploader("Upload Resumes (PDF/DOCX)", type=['pdf', 'docx'], accept_multiple_files=True)

if st.sidebar.button("🔍 Match Resumes"):
    if not jd_file or not resume_files:
        st.warning("Please upload both job description and at least one resume.")
    else:
        jd_text = preprocess_text(extract_text(jd_file))
        resume_texts = []
        resume_names = []

        for resume in resume_files:
            text = extract_text(resume)
            resume_texts.append(preprocess_text(text))
            resume_names.append(resume.name)

        scores = match_resumes(jd_text, resume_texts)
        results = sorted(zip(resume_names, scores), key=lambda x: x[1], reverse=True)

        st.subheader("🏆 Top Matching Resumes")
        for name, score in results:
            st.markdown(f"**{name}** — Match Score: `{score:.2f}`")

