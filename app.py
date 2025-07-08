import streamlit as st
import fitz
import docx2txt
import os
import re
import nltk
import pandas as pd
import io
from fpdf import FPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')

# Dark theme styling
st.markdown("""
    <style>
        .stApp {
            background-color: #000000;
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #1E1E1E;
            color: white;
            border: 1px solid white;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .css-1cpxqw2, .css-h5rgaw { color: white !important; }
    </style>
""", unsafe_allow_html=True)

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

def anonymize(text):
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    text = re.sub(r'\b\d{10}\b', '', text)
    text = re.sub(r'\b(mr|mrs|ms|miss|he|she|him|her)\b', '', text, flags=re.IGNORECASE)
    return text

def preprocess_text(text):
    text = anonymize(text)
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

def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Top Matched Resumes", ln=True, align='C')
    pdf.ln(10)
    for name, score in data:
        pdf.cell(200, 10, txt=f"{name} — Score: {score:.2f}", ln=True)
    buffer = io.BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()

# Streamlit App
st.set_page_config(page_title="Resume Matcher", layout="centered")
st.title("📄 Smart Resume Matcher ")

col1, col2 = st.columns(2)

with col1:
    jd_file = st.file_uploader("📄 Upload Job Description", type=['pdf', 'docx'])

with col2:
    resume_files = st.file_uploader("👤 Upload Resumes", type=['pdf', 'docx'], accept_multiple_files=True)

if st.button("🔍 Match Resumes"):
    if not jd_file or not resume_files:
        st.warning("Please upload both job description and at least one resume.")
    else:
        with st.spinner("Matching resumes..."):
            jd_text = preprocess_text(extract_text(jd_file))
            resume_texts = []
            resume_names = []

            for resume in resume_files:
                text = extract_text(resume)
                resume_texts.append(preprocess_text(text))
                resume_names.append(resume.name)

            scores = match_resumes(jd_text, resume_texts)
            results = sorted(zip(resume_names, scores), key=lambda x: x[1], reverse=True)

        st.subheader("🎯 Top Matching Resumes")

        top_n = st.slider("Select number of top candidates to show:", 1, len(results), min(5, len(results)))
        top_results = results[:top_n]

        for name, score in top_results:
            with st.expander(f"{name} — Match Score: {score:.2f}"):
                st.write("Resume matches well with the job description.")

        # CSV download
        df_results = pd.DataFrame(top_results, columns=["Candidate", "Match Score"])
        csv_buffer = io.StringIO()
        df_results.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Top Results (CSV)",
            data=csv_buffer.getvalue(),
            file_name="top_ranked_resumes.csv",
            mime="text/csv"
        )

        # PDF download
        pdf_data = generate_pdf(top_results)
        st.download_button(
            label="📄 Download Top Results (PDF)",
            data=pdf_data,
            file_name="top_ranked_resumes.pdf",
            mime="application/pdf"
        )
