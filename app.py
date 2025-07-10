import streamlit as st
import fitz  # PyMuPDF
import docx2txt
import os
import re
import nltk
import pandas as pd
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')

# ---------- TEXT PROCESSING HELPERS ----------

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

def extract_sections(text):
    sections = {
        "skills": "",
        "experience": "",
        "education": "",
        "achievements": ""
    }

    text_lower = text.lower()
    idxs = {
        "skills": text_lower.find("skill"),
        "experience": text_lower.find("experience"),
        "education": text_lower.find("education"),
        "achievements": text_lower.find("achievement")
    }

    sorted_idxs = sorted([(i, s) for s, i in idxs.items() if i != -1])
    for i in range(len(sorted_idxs)):
        start_idx, sec = sorted_idxs[i]
        end_idx = sorted_idxs[i + 1][0] if i + 1 < len(sorted_idxs) else len(text)
        sections[sec] = preprocess_text(text[start_idx:end_idx])

    return sections

# ---------- MATCHING LOGIC ----------

def match_sections(jd_sections, resume_sections, weights):
    total_score = 0.0
    section_scores = {}

    for section in weights:
        jd_text = jd_sections.get(section, "")
        resume_text = resume_sections.get(section, "")

        if jd_text.strip() == "" or resume_text.strip() == "":
            score = 0
        else:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([jd_text, resume_text])
            score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

        section_scores[section] = score
        total_score += weights[section] * score

    return total_score, section_scores

# ---------- EXCEL EXPORT ----------

def save_to_excel(results):
    df = pd.DataFrame(results)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Match Results')
    return output.getvalue()

# ---------- STREAMLIT APP ----------

st.set_page_config(page_title="Resume Matcher", layout="centered")
st.title("📄 Automated Resume Matcher with Section Selection")

st.sidebar.header("Upload Files")
jd_file = st.sidebar.file_uploader("Upload Job Description (PDF/DOCX)", type=['pdf', 'docx'])
resume_files = st.sidebar.file_uploader("Upload Resumes (PDF/DOCX)", type=['pdf', 'docx'], accept_multiple_files=True)

# Section selection
aspects = st.sidebar.multiselect(
    "🔍 Choose Resume Sections to Match",
    ["Skills", "Experience", "Education", "Achievements"],
    default=["Skills", "Experience", "Education"]
)

aspect_map = {
    "Skills": "skills",
    "Experience": "experience",
    "Education": "education",
    "Achievements": "achievements"
}

selected_sections = [aspect_map[a] for a in aspects]
weights = {sec: 1 / len(selected_sections) for sec in selected_sections} if selected_sections else {}

if st.sidebar.button("🔍 Match Resumes"):
    if not jd_file or not resume_files:
        st.warning("Please upload both job description and at least one resume.")
    else:
        jd_raw = extract_text(jd_file)
        jd_sections = extract_sections(jd_raw)
        jd_full_text = preprocess_text(jd_raw)

        results = []

        for resume in resume_files:
            resume_raw = extract_text(resume)
            resume_sections = extract_sections(resume_raw)
            resume_full_text = preprocess_text(resume_raw)

            if selected_sections:
                total_score, section_scores = match_sections(jd_sections, resume_sections, weights)
            else:
                # fallback to full-text matching
                vectorizer = TfidfVectorizer()
                tfidf = vectorizer.fit_transform([jd_full_text, resume_full_text])
                total_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                section_scores = {}

            result = {
                "Resume Name": resume.name,
                "Total Match Score": round(total_score, 2)
            }

            for sec in selected_sections:
                result[f"{sec.capitalize()} Score"] = round(section_scores.get(sec, 0.0), 2)

            results.append(result)

        sorted_results = sorted(results, key=lambda x: x["Total Match Score"], reverse=True)

        st.subheader("🏆 Top Matching Resumes")
        for res in sorted_results:
            st.markdown(f"**{res['Resume Name']}** — Match Score: `{res['Total Match Score']}`")

        excel_data = save_to_excel(sorted_results)
        st.download_button(
            label="📥 Download Excel Report",
            data=excel_data,
            file_name='resume_match_results.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
