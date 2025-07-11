# Enhanced Resume Matcher with Education Filter and Stats

import streamlit as st
import fitz  # PyMuPDF
import docx2txt
import os
import re
import nltk
import pandas as pd
import base64
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

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
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(nltk.corpus.stopwords.words('english')) - {'not', 'no', 'nor'}
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def extract_sections(text):
    sections = {
        "skills": "",
        "experience": "",
        "education": "",
        "achievements": "",
        "projects": "",
        "certifications": "",
        "objective": "",
        "interests": ""
    }
    text_lower = text.lower()
    idxs = {
        "skills": text_lower.find("skill"),
        "experience": text_lower.find("experience"),
        "education": text_lower.find("education"),
        "achievements": text_lower.find("achievement"),
        "projects": text_lower.find("project"),
        "certifications": text_lower.find("certification"),
        "objective": text_lower.find("objective"),
        "interests": text_lower.find("interest")
    }
    sorted_idxs = sorted([(i, s) for s, i in idxs.items() if i != -1])
    for i in range(len(sorted_idxs)):
        start_idx, sec = sorted_idxs[i]
        end_idx = sorted_idxs[i + 1][0] if i + 1 < len(sorted_idxs) else len(text)
        sections[sec] = preprocess_text(text[start_idx:end_idx])
    return sections

# ---------- EDUCATION CLASSIFICATION ----------

def classify_education_section(text):
    categories = {
        "School": ["ssc", "cbse", "icse", "matriculation", "secondary"],
        "Intermediate/Diploma": ["intermediate", "diploma", "junior college"],
        "Undergraduate": ["btech", "b.e", "bachelor", "undergraduate"],
        "Postgraduate": ["mtech", "m.e", "master", "postgraduate"]
    }
    text = text.lower()
    classification = set()
    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            classification.add(category)
    return ", ".join(classification) if classification else "Unclassified"

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

st.set_page_config(page_title="Enhanced Resume Matcher", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #0e1117; color: white; }
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Enhanced Resume Matcher</h1>", unsafe_allow_html=True)

jd_file = st.file_uploader("Upload Job Description (PDF/DOCX)", type=['pdf', 'docx'])
resume_files = st.file_uploader("Upload Resumes (PDF/DOCX)", type=['pdf', 'docx'], accept_multiple_files=True)

aspects = st.multiselect(
    "Choose Resume Sections to Match",
    ["Skills", "Experience", "Education", "Achievements", "Projects", "Certifications", "Objective", "Interests"]
)

min_score = st.number_input("Set Minimum Qualification Score (%) (optional)", min_value=0, max_value=100, value=0, step=1)
analyze_edu = st.checkbox("Analyze Education Categories (Not Based on JD)")

aspect_map = {
    "Skills": "skills",
    "Experience": "experience",
    "Education": "education",
    "Achievements": "achievements",
    "Projects": "projects",
    "Certifications": "certifications",
    "Objective": "objective",
    "Interests": "interests"
}
selected_sections = [aspect_map[a] for a in aspects]
weights = {sec: 1 / len(selected_sections) for sec in selected_sections} if selected_sections else {}

if st.button("Start Matching"):
    if not jd_file or not resume_files:
        st.warning("Please upload both job description and at least one resume.")
    else:
        jd_raw = extract_text(jd_file)
        jd_sections = extract_sections(jd_raw)
        jd_full_text = preprocess_text(jd_raw)
        results = []

        for resume in resume_files:
            ext = os.path.splitext(resume.name)[1].lower()[1:]
            text = extract_text(resume)
            sections = extract_sections(text)
            full_text = preprocess_text(text)

            if selected_sections:
                total_score, section_scores = match_sections(jd_sections, sections, weights)
            else:
                vectorizer = TfidfVectorizer()
                tfidf = vectorizer.fit_transform([jd_full_text, full_text])
                total_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                section_scores = {}

            result = {"Resume": resume.name}
            if len(selected_sections) > 1:
                result["Total Match Score (%)"] = round(total_score * 100, 2)
            elif len(selected_sections) == 1:
                sec = selected_sections[0].capitalize()
                result[f"{sec} Score (%)"] = round(section_scores.get(selected_sections[0], 0.0) * 100, 2)
            else:
                result["Score (%)"] = round(total_score * 100, 2)

            if analyze_edu:
                result["Education Category"] = classify_education_section(sections.get("education", ""))

            results.append(result)

        df = pd.DataFrame(results)

        if analyze_edu and "Education Category" in df.columns:
            edu_counts = df["Education Category"].value_counts().reset_index()
            edu_counts.columns = ["Education Category", "Count"]
            st.subheader("Education Category Distribution")
            st.bar_chart(edu_counts.set_index("Education Category"))

            selected_filter = st.multiselect("Filter by Education Category", options=edu_counts["Education Category"].tolist())
            if selected_filter:
                df = df[df["Education Category"].isin(selected_filter)]

        if min_score > 0:
            score_col = "Total Match Score (%)" if "Total Match Score (%)" in df.columns else df.columns[-1]
            df = df[df[score_col] >= min_score]

        st.dataframe(df)
        st.download_button("Download Results", save_to_excel(df.to_dict('records')), file_name="results.xlsx")
