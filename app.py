# -*- coding: utf-8 -*-

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

st.set_page_config(page_title="Enhanced Resume Matcher", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #0e1117; color: white; }
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Enhanced Resume Matcher</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Resume Matcher", "Match Report"])

with tab1:
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            jd_file = st.file_uploader("Upload Job Description (PDF/DOCX)", type=['pdf', 'docx'])
        with col2:
            resume_files = st.file_uploader("Upload Resumes (PDF/DOCX)", type=['pdf', 'docx'], accept_multiple_files=True)

    aspects = st.multiselect(
        "Choose Resume Sections to Match",
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

    if st.button("Start Matching"):
        if not jd_file or not resume_files:
            st.warning("Please upload both job description and at least one resume.")
        else:
            jd_raw = extract_text(jd_file)
            jd_sections = extract_sections(jd_raw)
            jd_full_text = preprocess_text(jd_raw)
            results = []
            resume_bytes_dict = {r.name: r.read() for r in resume_files}

            for resume in resume_files:
                file_bytes = resume_bytes_dict[resume.name]
                resume.seek(0)
                ext = os.path.splitext(resume.name)[1].lower()[1:]
                resume_raw = extract_text(resume)
                resume_sections = extract_sections(resume_raw)
                resume_full_text = preprocess_text(resume_raw)

                if selected_sections:
                    total_score, section_scores = match_sections(jd_sections, resume_sections, weights)
                else:
                    vectorizer = TfidfVectorizer()
                    tfidf = vectorizer.fit_transform([jd_full_text, resume_full_text])
                    total_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                    section_scores = {}

                encoded = base64.b64encode(file_bytes).decode()
                href = f'<a href="data:application/{ext};base64,{encoded}" download="{resume.name}" target="_blank">{resume.name}</a>'

                result = {
                    "Resume": href,
                    "Total Match Score": round(total_score, 2)
                }
                for sec in selected_sections:
                    result[f"{sec.capitalize()} Score"] = round(section_scores.get(sec, 0.0), 2)

                results.append(result)

            sorted_results = sorted(results, key=lambda x: x["Total Match Score"], reverse=True)
            st.session_state["results"] = sorted_results
            st.success("Matching complete! View results in the Match Report tab.")

with tab2:
    if "results" in st.session_state:
        sorted_results = st.session_state["results"]
        st.subheader("Top Matching Resumes")
        styled_table = pd.DataFrame(sorted_results)
        st.write(styled_table.to_html(escape=False, index=False), unsafe_allow_html=True)

        chart_data = pd.DataFrame([
            {"Resume Name": re.sub('<.*?>', '', r["Resume"]), "Score": r["Total Match Score"]}
            for r in sorted_results
        ])
        chart = alt.Chart(chart_data).mark_bar().encode(
            x="Score:Q",
            y=alt.Y("Resume Name:N", sort='-x'),
            tooltip=["Resume Name", "Score"]
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)

        downloadable = [
            {k: re.sub('<.*?>', '', v) if k == "Resume" else v for k, v in r.items()}
            for r in sorted_results
        ]
        excel_data = save_to_excel(downloadable)
        st.download_button(
            label="Download Excel Report",
            data=excel_data,
            file_name='resume_match_results.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

        st.markdown(f"**Highest Score:** `{max([r['Total Match Score'] for r in sorted_results]):.2f}`")
        st.markdown(f"**Average Score:** `{sum([r['Total Match Score'] for r in sorted_results])/len(sorted_results):.2f}`")

st.markdown("""
---
<p style='text-align: center;'>Made by CodeSquad</p>
""", unsafe_allow_html=True)

