import streamlit as st
import pandas as pd
import numpy as np
import nltk
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer





try:
    model = pickle.load(open("clf.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf.pkl", "rb"))
    st.success("Model & vectorizer loaded successfully!")
except:
    model = None
    vectorizer = TfidfVectorizer(stop_words="english")
    st.warning("No model found. Using only text analysis.")


def clean_text(txt):
    # Remove URLs
    cleanText = re.sub(r'http\S+', ' ', txt)

    # Remove RT and CC
    cleanText = re.sub(r'\b(RT|CC)\b', ' ', cleanText)

    # Remove hashtags
    cleanText = re.sub(r'#\S+', ' ', cleanText)

    # Remove mentions
    cleanText = re.sub(r'@\S+', ' ', cleanText)

    # Remove punctuations and special characters
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)

    # Remove non-ASCII characters
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)

    # Replace multiple spaces with a single space
    cleanText = re.sub(r'\s+', ' ', cleanText).strip()

    return cleanText.strip().lower()



SKILLS_DB = ["python", "java", "sql", "excel", "machine learning", "deep learning",
             "nlp", "flask", "django", "react", "javascript", "html", "css", "aws"]

def extract_skills(text):
    skills_found = []
    for skill in SKILLS_DB:
        if skill.lower() in text:
            skills_found.append(skill.capitalize())
    return skills_found


st.set_page_config(page_title="Resume Analyzer", layout="wide")
st.title("📄 AI Resume Analyzer")

uploaded_file = st.file_uploader("Upload your Resume (TXT or PDF)", type=["txt", "pdf"])

if uploaded_file is not None:
    # Read file
    raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
    cleaned_text = clean_text(raw_text)

    st.subheader(" Resume Preview")
    st.write(raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text)

    # Skills extraction
    st.subheader(" Skills Found")
    skills = extract_skills(cleaned_text)
    if skills:
        st.success(f"Skills: {', '.join(skills)}")
    else:
        st.warning("No known skills found in resume.")

    # Prediction
    if model:
        X = vectorizer.transform([cleaned_text])
        prediction = model.predict(X)[0]
        st.subheader("📌 Predicted Job Category")
        st.success(prediction)
    else:
        st.info("No ML model loaded — only skills extraction is available.")
