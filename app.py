import streamlit as st
import pandas as pd
import pickle
import re
from pathlib import Path

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide"
)

# =========================
# File Paths
# =========================
MODEL_PATH = Path("models/best_email_threat_model.pkl")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.pkl")
DATA_PATH = Path("data/final_balanced_multilingual_email_dataset.csv")

# =========================
# Load Model and Vectorizer
# =========================
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(VECTORIZER_PATH, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer


# =========================
# Load Dataset
# =========================
@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


# =========================
# Text Cleaning Function
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# Main App
# =========================
st.title("📧 Email Spam Detector")
st.write("This app predicts whether an email is spam/threat or legitimate using NLP and Machine Learning.")

# Load files
try:
    model, vectorizer = load_model()
    data = load_data()
except Exception as e:
    st.error("Error loading model or vectorizer.")
    st.write(e)
    st.stop()


# =========================
# Sidebar Navigation
# =========================
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Email Analyzer", "Data Explorer", "Model Info"]
)


# =========================
# Home Page
# =========================
if page == "Home":
    st.header("Project Overview")

    st.write("""
    This project is an Email Spam Detector that uses Natural Language Processing (NLP)
    to classify email text as spam/threat or legitimate.
    """)

    st.subheader("How to Use")
    st.write("""
    1. Go to the Email Analyzer page.
    2. Paste or type an email message.
    3. Click the Predict button.
    4. The app will show whether the email is spam/threat or legitimate.
    """)

    st.subheader("Project Features")
    st.write("""
    - Text preprocessing
    - TF-IDF feature extraction
    - Machine learning classification
    - Instant prediction
    - Confidence score
    - Dataset preview
    """)


# =========================
# Email Analyzer Page
# =========================
elif page == "Email Analyzer":
    st.header("Email Analyzer")

    email_text = st.text_area(
        "Enter email content here:",
        height=200,
        placeholder="Example: Congratulations! You have won a free prize. Click the link now..."
    )

    if st.button("Predict"):
        if email_text.strip() == "":
            st.warning("Please enter an email message first.")
        else:
            cleaned_email = clean_text(email_text)
            email_vector = vectorizer.transform([cleaned_email])
            prediction = model.predict(email_vector)[0]

            st.subheader("Prediction Result")

            # Confidence score if model supports predict_proba
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(email_vector)[0]
                confidence = max(probabilities) * 100
                st.metric("Confidence Score", f"{confidence:.2f}%")

            # Display prediction
            if str(prediction).lower() in ["spam", "threat", "malicious", "1"]:
                st.error(f"🚨 Prediction: {prediction}")
            else:
                st.success(f"✅ Prediction: {prediction}")

            st.write("Cleaned text used by model:")
            st.code(cleaned_email)


# =========================
# Data Explorer Page
# =========================
elif page == "Data Explorer":
    st.header("Data Explorer")

    if data is not None:
        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        st.subheader("Dataset Shape")
        st.write(f"Rows: {data.shape[0]}")
        st.write(f"Columns: {data.shape[1]}")

        st.subheader("Column Names")
        st.write(list(data.columns))

        if "label" in data.columns:
            st.subheader("Class Distribution")
            label_counts = data["label"].value_counts()
            st.bar_chart(label_counts)
    else:
        st.warning("Dataset file not found. Please check the file name inside the data folder.")


# =========================
# Model Info Page
# =========================
elif page == "Model Info":
    st.header("Model Information")

    st.write("""
    The model uses TF-IDF vectorization to convert email text into numerical features.
    A trained machine learning classifier then predicts whether the email is spam/threat or legitimate.
    """)

    st.subheader("Files Used")
    st.code("""
models/best_email_threat_model.pkl
models/tfidf_vectorizer.pkl
data/final_balanced_multilingual_email_dataset.csv
""")
