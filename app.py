import streamlit as st
import pandas as pd
import joblib
import re
import html
import os
from pathlib import Path
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide"
)

# =========================
# Create Images Folder
# =========================
os.makedirs("images", exist_ok=True)

# =========================
# Load External CSS
# =========================
def load_css(file_path):
    if Path(file_path).exists():
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("CSS file not found. App is running without custom styling.")

load_css("styles.css")

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
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

# =========================
# Load Dataset
# =========================
@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    csv_files = list(Path("data").glob("*.csv"))

    if len(csv_files) > 0:
        return pd.read_csv(csv_files[0])

    return None

# =========================
# Text Cleaning Function
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
# Spam Indicator Words
# =========================
spam_keywords = [
    "free", "win", "winner", "prize", "urgent", "click", "claim",
    "reward", "money", "cash", "offer", "limited", "account",
    "password", "verify", "login", "bank", "congratulations"
]

def find_spam_words(text):
    text_lower = text.lower()
    found_words = []

    for word in spam_keywords:
        if word in text_lower:
            found_words.append(word)

    return found_words

# =========================
# Load Files
# =========================
try:
    model, vectorizer = load_model()
    data = load_data()
except Exception as e:
    st.error("Error loading model, vectorizer, or dataset.")
    st.write(e)
    st.stop()

# =========================
# Top Navigation Bar
# =========================
if "page" not in st.session_state:
    st.session_state.page = "Home"

def set_page(page_name):
    st.session_state.page = page_name

st.markdown("""
<div class="top-navbar">
    <div class="top-brand">
        <div class="top-logo">📧</div>
        <div>
            <div class="top-title">Email Spam Detector</div>
            <div class="top-subtitle">AI-powered NLP email classification dashboard</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

nav1, nav2, nav3, nav4, nav5 = st.columns([1, 1.3, 1.2, 1.2, 1.1])

with nav1:
    if st.button("🏠 Home", use_container_width=True):
        set_page("Home")

with nav2:
    if st.button("🔍 Analyzer", use_container_width=True):
        set_page("Email Analyzer")

with nav3:
    if st.button("📊 Data", use_container_width=True):
        set_page("Data Explorer")

with nav4:
    if st.button("📈 Visuals", use_container_width=True):
        set_page("Visualizations")

with nav5:
    if st.button("🤖 Model", use_container_width=True):
        set_page("Model Info")

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

page = st.session_state.page

# =========================
# Home Page
# =========================
if page == "Home":
    st.markdown("""
    <div class="hero-box">
        <div class="hero-badge">NLP · Machine Learning · Email Security</div>
        <div class="hero-title">Detect suspicious emails instantly.</div>
        <div class="hero-subtitle">
            A modern NLP-powered dashboard that analyzes email content and predicts whether it is spam, threat, or legitimate using a trained machine learning model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card blue-line">
            <h3>🎯 Objective</h3>
            <p>Classify email messages using Natural Language Processing and Machine Learning.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card green-line">
            <h3>⚙️ NLP Pipeline</h3>
            <p>Clean text, transform it using TF-IDF, and pass it into a trained classifier.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card purple-line">
            <h3>🚀 Output</h3>
            <p>Get instant prediction, confidence score, spam indicator words, and visual insights.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card orange-line">
        <h3>How to Use This App</h3>
        <p>
        1. Click <b>Analyzer</b> at the top navigation bar.<br>
        2. Paste or type an email message.<br>
        3. Click the <b>Predict Email</b> button.<br>
        4. View the prediction result, confidence score, and spam indicator words.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Email Analyzer Page
# =========================
elif page == "Email Analyzer":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">🔍 Email Analyzer</div>
        <div class="page-header-subtitle">
            Paste an email message below and let the NLP model classify it instantly.
        </div>
    </div>
    """, unsafe_allow_html=True)

    email_text = st.text_area(
        "Enter email content:",
        height=210,
        placeholder="Example: Congratulations! You have won a free prize. Click here to claim now..."
    )

    if st.button("Predict Email", use_container_width=True):
        if email_text.strip() == "":
            st.warning("Please enter an email message first.")
        else:
            cleaned_email = clean_text(email_text)
            safe_cleaned_email = html.escape(cleaned_email)

            email_vector = vectorizer.transform([cleaned_email])
            prediction = model.predict(email_vector)[0]

            prediction_text = str(prediction).lower()

            st.subheader("Prediction Result")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(email_vector)[0]
                confidence = max(probabilities) * 100
                st.metric("Confidence Score", f"{confidence:.2f}%")

            if prediction_text in ["spam", "threat", "malicious", "phishing", "1"]:
                st.markdown(f"""
                <div class="danger-box">
                    🚨 Prediction: {prediction}
                </div>
                """, unsafe_allow_html=True)
            elif prediction_text in ["ham", "legitimate", "safe", "normal", "0"]:
                st.markdown(f"""
                <div class="success-box">
                    ✅ Prediction: {prediction}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                    ⚠️ Prediction: {prediction}
                </div>
                """, unsafe_allow_html=True)

            found_words = find_spam_words(email_text)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div class="info-card blue-line">
                    <h3>Spam Indicator Words</h3>
                """, unsafe_allow_html=True)

                if found_words:
                    badges = "".join([f"<span class='keyword-badge'>{word}</span>" for word in found_words])
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<div class='indicator-text'>No common spam indicator words found.</div>",
                        unsafe_allow_html=True
                    )

                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="info-card green-line">
                    <h3>Cleaned Text Preview</h3>
                """, unsafe_allow_html=True)

                st.markdown(
                    f"<div class='text-preview-box'>{safe_cleaned_email}</div>",
                    unsafe_allow_html=True
                )

                st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Data Explorer Page
# =========================
elif page == "Data Explorer":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">📊 Data Explorer</div>
        <div class="page-header-subtitle">
            Explore sample data, dataset statistics, and overall class distribution.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if data is None:
        st.error("Dataset not found. Please check your data folder.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Rows", data.shape[0])

    with col2:
        st.metric("Total Columns", data.shape[1])

    with col3:
        if "label" in data.columns:
            st.metric("Total Classes", data["label"].nunique())
        else:
            st.metric("Total Classes", "N/A")

    st.markdown("""
    <div class="info-card blue-line">
        <h3>Sample Data</h3>
        <p>The table below shows the first few rows of the email dataset using st.dataframe.</p>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(data.head(10), use_container_width=True)

    st.markdown("""
    <div class="info-card green-line">
        <h3>Dataset Statistics</h3>
        <p>This section summarizes column names and label counts in the dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    stat_col1, stat_col2 = st.columns(2)

    with stat_col1:
        st.subheader("Column Names")
        st.write(list(data.columns))

    with stat_col2:
        if "label" in data.columns:
            st.subheader("Label Counts")
            st.write(data["label"].value_counts())
        else:
            st.warning("No 'label' column found in the dataset.")

    st.markdown("""
    <div class="info-card purple-line">
        <h3>Data Distribution</h3>
        <p>This pie chart gives a quick overview of how the dataset is distributed across classes.</p>
    </div>
    """, unsafe_allow_html=True)

    if "label" in data.columns:
        left_space, chart_space, right_space = st.columns([1, 1.25, 1])

        with chart_space:
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            data["label"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            ax.set_title("Overall Email Class Distribution", fontsize=10)

            fig.savefig("images/label_distribution_pie.png", bbox_inches="tight", dpi=300)

            st.pyplot(fig)
            plt.close(fig)
    else:
        st.warning("Cannot show data distribution because the dataset has no 'label' column.")

# =========================
# Visualizations Page
# =========================
elif page == "Visualizations":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">📈 Visualizations</div>
        <div class="page-header-subtitle">
            Visual insights from labels, words, and email text patterns.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if data is None:
        st.error("Dataset not found. Please check your data folder.")
        st.stop()

    text_column = "clean_text" if "clean_text" in data.columns else "text"
    all_text = " ".join(data[text_column].dropna().astype(str))

    tab1, tab2, tab3 = st.tabs(["📊 Label Distribution", "☁️ Word Analysis", "📏 Text Patterns"])

    with tab1:
        st.subheader("Bar Chart of Label Distribution")

        if "label" in data.columns:
            left_space, chart_space, right_space = st.columns([0.6, 1.8, 0.6])

            with chart_space:
                fig, ax = plt.subplots(figsize=(5.8, 3.4))
                data["label"].value_counts().plot(kind="bar", ax=ax)
                ax.set_xlabel("Class Label")
                ax.set_ylabel("Number of Emails")
                ax.set_title("Email Label Distribution", fontsize=10)
                plt.xticks(rotation=0)

                fig.savefig("images/label_distribution_bar.png", bbox_inches="tight", dpi=300)

                st.pyplot(fig)
                plt.close(fig)
        else:
            st.warning("No 'label' column found in the dataset.")

    with tab2:
        st.subheader("Word Cloud of Most Common Words")

        if all_text.strip() != "":
            wordcloud = WordCloud(
                width=800,
                height=300,
                background_color="white"
            ).generate(all_text)

            left_space, chart_space, right_space = st.columns([0.4, 2.2, 0.4])

            with chart_space:
                fig, ax = plt.subplots(figsize=(7.2, 3.2))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")

                fig.savefig("images/wordcloud.png", bbox_inches="tight", dpi=300)

                st.pyplot(fig)
                plt.close(fig)
        else:
            st.warning("No text available to generate word cloud.")

        st.subheader("Top 20 Common Words")

        words = all_text.split()

        if len(words) > 0:
            word_freq = pd.Series(words).value_counts().head(20)

            left_space, chart_space, right_space = st.columns([0.4, 2.2, 0.4])

            with chart_space:
                fig, ax = plt.subplots(figsize=(7.0, 3.4))
                word_freq.plot(kind="bar", ax=ax)
                ax.set_xlabel("Words")
                ax.set_ylabel("Frequency")
                ax.set_title("Top 20 Most Common Words", fontsize=10)
                plt.xticks(rotation=45, ha="right")

                fig.savefig("images/top_common_words.png", bbox_inches="tight", dpi=300)

                st.pyplot(fig)
                plt.close(fig)
        else:
            st.warning("No words available to plot.")

    with tab3:
        st.subheader("Text Length Distribution")

        data["text_length"] = data[text_column].astype(str).apply(len)

        left_space, chart_space, right_space = st.columns([0.4, 2.2, 0.4])

        with chart_space:
            fig, ax = plt.subplots(figsize=(7.0, 3.4))
            ax.hist(data["text_length"], bins=30)
            ax.set_xlabel("Text Length")
            ax.set_ylabel("Number of Emails")
            ax.set_title("Distribution of Email Text Length", fontsize=10)

            fig.savefig("images/text_length_distribution.png", bbox_inches="tight", dpi=300)

            st.pyplot(fig)
            plt.close(fig)

        if "language" in data.columns:
            st.subheader("Language Distribution")

            left_space, chart_space, right_space = st.columns([0.4, 2.2, 0.4])

            with chart_space:
                fig, ax = plt.subplots(figsize=(7.0, 3.4))
                data["language"].value_counts().plot(kind="bar", ax=ax)
                ax.set_xlabel("Language")
                ax.set_ylabel("Number of Emails")
                ax.set_title("Email Language Distribution", fontsize=10)
                plt.xticks(rotation=45, ha="right")

                fig.savefig("images/language_distribution.png", bbox_inches="tight", dpi=300)

                st.pyplot(fig)
                plt.close(fig)

# =========================
# Model Info Page
# =========================
elif page == "Model Info":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">🤖 Model Information</div>
        <div class="page-header-subtitle">
            Understand how the trained NLP classification model works.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card blue-line">
            <h3>NLP Pipeline</h3>
            <p>
            The system cleans the input email, converts it into TF-IDF numerical features,
            and passes it into a trained machine learning classifier.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card green-line">
            <h3>Feature Extraction</h3>
            <p>
            TF-IDF is used to represent important words in the email while reducing the influence of common words.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card purple-line">
        <h3>Files Used</h3>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
models/best_email_threat_model.pkl
models/tfidf_vectorizer.pkl
data/final_balanced_multilingual_email_dataset.csv
""")

    st.markdown("""
    <div class="info-card orange-line">
        <h3>Pipeline Steps</h3>
        <p>
        1. User enters email text.<br>
        2. Text is cleaned and preprocessed.<br>
        3. TF-IDF vectorizer converts text into numerical features.<br>
        4. Machine learning model predicts the email class.<br>
        5. App displays prediction result, confidence score, and spam indicator words.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer-note">
    Email Spam Detector · NLP Final Project · Built with Streamlit
</div>
""", unsafe_allow_html=True)
