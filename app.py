import streamlit as st

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide"
)

st.title("📧 Email Spam Detector")
st.write("This is an NLP-based Streamlit application for detecting spam emails.")

st.header("Text Analyzer")

email_text = st.text_area("Paste your email message here:")

if st.button("Analyze"):
    if email_text.strip() == "":
        st.warning("Please enter an email message first.")
    else:
        st.success("Prediction feature will be added after the model is ready.")
