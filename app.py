import streamlit as st
from transformers import pipeline
import re

# Initialize the CodeBERT model for code analysis
@st.cache_resource
def load_codebert_pipeline():
    return pipeline("feature-extraction", model="microsoft/codebert-base")

# Function to analyze code
def analyze_code(code, model):
    suggestions = []

    # 1. Check for readability and comments
    if len(code.splitlines()) > 0:
        comment_lines = [line for line in code.splitlines() if line.strip().startswith("#")]
        comment_ratio = len(comment_lines) / len(code.splitlines())
        if comment_ratio < 0.2:
            suggestions.append("Consider adding more comments to improve readability.")

    # 2. Check adherence to PEP 8 style guide
    if re.search(r'[^ ]{81,}', code):
        suggestions.append("Code lines exceed 80 characters. Consider wrapping lines for better readability.")

    if not re.search(r'[\n]{2,}', code):
        suggestions.append("Code may not adhere to PEP 8 style guidelines. Add blank lines between functions and classes.")

    # 3. Highlight potential code smells or bugs
    embedding = model(code)[0]
    if len(embedding) < 50:
        suggestions.append("Code length too short for robust analysis.")
    
    return suggestions

# Streamlit App UI
st.title("Code Review Application")
st.write("Paste your code snippet below, and the app will analyze it for common issues and suggest improvements.")

code_input = st.text_area("Paste your code here:", height=300)
if st.button("Analyze Code"):
    if code_input.strip():
        model = load_codebert_pipeline()
        feedback = analyze_code(code_input, model)
        st.subheader("Suggestions:")
        if feedback:
            for suggestion in feedback:
                st.write(f"- {suggestion}")
        else:
            st.write("No issues found. Great job!")
    else:
        st.write("Please paste some code to analyze.")
