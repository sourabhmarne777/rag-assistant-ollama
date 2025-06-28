import streamlit as st
from rag_chain import run_rag_pipeline

st.title("🔍 RAG-based AI Assistant")

query = st.text_input("Enter your question")
urls_input = st.text_area("Enter URLs (one per line)", height=150)

if st.button("Get Answer"):
    urls = urls_input.strip().splitlines()
    with st.spinner("Processing..."):
        result = run_rag_pipeline(query, urls)
        st.success(result)
