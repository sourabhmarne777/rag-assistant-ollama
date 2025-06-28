import streamlit as st
import logging
from rag_pipeline import rag_pipeline, vector_store

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page config
st.set_page_config(
    page_title="🧠 RAG Assistant",
    page_icon="🧠",
    layout="wide"
)

# Compact CSS for better UX
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.2rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .answer-card {
        background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .answer-card h3 {
        color: #155724;
        margin: 0 0 1rem 0;
        font-size: 1.2rem;
    }
    .answer-text {
        color: #155724;
        font-size: 15px;
        line-height: 1.6;
        margin: 0;
        white-space: pre-wrap;
    }
    .storage-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .storage-good { border-left: 4px solid #28a745; }
    .storage-warning { border-left: 4px solid #ffc107; }
    .storage-critical { border-left: 4px solid #dc3545; }
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    .metric-item {
        text-align: center;
        flex: 1;
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 6px;
        margin: 0 0.2rem;
    }
    .metric-number {
        font-size: 1.3rem;
        font-weight: bold;
        color: #667eea;
        margin: 0;
    }
    .metric-label {
        color: #6c757d;
        font-size: 0.75rem;
        margin: 0;
    }
    /* Reduce Streamlit default spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }
    .stTextArea > div > div > textarea {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Compact header with tips
    st.markdown("""
    <div class="main-header">
        <h1>🧠 RAG Assistant</h1>
        <p>Ask questions about web content • Provide URLs with relevant info • Get AI-powered answers</p>
    </div>
    """, unsafe_allow_html=True)

    # Main content in columns
    col1, col2 = st.columns([2.5, 1])
    
    with col1:
        # Question input
        st.markdown("### 🔍 Your Question")
        question = st.text_area(
            "Question input",
            height=80,
            placeholder="What would you like to know? (e.g., 'What are the key features of Python 3.12?')",
            label_visibility="collapsed"
        )
        
        # URLs input
        st.markdown("### 🌐 Source URLs")
        urls_input = st.text_area(
            "URLs input",
            height=100,
            placeholder="Enter URLs (one per line):\nhttps://docs.python.org/3.12/whatsnew/3.12.html\nhttps://realpython.com/python312-new-features/",
            label_visibility="collapsed"
        )
        
        # Submit button
        if st.button("🚀 Get Answer", type="primary", use_container_width=True):
            if not question.strip():
                st.error("❌ Please enter a question!")
            elif not urls_input.strip():
                st.error("❌ Please provide at least one URL!")
            else:
                urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
                
                with st.spinner("🔄 Processing your request..."):
                    try:
                        answer = rag_pipeline.query(question, urls)
                        
                        st.markdown(f"""
                        <div class="answer-card">
                            <h3>💬 AI Answer</h3>
                            <div class="answer-text">{answer}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    with col2:
        # Storage monitoring
        st.markdown("### 💾 Storage Monitor")
        
        try:
            stats = rag_pipeline.get_storage_stats()
            
            # Storage card with metrics
            card_class = f"storage-card storage-{stats['status']}"
            
            st.markdown(f"""
            <div class="{card_class}">
                <div class="metric-row">
                    <div class="metric-item">
                        <p class="metric-number">{stats['usage_percent']:.1f}%</p>
                        <p class="metric-label">Used</p>
                    </div>
                    <div class="metric-item">
                        <p class="metric-number">{stats['current_vectors']:,}</p>
                        <p class="metric-label">Documents</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar
            progress_color = "normal"
            if stats['status'] == 'critical':
                progress_color = "red"
            elif stats['status'] == 'warning':
                progress_color = "orange"
            
            st.progress(stats['usage_percent'] / 100)
            
            # Status messages
            if stats['status'] == 'critical':
                st.error("⚠️ Storage >90% full!")
            elif stats['status'] == 'warning':
                st.warning("⚠️ Storage >80% full")
            else:
                st.success("✅ Storage OK")
            
            # Cleanup option for high usage
            if stats['status'] in ['warning', 'critical']:
                st.markdown("**🧹 Cleanup Options:**")
                if st.button("🗑️ Clear All Data", type="secondary", use_container_width=True):
                    if st.checkbox("⚠️ Confirm deletion"):
                        if vector_store.reset_collection():
                            st.success("✅ Data cleared!")
                            st.rerun()
                        else:
                            st.error("❌ Failed!")
            
        except Exception as e:
            st.error(f"❌ Storage stats unavailable: {str(e)}")

if __name__ == "__main__":
    main()