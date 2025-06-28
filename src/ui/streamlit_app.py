import streamlit as st
import logging
from src.rag.pipeline import rag_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🧠 RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stats-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .answer-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .url-input {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .storage-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .storage-critical {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 RAG Assistant</h1>
        <p>Powered by Mistral 7B • Qdrant Cloud • LangChain</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        # Get stats
        stats = rag_pipeline.get_stats()
        storage_stats = rag_pipeline.qdrant_service.get_storage_stats()
        
        if stats['collection_info']:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin: 0;">Documents</h3>
                    <h2 style="margin: 0;">{}</h2>
                </div>
                """.format(stats['collection_info']['vectors_count']), unsafe_allow_html=True)
            
            with col2:
                status_color = "#28a745" if stats['collection_info']['status'] == "green" else "#ffc107"
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: {}; margin: 0;">Status</h3>
                    <h2 style="margin: 0; color: {};">●</h2>
                </div>
                """.format(status_color, status_color), unsafe_allow_html=True)
        else:
            st.error("❌ Qdrant Cloud connection failed")
        
        # Storage monitoring
        if storage_stats:
            st.markdown("### 💾 Storage Usage")
            
            # Progress bar
            progress_color = "normal"
            if storage_stats['status'] == 'critical':
                progress_color = "red"
            elif storage_stats['status'] == 'warning':
                progress_color = "orange"
            
            st.progress(storage_stats['usage_percent'] / 100)
            st.write(f"**{storage_stats['usage_percent']:.1f}%** used ({storage_stats['current_vectors']:,} / {storage_stats['limit']:,})")
            
            # Warning messages
            if storage_stats['status'] == 'critical':
                st.markdown("""
                <div class="storage-critical">
                    <strong>⚠️ CRITICAL:</strong> Storage >90% full!<br>
                    Consider cleanup or upgrade.
                </div>
                """, unsafe_allow_html=True)
            elif storage_stats['status'] == 'warning':
                st.markdown("""
                <div class="storage-warning">
                    <strong>⚠️ WARNING:</strong> Storage >80% full.<br>
                    Monitor usage closely.
                </div>
                """, unsafe_allow_html=True)
            
            # Cleanup options
            if storage_stats['status'] in ['warning', 'critical']:
                st.markdown("### 🧹 Cleanup Options")
                
                if st.button("🗑️ Reset All Vectors", type="secondary"):
                    if st.confirm("This will delete ALL stored vectors. Continue?"):
                        success = rag_pipeline.qdrant_service.reset_collection()
                        if success:
                            st.success("✅ All vectors deleted!")
                            st.rerun()
                        else:
                            st.error("❌ Reset failed!")
        
        st.markdown("---")
        
        st.markdown("""
        <div class="stats-card">
            <h4>💡 Tips</h4>
            <ul style="margin: 0; padding-left: 1rem;">
                <li>URLs are cached automatically</li>
                <li>Ask specific questions</li>
                <li>Provide relevant sources</li>
                <li>Monitor storage usage</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Ask Your Question")
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            height=120,
            placeholder="What would you like to know?",
            help="Ask any question related to the content from your URLs"
        )
        
        # URLs input
        st.header("🌐 Source URLs")
        urls_input = st.text_area(
            "Enter URLs (one per line):",
            height=180,
            placeholder="https://example.com/article1\nhttps://example.com/article2",
            help="Provide URLs containing relevant information"
        )
        
        # Process button
        if st.button("🚀 Get Answer", type="primary", use_container_width=True):
            if not question:
                st.error("Please enter a question!")
                return
            
            if not urls_input:
                st.error("Please provide at least one URL!")
                return
            
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
            
            if not urls:
                st.error("Please provide valid URLs!")
                return
            
            # Show processing status
            with st.spinner("🔄 Processing your request..."):
                try:
                    answer = rag_pipeline.query(question, urls)
                    
                    # Display results
                    st.markdown("## 💬 Answer")
                    st.markdown(f"""
                    <div class="answer-box">
                        {answer}
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logger.error(f"UI Error: {e}")

    with col2:
        st.header("📋 How It Works")
        st.markdown("""
        **🔄 Process:**
        
        1. **Web Scraping** - Extract content from URLs
        2. **Embedding** - Convert to vectors using nomic-embed-text
        3. **Storage** - Store in Qdrant Cloud with metadata
        4. **Search** - Find relevant content via cosine similarity
        5. **Generation** - Create answer using Mistral 7B
        
        **✨ Features:**
        - Smart URL caching
        - Context window management
        - Relevance scoring
        - Source attribution
        - Storage monitoring
        """)
        
        st.markdown("---")
        
        st.markdown("""
        **🎯 Best Practices:**
        - Use authoritative sources
        - Ask specific questions
        - Provide multiple relevant URLs
        - Monitor storage usage
        - Clean up old vectors when needed
        """)

if __name__ == "__main__":
    main()