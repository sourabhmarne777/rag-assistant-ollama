import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
from typing import List, Dict, Any
import asyncio
from datetime import datetime

# Import our custom modules
from src.rag_system import RAGSystem
from src.document_processor import DocumentProcessor
from src.web_scraper import WebScraper
from src.chat_interface import ChatInterface

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .sidebar-section {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .status-info {
        color: #17a2b8;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = []
    
    if 'urls_processed' not in st.session_state:
        st.session_state.urls_processed = []

def main():
    initialize_session_state()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG Assistant Chatbot</h1>
        <p>Upload documents, add URLs, and chat with your knowledge base!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for document management
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("📚 Knowledge Base")
        
        # Document upload section
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files to add to your knowledge base"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                process_uploaded_documents(uploaded_files)
        
        st.divider()
        
        # URL input section
        st.subheader("🌐 Add Web Content")
        url_input = st.text_input(
            "Enter URL",
            placeholder="https://example.com/article",
            help="Enter a URL to scrape and add to your knowledge base"
        )
        
        if url_input:
            if st.button("Process URL", type="primary"):
                process_url(url_input)
        
        st.divider()
        
        # Knowledge base status
        st.subheader("📊 Knowledge Base Status")
        display_knowledge_base_status()
        
        # Clear knowledge base
        if st.button("🗑️ Clear Knowledge Base", type="secondary"):
            clear_knowledge_base()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Chat history display
        chat_container = st.container()
        with chat_container:
            display_chat_history()
        
        # Chat input
        user_input = st.chat_input("Ask me anything about your documents...")
        
        if user_input:
            handle_user_input(user_input)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("🔧 Settings")
        
        # Model settings
        with st.expander("Model Configuration"):
            ollama_model = st.selectbox(
                "Ollama Model",
                ["mistral", "llama2", "codellama", "neural-chat"],
                index=0
            )
            
            embedding_model = st.selectbox(
                "Embedding Model",
                ["nomic-embed-text", "all-MiniLM-L6-v2"],
                index=0
            )
            
            if st.button("Update Models"):
                update_model_settings(ollama_model, embedding_model)
        
        # RAG settings
        with st.expander("RAG Configuration"):
            similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Minimum similarity score for relevant documents"
            )
            
            max_results = st.slider(
                "Max Results",
                min_value=1,
                max_value=10,
                value=5,
                help="Maximum number of relevant documents to retrieve"
            )
            
            if st.button("Update RAG Settings"):
                update_rag_settings(similarity_threshold, max_results)

def process_uploaded_documents(uploaded_files):
    """Process uploaded documents and add to knowledge base"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        processor = DocumentProcessor()
        total_files = len(uploaded_files)
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Process document
                chunks = processor.process_document(tmp_file_path, uploaded_file.name)
                
                # Add to vector store
                st.session_state.rag_system.add_documents(chunks, uploaded_file.name)
                
                # Update processed documents list
                st.session_state.documents_processed.append({
                    'name': uploaded_file.name,
                    'chunks': len(chunks),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                progress_bar.progress((i + 1) / total_files)
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        status_text.markdown('<p class="status-success">✅ All documents processed successfully!</p>', unsafe_allow_html=True)
        st.rerun()
        
    except Exception as e:
        status_text.markdown(f'<p class="status-error">❌ Error processing documents: {str(e)}</p>', unsafe_allow_html=True)

def process_url(url):
    """Process URL and add content to knowledge base"""
    status_text = st.empty()
    
    try:
        status_text.text("Scraping URL content...")
        
        scraper = WebScraper()
        content = scraper.scrape_url(url)
        
        if content:
            # Process the scraped content
            processor = DocumentProcessor()
            chunks = processor.process_text(content, url)
            
            # Add to vector store
            st.session_state.rag_system.add_documents(chunks, url)
            
            # Update processed URLs list
            st.session_state.urls_processed.append({
                'url': url,
                'chunks': len(chunks),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            status_text.markdown('<p class="status-success">✅ URL content processed successfully!</p>', unsafe_allow_html=True)
            st.rerun()
        else:
            status_text.markdown('<p class="status-error">❌ Failed to scrape URL content</p>', unsafe_allow_html=True)
            
    except Exception as e:
        status_text.markdown(f'<p class="status-error">❌ Error processing URL: {str(e)}</p>', unsafe_allow_html=True)

def display_knowledge_base_status():
    """Display current knowledge base status"""
    total_docs = len(st.session_state.documents_processed)
    total_urls = len(st.session_state.urls_processed)
    total_chunks = sum(doc['chunks'] for doc in st.session_state.documents_processed) + \
                   sum(url['chunks'] for url in st.session_state.urls_processed)
    
    st.metric("Documents", total_docs)
    st.metric("URLs", total_urls)
    st.metric("Total Chunks", total_chunks)
    
    if st.session_state.documents_processed:
        st.write("**Recent Documents:**")
        for doc in st.session_state.documents_processed[-3:]:
            st.write(f"• {doc['name']} ({doc['chunks']} chunks)")
    
    if st.session_state.urls_processed:
        st.write("**Recent URLs:**")
        for url in st.session_state.urls_processed[-3:]:
            st.write(f"• {url['url'][:50]}... ({url['chunks']} chunks)")

def display_chat_history():
    """Display chat history"""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.write(f"• **{source['source']}** (Score: {source['score']:.3f})")
                        st.write(f"  _{source['content'][:200]}..._")

def handle_user_input(user_input):
    """Handle user input and generate response"""
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response, sources = st.session_state.rag_system.query(user_input)
                
                st.markdown(response)
                
                # Add assistant message to chat history
                assistant_message = {
                    "role": "assistant", 
                    "content": response,
                    "sources": sources
                }
                st.session_state.chat_history.append(assistant_message)
                
                # Show sources
                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.write(f"• **{source['source']}** (Score: {source['score']:.3f})")
                            st.write(f"  _{source['content'][:200]}..._")
                
            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_message)
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})

def clear_knowledge_base():
    """Clear the knowledge base"""
    try:
        st.session_state.rag_system.clear_collection()
        st.session_state.documents_processed = []
        st.session_state.urls_processed = []
        st.success("Knowledge base cleared successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Error clearing knowledge base: {str(e)}")

def update_model_settings(ollama_model, embedding_model):
    """Update model settings"""
    try:
        st.session_state.rag_system.update_models(ollama_model, embedding_model)
        st.success("Model settings updated successfully!")
    except Exception as e:
        st.error(f"Error updating model settings: {str(e)}")

def update_rag_settings(similarity_threshold, max_results):
    """Update RAG settings"""
    try:
        st.session_state.rag_system.update_settings(similarity_threshold, max_results)
        st.success("RAG settings updated successfully!")
    except Exception as e:
        st.error(f"Error updating RAG settings: {str(e)}")

if __name__ == "__main__":
    main()