import streamlit as st
import os
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline
from src.document_processor import DocumentProcessor
from src.web_scraper import WebScraper
import uuid
import tempfile

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean, minimal styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    .upload-box {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background-color: #fafafa;
    }
    
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        max-height: 500px;
        overflow-y: auto;
    }
    
    .message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        color: #333333;
    }
    
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
        color: #1565c0;
    }
    
    .assistant-message {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        margin-right: 2rem;
        color: #333333;
    }
    
    .source-tag {
        background-color: #fff3e0;
        color: #f57c00;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .status-indicator {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .status-ready {
        background-color: #e8f5e8;
        color: #2e7d32;
    }
    
    .status-empty {
        background-color: #fff3e0;
        color: #f57c00;
    }
    
    .status-warning {
        background-color: #ffebee;
        color: #c62828;
    }
    
    .clear-button {
        background-color: #ff5722;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'documents_count' not in st.session_state:
        st.session_state.documents_count = 0
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    
    if 'processed_urls' not in st.session_state:
        st.session_state.processed_urls = set()

def check_system_status():
    """Check if Ollama is running"""
    try:
        from src.llm_client import OllamaClient
        ollama_client = OllamaClient()
        return ollama_client.check_connection()
    except:
        return False

def process_files(uploaded_files):
    """Process uploaded files"""
    processor = DocumentProcessor()
    total_chunks = 0
    
    # Filter out already processed files
    new_files = []
    for file in uploaded_files:
        file_key = f"{file.name}_{file.size}"
        if file_key not in st.session_state.processed_files:
            new_files.append(file)
            st.session_state.processed_files.add(file_key)
    
    if not new_files:
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(new_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Process the document
            chunks = processor.process_document(tmp_file_path, uploaded_file.name)
            
            # Store in vector database
            success = st.session_state.rag_pipeline.add_documents(
                chunks, 
                source_type="document",
                source_name=uploaded_file.name,
                session_id=st.session_state.session_id
            )
            
            if success:
                total_chunks += len(chunks)
            else:
                st.warning(f"⚠️ Issues processing {uploaded_file.name} - check console for details")
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
        
        # Update progress
        progress_bar.progress((i + 1) / len(new_files))
    
    status_text.empty()
    progress_bar.empty()
    
    st.session_state.documents_count += len(new_files)
    if total_chunks > 0:
        st.success(f"✅ Processed {len(new_files)} files ({total_chunks} chunks)")

def process_url(url):
    """Process web URL"""
    if url in st.session_state.processed_urls:
        return
    
    try:
        scraper = WebScraper()
        
        with st.spinner("Scraping content..."):
            content = scraper.scrape_url(url)
        
        if content:
            processor = DocumentProcessor()
            chunks = processor.process_text(content, url)
            
            # Store in vector database
            success = st.session_state.rag_pipeline.add_documents(
                chunks,
                source_type="web",
                source_name=url,
                session_id=st.session_state.session_id
            )
            
            if success:
                st.session_state.documents_count += 1
                st.session_state.processed_urls.add(url)
                st.success(f"✅ Processed content from URL ({len(chunks)} chunks)")
            else:
                st.warning("⚠️ Issues processing URL content - check console for details")
        else:
            st.error("❌ Failed to scrape content from the URL")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def handle_chat_input(user_question):
    """Handle chat input and auto-process if needed"""
    if st.session_state.documents_count == 0:
        st.warning("⚠️ Please add some documents or web content first!")
        return
    
    # Add user message
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_question
    })
    
    with st.spinner("Thinking..."):
        try:
            # Get response
            response, sources = st.session_state.rag_pipeline.query(
                user_question,
                session_id=st.session_state.session_id
            )
            
            # Add assistant response
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'sources': sources
            })
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def clear_all_data():
    """Clear all data including chat, documents, and URLs"""
    st.session_state.chat_history = []
    st.session_state.documents_count = 0
    st.session_state.processed_files = set()
    st.session_state.processed_urls = set()
    st.session_state.session_id = str(uuid.uuid4())
    # Clear from vector store
    try:
        st.session_state.rag_pipeline.clear_session(st.session_state.session_id)
    except:
        pass

def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.title("🤖 RAG Assistant")
    st.markdown("Upload documents or add web content, then chat with your data")
    
    # System status check
    ollama_running = check_system_status()
    
    # Status indicator
    if not ollama_running:
        st.markdown("""
        <div class="status-indicator status-warning">
            ⚠️ Ollama not detected - Chat responses will be limited. Please run 'ollama serve'
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.documents_count > 0:
        st.markdown(f"""
        <div class="status-indicator status-ready">
            📚 Ready to chat • {st.session_state.documents_count} sources loaded
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-indicator status-empty">
            📁 Add some documents or web content to get started
        </div>
        """, unsafe_allow_html=True)
    
    # Input section
    st.markdown("### Add Content")
    
    # Create two columns for file upload and URL input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Upload Files**")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'csv'],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        # Auto-process files when uploaded
        if uploaded_files:
            process_files(uploaded_files)
    
    with col2:
        st.markdown("**Add Web Content**")
        url = st.text_input(
            "Enter URL",
            placeholder="https://example.com/article",
            label_visibility="collapsed",
            key="url_input"
        )
        
        # Auto-process URL when entered and valid
        if url and url.startswith(('http://', 'https://')):
            process_url(url)
    
    # Chat section
    st.markdown("### Chat")
    
    # Clear All button at the top of chat section
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🗑️ Clear All", use_container_width=True, help="Clear chat, documents, and URLs"):
            clear_all_data()
            st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="message user-message">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                sources_html = ""
                if message.get('sources'):
                    sources_html = "<br>" + "".join([f'<span class="source-tag">{source}</span>' for source in message['sources']])
                
                st.markdown(f"""
                <div class="message assistant-message">
                    <strong>Assistant:</strong> {message['content']}{sources_html}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    if 'chat_input_key' not in st.session_state:
        st.session_state.chat_input_key = 0
    
    user_question = st.text_input(
        "Ask a question about your content:",
        placeholder="What is the main topic discussed?",
        key=f"chat_input_{st.session_state.chat_input_key}"
    )
    
    # Send button
    if st.button("💬 Send", use_container_width=True):
        if user_question.strip():
            handle_chat_input(user_question)
            # Clear the input by incrementing the key
            st.session_state.chat_input_key += 1
            st.rerun()

if __name__ == "__main__":
    main()