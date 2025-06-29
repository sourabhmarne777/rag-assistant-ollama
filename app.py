import streamlit as st
import os
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline
from src.document_processor import DocumentProcessor
from src.web_scraper import WebScraper
import uuid
import tempfile

# Load environment variables from .env file
load_dotenv()

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean, minimal styling with improved UX
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Hide Streamlit header */
    .stApp > header {
        background-color: transparent;
    }
    
    /* Hide empty elements and whitespace */
    .element-container:has(> .stMarkdown > div[data-testid="stMarkdownContainer"] > p:empty) {
        display: none;
    }
    
    /* Hide empty markdown containers */
    .stMarkdown > div[data-testid="stMarkdownContainer"]:empty {
        display: none;
    }
    
    /* Upload box styling */
    .upload-box {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background-color: #fafafa;
    }
    
    /* Chat container styling */
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        max-height: 500px;
        overflow-y: auto;
    }
    
    /* Message styling */
    .message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        color: #333333;
    }
    
    /* User message styling */
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
        color: #1565c0;
    }
    
    /* Assistant message styling */
    .assistant-message {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        margin-right: 2rem;
        color: #333333;
    }
    
    /* Source tag styling */
    .source-tag {
        background-color: #fff3e0;
        color: #f57c00;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    
    /* Status indicator styling */
    .status-indicator {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    /* Status indicator variants */
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
    
    /* Clear button styling */
    .clear-button {
        background-color: #ff5722;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        margin-top: 1rem;
    }
    
    /* Fix send button styling */
    .stButton > button {
        background-color: #1976d2;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1565c0;
        border: none;
    }
    
    .stButton > button:focus {
        background-color: #1565c0;
        border: none;
        box-shadow: 0 0 0 2px rgba(25, 118, 210, 0.3);
    }
    
    /* URL section styling */
    .url-section {
        margin-bottom: 1rem;
    }
    
    .url-button-container {
        margin-top: 0.5rem;
    }
    
    /* Loading indicator styling */
    .loading-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 1rem;
        background-color: #f0f8ff;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
    }
    
    .loading-spinner {
        width: 20px;
        height: 20px;
        border: 2px solid #e3f2fd;
        border-top: 2px solid #1976d2;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-text {
        color: #1976d2;
        font-weight: 500;
    }
    
    /* Hide file uploader label when files are uploaded */
    .stFileUploader > label {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables for the application"""
    # Generate unique session ID for document isolation
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Initialize RAG pipeline (main orchestrator)
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    
    # Initialize chat history for conversation tracking
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Track number of processed documents
    if 'documents_count' not in st.session_state:
        st.session_state.documents_count = 0
    
    # Track processed files to avoid duplicates
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    
    # Track processed URLs to avoid duplicates
    if 'processed_urls' not in st.session_state:
        st.session_state.processed_urls = set()
    
    # Current URL input state
    if 'current_url' not in st.session_state:
        st.session_state.current_url = ""
    
    # File uploader key for clearing uploaded files
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = 0
    
    # Chat input key for clearing input after submission
    if 'chat_input_key' not in st.session_state:
        st.session_state.chat_input_key = 0

def check_system_status():
    """Check if Ollama is running and accessible"""
    try:
        from src.llm_client import OllamaClient
        ollama_client = OllamaClient()
        return ollama_client.check_connection()
    except Exception as e:
        print(f"Error checking system status: {e}")
        return False

def show_loading_indicator(message: str):
    """Display a loading indicator with custom message"""
    st.markdown(f"""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <div class="loading-text">{message}</div>
    </div>
    """, unsafe_allow_html=True)

def process_files(uploaded_files):
    """Process uploaded files and convert them to searchable chunks"""
    if not uploaded_files:
        return
        
    processor = DocumentProcessor()
    total_chunks = 0
    
    # Filter out already processed files to avoid duplicates
    new_files = []
    for file in uploaded_files:
        # Create unique identifier for file (name + size)
        file_key = f"{file.name}_{file.size}"
        if file_key not in st.session_state.processed_files:
            new_files.append(file)
            st.session_state.processed_files.add(file_key)
    
    # Exit early if no new files to process
    if not new_files:
        return
    
    # Show loading indicator for user feedback
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        show_loading_indicator("Processing uploaded files and creating chunks...")
    
    # Create progress bar for visual feedback
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process each file individually
    for i, uploaded_file in enumerate(new_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # Save uploaded file temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Process the document into chunks
            chunks = processor.process_document(tmp_file_path, uploaded_file.name)
            
            # Store chunks in vector database with session isolation
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
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(new_files))
    
    # Clear loading indicators
    loading_placeholder.empty()
    status_text.empty()
    progress_bar.empty()
    
    # Update document count and show success message
    st.session_state.documents_count += len(new_files)
    if total_chunks > 0:
        st.success(f"✅ Processed {len(new_files)} files ({total_chunks} chunks)")

def process_url(url):
    """Process web URL and extract content for RAG"""
    # Check if URL already processed to avoid duplicates
    if url in st.session_state.processed_urls:
        st.info("This URL has already been processed.")
        return
    
    # Show loading indicator for user feedback
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        show_loading_indicator("Scraping web content and creating chunks...")
    
    try:
        # Initialize web scraper
        scraper = WebScraper()
        
        # Scrape content from URL
        content = scraper.scrape_url(url)
        
        if content:
            # Process scraped content into chunks
            processor = DocumentProcessor()
            chunks = processor.process_text(content, url)
            
            # Store chunks in vector database with session isolation
            success = st.session_state.rag_pipeline.add_documents(
                chunks,
                source_type="web",
                source_name=url,
                session_id=st.session_state.session_id
            )
            
            # Clear loading indicator
            loading_placeholder.empty()
            
            if success:
                # Update state and show success
                st.session_state.documents_count += 1
                st.session_state.processed_urls.add(url)
                st.success(f"✅ Processed content from URL ({len(chunks)} chunks)")
                # Clear the URL input after successful processing
                st.session_state.current_url = ""
                # Force rerun to clear the URL input field
                st.rerun()
            else:
                st.warning("⚠️ Issues processing URL content - check console for details")
        else:
            loading_placeholder.empty()
            st.error("❌ Failed to scrape content from the URL")
            
    except Exception as e:
        loading_placeholder.empty()
        st.error(f"❌ Error: {str(e)}")

def handle_chat_input(user_question):
    """Handle chat input and generate RAG response"""
    # Check if documents are available
    if st.session_state.documents_count == 0:
        st.warning("⚠️ Please add some documents or web content first!")
        return
    
    # Add user message to chat history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_question
    })
    
    # Generate response using RAG pipeline
    with st.spinner("Thinking..."):
        try:
            # Query the RAG system with session isolation
            response, sources = st.session_state.rag_pipeline.query(
                user_question,
                session_id=st.session_state.session_id
            )
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'sources': sources
            })
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def clear_all_data():
    """Clear all data including chat, documents, URLs, and uploaded files"""
    # Reset all session state variables
    st.session_state.chat_history = []
    st.session_state.documents_count = 0
    st.session_state.processed_files = set()
    st.session_state.processed_urls = set()
    st.session_state.current_url = ""
    
    # Generate new session ID for complete isolation
    st.session_state.session_id = str(uuid.uuid4())
    
    # Increment file uploader key to clear uploaded files
    st.session_state.file_uploader_key += 1
    
    # Increment chat input key to clear chat input
    st.session_state.chat_input_key += 1
    
    # Clear from vector store (session-based cleanup)
    try:
        st.session_state.rag_pipeline.clear_session(st.session_state.session_id)
    except Exception as e:
        print(f"Error clearing session data: {e}")

def main():
    """Main application function"""
    # Initialize session state variables
    initialize_session_state()
    
    # Application header
    st.title("🤖 RAG Assistant")
    st.markdown("Upload documents or add web content, then chat with your data")
    
    # Check system status (Ollama connection)
    ollama_running = check_system_status()
    
    # Display system status indicator
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
    
    # Content input section
    st.markdown("### Add Content")
    
    # Create two columns for file upload and URL input
    col1, col2 = st.columns(2)
    
    # File upload column
    with col1:
        st.markdown("**Upload Files**")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'csv'],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"file_uploader_{st.session_state.file_uploader_key}"
        )
        
        # Auto-process files when uploaded
        if uploaded_files:
            process_files(uploaded_files)
    
    # URL input column
    with col2:
        st.markdown("**Add Web Content**")
        
        # URL input section with button below
        url = st.text_input(
            "Enter URL",
            placeholder="https://example.com/article",
            label_visibility="collapsed",
            key="url_input",
            value=st.session_state.current_url
        )
        
        # Add URL button below the text input
        add_url_clicked = st.button("🌐 Add URL", use_container_width=True, type="primary")
        
        # Process URL when button is clicked
        if add_url_clicked and url and url.startswith(('http://', 'https://')):
            process_url(url)
        elif add_url_clicked and url:
            st.error("Please enter a valid URL starting with http:// or https://")
        elif add_url_clicked:
            st.error("Please enter a URL")
    
    # Chat section
    st.markdown("### Chat")
    
    # Clear All button at the top of chat section
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🗑️ Clear All", use_container_width=True, help="Clear chat, documents, URLs, and uploaded files"):
            clear_all_data()
            st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                # Display user message
                st.markdown(f"""
                <div class="message user-message">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Display assistant message with sources
                sources_html = ""
                if message.get('sources'):
                    sources_html = "<br>" + "".join([f'<span class="source-tag">{source}</span>' for source in message['sources']])
                
                st.markdown(f"""
                <div class="message assistant-message">
                    <strong>Assistant:</strong> {message['content']}{sources_html}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input with Enter key support
    # Create form for Enter key support
    with st.form(key=f"chat_form_{st.session_state.chat_input_key}", clear_on_submit=True):
        user_question = st.text_input(
            "Ask a question about your content:",
            placeholder="What is the main topic discussed?",
            key=f"chat_input_{st.session_state.chat_input_key}"
        )
        
        # Send button with proper styling
        send_clicked = st.form_submit_button("💬 Send", use_container_width=True, type="primary")
    
    # Handle form submission
    if send_clicked and user_question.strip():
        handle_chat_input(user_question)
        # Clear the input by incrementing the key
        st.session_state.chat_input_key += 1
        st.rerun()

# Application entry point
if __name__ == "__main__":
    main()