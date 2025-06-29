import streamlit as st
import logging
from rag_pipeline import rag_pipeline
from vector_store import vector_store
from file_processor import file_processor

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page config
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def load_css():
    """Load custom CSS styles"""
    with open("styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "context_files" not in st.session_state:
        st.session_state.context_files = []
    if "context_urls" not in st.session_state:
        st.session_state.context_urls = []
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = ""
    if "url_input_key" not in st.session_state:
        st.session_state.url_input_key = 0

def clear_chat():
    """Clear current chat session only"""
    st.session_state.messages = []
    st.session_state.conversation_context = ""
    st.success("🧹 Chat cleared successfully!")

def clear_all_context():
    """Clear all context including files, URLs, chat, and vector database"""
    # Clear session state
    st.session_state.context_files = []
    st.session_state.context_urls = []
    st.session_state.messages = []
    st.session_state.conversation_context = ""
    
    # Force URL input field to clear
    st.session_state.url_input_key += 1
    
    # Reset the RAG pipeline context tracking
    rag_pipeline.reset_context()
    
    # Clear the vector database completely
    try:
        vector_store.clear_all_data()
        st.success("🗑️ All context and data cleared successfully!")
    except Exception as e:
        st.error(f"❌ Error clearing vector database: {str(e)}")

def remove_file(file_index):
    """Remove a specific file from context"""
    if 0 <= file_index < len(st.session_state.context_files):
        removed_file = st.session_state.context_files.pop(file_index)
        st.success(f"🗑️ Removed: {removed_file['name']}")
        st.rerun()

def remove_url(url_index):
    """Remove a specific URL from context"""
    if 0 <= url_index < len(st.session_state.context_urls):
        removed_url = st.session_state.context_urls.pop(url_index)
        st.success(f"🗑️ Removed URL: {removed_url}")
        st.rerun()

def format_file_size(size_bytes):
    """Convert bytes to human readable format"""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 1)
    return f"{s} {size_names[i]}"

def manage_context_window(conversation_context, new_content, max_length=8000):
    """Manage conversation context to stay within model limits"""
    combined = conversation_context + "\n" + new_content
    
    if len(combined) <= max_length:
        return combined
    
    # If too long, keep only recent context
    lines = combined.split('\n')
    result = new_content
    
    # Add lines from the end until we approach the limit
    for line in reversed(lines[:-1]):
        if len(result + "\n" + line) < max_length:
            result = line + "\n" + result
        else:
            break
    
    return result

def main():
    """Main application function"""
    initialize_session_state()
    
    # Sidebar for context and controls
    with st.sidebar:
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clear Chat", use_container_width=True, help="Clear current conversation"):
                clear_chat()
                st.rerun()
        with col2:
            if st.button("🗑️ Reset All", use_container_width=True, help="Clear chat and all context"):
                clear_all_context()
                st.rerun()
        
        # File upload section
        st.markdown("### 📁 Upload Files")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['txt', 'pdf', 'docx'],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="Supports TXT, PDF, and DOCX files"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Check if file already exists
                file_exists = any(f['name'] == uploaded_file.name for f in st.session_state.context_files)
                if not file_exists:
                    processed_file = file_processor.process_file(uploaded_file)
                    if processed_file:
                        st.session_state.context_files.append(processed_file)
                        st.success(f"📄 Added: {uploaded_file.name}")
                        st.rerun()
        
        # Show uploaded files
        if st.session_state.context_files:
            st.markdown("**📄 Current Files:**")
            for i, file_info in enumerate(st.session_state.context_files):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"📄 **{file_info['name']}**")
                with col2:
                    if st.button("❌", key=f"remove_file_{i}", help="Remove file"):
                        remove_file(i)
        
        # URL input section
        st.markdown("### 🌐 Add URLs")
        
        # Single URL input with dynamic key
        single_url = st.text_input(
            "Enter URL",
            placeholder="https://example.com",
            label_visibility="collapsed",
            key=f"url_input_{st.session_state.url_input_key}"
        )
        
        if st.button("➕ Add URL", use_container_width=True):
            if single_url and single_url.strip():
                if single_url not in st.session_state.context_urls:
                    st.session_state.context_urls.append(single_url)
                    # Clear the input by incrementing the key
                    st.session_state.url_input_key += 1
                    st.success(f"🌐 Added URL: {single_url}")
                    st.rerun()
                else:
                    st.warning("⚠️ URL already added")
        
        # Show current URLs
        if st.session_state.context_urls:
            st.markdown("**🌐 Current URLs:**")
            for i, url in enumerate(st.session_state.context_urls):
                col1, col2 = st.columns([4, 1])
                with col1:
                    display_url = url if len(url) <= 30 else f"{url[:27]}..."
                    st.markdown(f"🌐 **{display_url}**")
                with col2:
                    if st.button("❌", key=f"remove_url_{i}", help="Remove URL"):
                        remove_url(i)
        
        # Storage analytics
        st.markdown("### 📊 Storage Analytics")
        try:
            stats = vector_store.get_storage_stats()
            used_size = stats['current_vectors'] * 768 * 4
            
            # Two cards in one row
            col1, col2 = st.columns(2)
            
            with col1:
                # Usage Card
                st.markdown(f"""
                <div class="storage-card usage-card">
                    <div class="storage-card-header">📈 Usage</div>
                    <div class="storage-card-value">{stats['usage_percent']:.1f}%</div>
                    <div class="storage-card-subtext">{stats['current_vectors']:,} vectors</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Status Card with proper colors
                if stats['status'] == 'critical':
                    status_class = "status-critical"
                    status_text = "Critical"
                    status_emoji = "🔴"
                elif stats['status'] == 'warning':
                    status_class = "status-warning"
                    status_text = "Warning"
                    status_emoji = "🟡"
                else:
                    status_class = "status-healthy"
                    status_text = "Healthy"
                    status_emoji = "🟢"
                
                st.markdown(f"""
                <div class="storage-card status-card {status_class}">
                    <div class="storage-card-header">{status_emoji} Status</div>
                    <div class="storage-card-value">{status_text}</div>
                    <div class="storage-card-subtext">{format_file_size(used_size)}</div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error("❌ Storage stats unavailable")
    
    # Main chat area
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG Chat Assistant</h1>
        <p>📁 Upload files or 🌐 URLs for context • 💬 Ask anything • 🧠 Get intelligent answers!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Context indicator
    if st.session_state.context_files or st.session_state.context_urls:
        total_context = len(st.session_state.context_files) + len(st.session_state.context_urls)
        st.info(f"🎯 Using {total_context} context source(s) for enhanced answers")
    else:
        st.info("💬 General AI chat mode - upload files or URLs for context-aware answers")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            try:
                # Check if we should use context
                has_context = bool(st.session_state.context_files or st.session_state.context_urls)
                
                if has_context:
                    # Prepare context from files
                    file_contexts = []
                    for file_info in st.session_state.context_files:
                        file_contexts.append(f"File: {file_info['name']}\n{file_info['content']}")
                    
                    # Get URLs
                    urls_to_process = st.session_state.context_urls.copy()
                    
                    # Get AI response with context
                    response = rag_pipeline.chat_query(
                        question=prompt,
                        urls=urls_to_process,
                        file_contexts=file_contexts,
                        conversation_context=st.session_state.conversation_context
                    )
                else:
                    # Normal AI chat without context
                    response = rag_pipeline.chat_query(
                        question=prompt,
                        urls=[],
                        file_contexts=[],
                        conversation_context=st.session_state.conversation_context
                    )
                
                # Update conversation context
                new_context = f"User: {prompt}\nAssistant: {response}"
                st.session_state.conversation_context = manage_context_window(
                    st.session_state.conversation_context, 
                    new_context
                )
                
                # Show response
                st.markdown(response)
                
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()