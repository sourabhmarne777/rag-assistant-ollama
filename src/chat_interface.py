import streamlit as st
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatInterface:
    """Chat interface utilities for Streamlit"""
    
    def __init__(self):
        self.message_types = {
            'user': '👤',
            'assistant': '🤖',
            'system': '⚙️',
            'error': '❌'
        }
    
    def display_message(self, message: Dict[str, Any], show_sources: bool = True):
        """Display a single chat message"""
        try:
            role = message.get('role', 'user')
            content = message.get('content', '')
            sources = message.get('sources', [])
            
            # Get appropriate icon
            icon = self.message_types.get(role, '💬')
            
            with st.chat_message(role, avatar=icon):
                # Display main content
                st.markdown(content)
                
                # Display sources if available and requested
                if show_sources and sources and role == 'assistant':
                    self._display_sources(sources)
                    
        except Exception as e:
            logger.error(f"Error displaying message: {str(e)}")
            st.error(f"Error displaying message: {str(e)}")
    
    def _display_sources(self, sources: List[Dict[str, Any]]):
        """Display source information"""
        try:
            with st.expander("📚 Sources", expanded=False):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**Source {i}: {source.get('source', 'Unknown')}**")
                    
                    # Show relevance score if available
                    score = source.get('score', 0.0)
                    if score > 0:
                        st.markdown(f"*Relevance Score: {score:.3f}*")
                    
                    # Show content preview
                    content = source.get('content', '')
                    if content:
                        preview = content[:300] + "..." if len(content) > 300 else content
                        st.markdown(f"*Preview:* {preview}")
                    
                    # Show page number if available
                    page = source.get('page', 0)
                    if page > 0:
                        st.markdown(f"*Page: {page}*")
                    
                    if i < len(sources):
                        st.divider()
                        
        except Exception as e:
            logger.error(f"Error displaying sources: {str(e)}")
            st.error("Error displaying sources")
    
    def display_chat_history(self, chat_history: List[Dict[str, Any]], show_sources: bool = True):
        """Display entire chat history"""
        try:
            for message in chat_history:
                self.display_message(message, show_sources)
                
        except Exception as e:
            logger.error(f"Error displaying chat history: {str(e)}")
            st.error(f"Error displaying chat history: {str(e)}")
    
    def create_message(self, role: str, content: str, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a properly formatted message"""
        message = {
            'role': role,
            'content': content,
            'timestamp': st.session_state.get('current_time', '')
        }
        
        if sources:
            message['sources'] = sources
        
        return message
    
    def format_response_with_context(self, response: str, sources: List[Dict[str, Any]]) -> str:
        """Format response to include context information"""
        try:
            if not sources:
                return response
            
            # Add context information to response
            context_info = f"\n\n*This response is based on {len(sources)} relevant document(s) from your knowledge base.*"
            
            return response + context_info
            
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return response
    
    def get_chat_statistics(self, chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the chat history"""
        try:
            stats = {
                'total_messages': len(chat_history),
                'user_messages': 0,
                'assistant_messages': 0,
                'messages_with_sources': 0,
                'total_sources_referenced': 0
            }
            
            for message in chat_history:
                role = message.get('role', '')
                
                if role == 'user':
                    stats['user_messages'] += 1
                elif role == 'assistant':
                    stats['assistant_messages'] += 1
                    
                    sources = message.get('sources', [])
                    if sources:
                        stats['messages_with_sources'] += 1
                        stats['total_sources_referenced'] += len(sources)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating chat statistics: {str(e)}")
            return {}
    
    def export_chat_history(self, chat_history: List[Dict[str, Any]]) -> str:
        """Export chat history as formatted text"""
        try:
            export_text = "# Chat History Export\n\n"
            
            for i, message in enumerate(chat_history, 1):
                role = message.get('role', 'unknown')
                content = message.get('content', '')
                timestamp = message.get('timestamp', '')
                
                export_text += f"## Message {i} - {role.title()}\n"
                if timestamp:
                    export_text += f"*Time: {timestamp}*\n\n"
                
                export_text += f"{content}\n\n"
                
                # Add sources if available
                sources = message.get('sources', [])
                if sources:
                    export_text += "### Sources:\n"
                    for j, source in enumerate(sources, 1):
                        export_text += f"{j}. **{source.get('source', 'Unknown')}**"
                        score = source.get('score', 0.0)
                        if score > 0:
                            export_text += f" (Score: {score:.3f})"
                        export_text += "\n"
                    export_text += "\n"
                
                export_text += "---\n\n"
            
            return export_text
            
        except Exception as e:
            logger.error(f"Error exporting chat history: {str(e)}")
            return "Error exporting chat history"
    
    def clear_chat_history(self):
        """Clear the chat history from session state"""
        try:
            if 'chat_history' in st.session_state:
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
                st.rerun()
            
        except Exception as e:
            logger.error(f"Error clearing chat history: {str(e)}")
            st.error("Error clearing chat history")
    
    def suggest_questions(self, context_available: bool = False) -> List[str]:
        """Suggest relevant questions based on context"""
        if context_available:
            return [
                "What are the main topics covered in the uploaded documents?",
                "Can you summarize the key points from the documents?",
                "What specific information can you find about [topic]?",
                "Are there any important dates or numbers mentioned?",
                "What conclusions or recommendations are made?"
            ]
        else:
            return [
                "Hello! How can I help you today?",
                "What would you like to know?",
                "Please upload some documents so I can help you with specific information.",
                "You can ask me general questions or upload documents for context-specific answers."
            ]