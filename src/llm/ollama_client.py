from langchain_ollama import OllamaLLM
from config.settings import OLLAMA_MODEL, OLLAMA_BASE_URL, MAX_TEXT_LENGTH
import logging

logger = logging.getLogger(__name__)

class OllamaService:
    def __init__(self):
        try:
            self.llm = OllamaLLM(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.1,
                top_p=0.9
            )
            logger.info(f"Initialized Ollama with model: {OLLAMA_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            raise

    def generate_answer(self, prompt):
        """Generate answer using Ollama LLM"""
        try:
            # Ensure prompt fits within context window
            if len(prompt) > MAX_TEXT_LENGTH:
                prompt = prompt[:MAX_TEXT_LENGTH]
                logger.warning("Prompt truncated to fit context window")
            
            response = self.llm.invoke(prompt)
            logger.info("Generated response from Ollama")
            return response
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Sorry, I couldn't generate an answer due to an error."

    def create_rag_prompt(self, context, query):
        """Create a well-structured prompt for RAG with context window management"""
        base_prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context. If the context doesn't contain enough information to answer the question, say so clearly.

Question: {query}

Answer: Provide a clear, concise answer based on the context above. If the context is insufficient, explain what information is missing."""
        
        # Calculate available space for context
        available_space = MAX_TEXT_LENGTH - len(base_prompt) - 100  # Buffer
        
        if len(context) > available_space:
            context = context[:available_space]
            logger.warning("Context truncated to fit within context window")
        
        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}

Question: {query}

Answer: Provide a clear, concise answer based on the context above. If the context is insufficient, explain what information is missing."""
        
        return prompt

# Global instance
ollama_service = OllamaService()