from langchain_ollama import OllamaLLM
from settings import OLLAMA_MODEL, OLLAMA_BASE_URL, MAX_TEXT_LENGTH
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1
        )

    def generate_answer(self, context, question):
        """Generate answer using context and question with better prompt handling"""
        try:
            if context.strip():
                # RAG mode with context
                prompt = f"""You are a helpful AI assistant. Based on the provided context, answer the user's question clearly and accurately.

Context:
{context}

Question: {question}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain relevant information, say so clearly
- Be concise but comprehensive
- Use a conversational tone

Answer:"""
            else:
                # General conversation mode
                prompt = f"""You are a helpful AI assistant. Answer the user's question in a conversational and helpful manner.

Question: {question}

Answer:"""
            
            # Ensure prompt fits in context window
            if len(prompt) > MAX_TEXT_LENGTH:
                # Truncate context while keeping the question and instructions
                context_limit = MAX_TEXT_LENGTH - 500  # Reserve space for question and instructions
                if context.strip():
                    truncated_context = context[:context_limit] + "...[truncated]"
                    prompt = f"""You are a helpful AI assistant. Based on the provided context, answer the user's question clearly and accurately.

Context:
{truncated_context}

Question: {question}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain relevant information, say so clearly
- Be concise but comprehensive
- Use a conversational tone

Answer:"""
                else:
                    prompt = f"""You are a helpful AI assistant. Answer the user's question in a conversational and helpful manner.

Question: {question}

Answer:"""
            
            response = self.llm.invoke(prompt)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I'm sorry, I encountered an error while generating a response. Please try again."

llm_service = LLMService()