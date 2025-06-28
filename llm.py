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
        """Generate answer using context and question"""
        try:
            prompt = f"""Based on the following context, answer the question clearly and concisely.

Context:
{context}

Question: {question}

Answer:"""
            
            # Ensure prompt fits in context window
            if len(prompt) > MAX_TEXT_LENGTH:
                prompt = prompt[:MAX_TEXT_LENGTH]
            
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Sorry, I couldn't generate an answer due to an error."

llm_service = LLMService()