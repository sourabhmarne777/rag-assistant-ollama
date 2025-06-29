import os
import requests
from typing import Optional

class OllamaClient:
    """
    Client for interacting with Ollama LLM for response generation.
    
    This class handles:
    - Connection to local Ollama server
    - RAG prompt construction
    - Response generation with context
    - Error handling and fallback responses
    """
    
    def __init__(self):
        """Initialize Ollama client with configuration from environment"""
        # Ollama server configuration
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "mistral")
        self.api_url = f"{self.base_url}/api/generate"
    
    def generate_response(self, question: str, context: str) -> str:
        """
        Generate response using Ollama with RAG context.
        
        Args:
            question: User's question
            context: Retrieved context from vector search
            
        Returns:
            Generated response text
        """
        try:
            # Check if Ollama is available before proceeding
            if not self.check_connection():
                return self._get_fallback_response()
            
            # Create a comprehensive RAG prompt
            prompt = self._create_rag_prompt(question, context)
            
            # Prepare request payload with model parameters
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,  # Get complete response at once
                "options": {
                    "temperature": 0.7,    # Balance creativity and consistency
                    "top_p": 0.9,         # Nucleus sampling for quality
                    "max_tokens": 1000    # Limit response length
                }
            }
            
            # Make request to Ollama API
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=60  # Allow time for model inference
            )
            
            # Handle successful response
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "Sorry, I couldn't generate a response.")
            else:
                return f"Error: Ollama API returned status code {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {str(e)}. Please make sure Ollama is running with 'ollama serve'."
        except Exception as e:
            return f"Unexpected error: {str(e)}"
    
    def _create_rag_prompt(self, question: str, context: str) -> str:
        """
        Create a well-structured RAG prompt for better responses.
        
        Args:
            question: User's question
            context: Retrieved context from documents
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context. 
Use the following context to answer the user's question. If the answer cannot be found in the context, 
say so clearly and don't make up information.

Context:
{context}

Question: {question}

Answer: Based on the provided context, """
        
        return prompt
    
    def _get_fallback_response(self) -> str:
        """
        Get fallback response when Ollama is not available.
        
        Returns:
            Informative error message with setup instructions
        """
        return """I'm sorry, but I can't connect to the local AI model (Ollama). 

Please make sure:
1. Ollama is installed and running
2. Run 'ollama serve' in your terminal
3. The models are downloaded: 'ollama pull mistral' and 'ollama pull nomic-embed-text'

For now, I can tell you that I found relevant content in your documents, but I need the AI model to provide a proper response."""
    
    def check_connection(self) -> bool:
        """
        Check if Ollama is running and accessible.
        
        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> list:
        """
        List available models in Ollama.
        
        Returns:
            List of available model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except:
            return []