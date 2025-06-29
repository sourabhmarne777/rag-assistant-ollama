import os
import requests
from typing import List

class EmbeddingClient:
    """
    Client for generating embeddings using Ollama's embedding models.
    
    This class handles:
    - Connection to Ollama embedding API
    - Batch embedding generation for documents
    - Single query embedding generation
    - Fallback handling when Ollama is unavailable
    """
    
    def __init__(self):
        """Initialize embedding client with configuration from environment"""
        # Ollama server configuration
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        self.api_url = f"{self.base_url}/api/embeddings"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (one per input text)
        """
        embeddings = []
        
        # Check if Ollama is available first
        if not self._check_ollama_connection():
            print("Warning: Ollama is not available. Using fallback embeddings.")
            # Return dummy embeddings as fallback (768-dimensional zero vectors)
            return [[0.0] * 768 for _ in texts]
        
        # Generate embedding for each text
        for text in texts:
            embedding = self.embed_query(text)
            if embedding:
                embeddings.append(embedding)
            else:
                # Fallback: create zero vector if embedding fails
                embeddings.append([0.0] * 768)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            # Check connection first to avoid unnecessary API calls
            if not self._check_ollama_connection():
                print("Ollama not available, using fallback embedding")
                return [0.0] * 768
            
            # Prepare API request payload
            payload = {
                "model": self.model,
                "prompt": text
            }
            
            # Make request to Ollama embeddings API
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=30  # Allow time for embedding generation
            )
            
            # Handle successful response
            if response.status_code == 200:
                result = response.json()
                return result.get("embedding", [])
            else:
                print(f"Embedding API error: {response.status_code}")
                return [0.0] * 768
                
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return [0.0] * 768
        except Exception as e:
            print(f"Unexpected error in embedding: {e}")
            return [0.0] * 768
    
    def _check_ollama_connection(self) -> bool:
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
    
    def check_model_availability(self) -> bool:
        """
        Check if the embedding model is available and working.
        
        Returns:
            True if model is available and functional, False otherwise
        """
        try:
            if not self._check_ollama_connection():
                return False
            
            # Test with a simple text to verify model functionality
            test_embedding = self.embed_query("test")
            return len(test_embedding) > 0
        except:
            return False