import os
import requests
from typing import List

class EmbeddingClient:
    """Client for generating embeddings using Ollama"""
    
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        self.api_url = f"{self.base_url}/api/embeddings"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents"""
        embeddings = []
        
        # Check if Ollama is available first
        if not self._check_ollama_connection():
            print("Warning: Ollama is not available. Using fallback embeddings.")
            # Return dummy embeddings as fallback
            return [[0.0] * 768 for _ in texts]
        
        for text in texts:
            embedding = self.embed_query(text)
            if embedding:
                embeddings.append(embedding)
            else:
                # Fallback: create zero vector if embedding fails
                embeddings.append([0.0] * 768)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            # Check connection first
            if not self._check_ollama_connection():
                print("Ollama not available, using fallback embedding")
                return [0.0] * 768
            
            payload = {
                "model": self.model,
                "prompt": text
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=30
            )
            
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
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def check_model_availability(self) -> bool:
        """Check if the embedding model is available"""
        try:
            if not self._check_ollama_connection():
                return False
            # Test with a simple text
            test_embedding = self.embed_query("test")
            return len(test_embedding) > 0
        except:
            return False