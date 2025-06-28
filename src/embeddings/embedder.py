from langchain_ollama import OllamaEmbeddings
from config.settings import EMBEDDING_MODEL, OLLAMA_BASE_URL
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        try:
            self.embedder = OllamaEmbeddings(
                model=EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL
            )
            logger.info(f"Initialized embedder with model: {EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            raise

    def embed_texts(self, texts):
        """Embed multiple texts and return embeddings"""
        try:
            if not texts:
                return []
            embeddings = self.embedder.embed_documents(texts)
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            return []

    def embed_query(self, query):
        """Embed a single query and return embedding"""
        try:
            embedding = self.embedder.embed_query(query)
            logger.info("Generated query embedding")
            return embedding
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return None

# Global instance
embedding_service = EmbeddingService()