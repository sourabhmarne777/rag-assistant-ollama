from langchain_ollama import OllamaEmbeddings
from settings import EMBEDDING_MODEL, OLLAMA_BASE_URL
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.embedder = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )

    def embed_texts(self, texts):
        """Convert texts to embeddings"""
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
        """Convert query to embedding"""
        try:
            embedding = self.embedder.embed_query(query)
            return embedding
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return None

embedding_service = EmbeddingService()