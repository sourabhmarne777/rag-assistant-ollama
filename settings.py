import os
from dotenv import load_dotenv

load_dotenv()

# Qdrant Cloud Configuration (Required)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# Text Processing - More permissive settings
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "15000"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))  # Increased results
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))  # Much lower threshold

# Validation
if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env file")