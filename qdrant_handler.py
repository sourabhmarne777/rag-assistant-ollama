import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()  # Load variables from .env

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)
