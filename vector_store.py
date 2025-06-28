import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from settings import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, TOP_K_RESULTS, SIMILARITY_THRESHOLD
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.setup_collection()

    def setup_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            if COLLECTION_NAME not in [col.name for col in collections]:
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
                logger.info(f"Created collection: {COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Error setting up collection: {e}")

    def add_documents(self, documents, embeddings):
        """Add documents to vector store"""
        try:
            points = []
            for doc, embedding in zip(documents, embeddings):
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, doc['url']))
                point = PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload={
                        "content": doc['content'],
                        "url": doc['url'],
                        "title": doc['title']
                    }
                )
                points.append(point)
            
            self.client.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.info(f"Added {len(points)} documents")
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False

    def search_similar(self, query_vector):
        """Search for similar documents with more permissive settings"""
        try:
            # First try with threshold
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=TOP_K_RESULTS,
                score_threshold=SIMILARITY_THRESHOLD,
                with_payload=True
            )
            
            # If no results with threshold, try without threshold
            if not results:
                logger.info(f"No results with threshold {SIMILARITY_THRESHOLD}, trying without threshold")
                results = self.client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector,
                    limit=TOP_K_RESULTS,
                    with_payload=True
                )
            
            search_results = [{
                'content': hit.payload['content'],
                'url': hit.payload['url'],
                'title': hit.payload['title'],
                'score': hit.score
            } for hit in results]
            
            logger.info(f"Found {len(search_results)} documents (threshold: {SIMILARITY_THRESHOLD})")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

    def url_exists(self, url):
        """Check if URL already exists"""
        try:
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, url))
            result = self.client.retrieve(collection_name=COLLECTION_NAME, ids=[doc_id])
            return len(result) > 0
        except:
            return False

    def get_storage_stats(self):
        """Get storage usage statistics"""
        try:
            info = self.client.get_collection(COLLECTION_NAME)
            current_count = info.vectors_count or 0
            limit = 1000000  # Free tier limit
            usage_percent = (current_count / limit) * 100
            
            return {
                'current_vectors': current_count,
                'limit': limit,
                'usage_percent': usage_percent,
                'status': 'critical' if usage_percent > 90 else 'warning' if usage_percent > 80 else 'good'
            }
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {
                'current_vectors': 0,
                'limit': 1000000,
                'usage_percent': 0,
                'status': 'good'
            }

    def reset_collection(self):
        """Delete all vectors"""
        try:
            self.client.delete_collection(COLLECTION_NAME)
            self.setup_collection()
            logger.info("Collection reset")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False

    def get_all_documents(self):
        """Debug: Get all documents in collection"""
        try:
            points, _ = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                with_payload=True
            )
            return [{
                'url': point.payload['url'],
                'title': point.payload['title'],
                'content_length': len(point.payload['content'])
            } for point in points]
        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return []

vector_store = VectorStore()