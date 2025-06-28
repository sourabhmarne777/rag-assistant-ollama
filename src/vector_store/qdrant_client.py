import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config.settings import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, TOP_K_RESULTS, SIMILARITY_THRESHOLD
import logging

logger = logging.getLogger(__name__)

class QdrantService:
    def __init__(self):
        try:
            self.client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY
            )
            logger.info(f"Connected to Qdrant Cloud at {QDRANT_URL}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant Cloud: {e}")
            raise

    def setup_collection(self, vector_size=768):
        """Create collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if COLLECTION_NAME in collection_names:
                logger.info(f"Collection {COLLECTION_NAME} already exists")
                return True
            
            # Create collection
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Created collection: {COLLECTION_NAME}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up collection: {e}")
            return False

    def add_documents(self, documents, embeddings):
        """Add documents with embeddings to Qdrant"""
        try:
            if not documents or not embeddings:
                logger.warning("No documents or embeddings to add")
                return False
            
            # Check storage before adding
            if not self.check_storage_capacity(len(documents)):
                logger.warning("Approaching storage limit - consider cleanup")
            
            points = []
            for doc, embedding in zip(documents, embeddings):
                # Create unique ID based on URL to avoid duplicates
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, doc['url']))
                
                point = PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload={
                        "content": doc['content'],
                        "url": doc['url'],
                        "title": doc['title'],
                        "timestamp": doc.get('timestamp', None)
                    }
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            
            logger.info(f"Added {len(points)} documents to Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to Qdrant: {e}")
            return False

    def search_similar(self, query_vector, limit=TOP_K_RESULTS):
        """Search for similar documents using cosine similarity"""
        try:
            if not query_vector:
                logger.error("No query vector provided")
                return []
            
            search_result = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=limit,
                score_threshold=SIMILARITY_THRESHOLD,
                with_payload=True
            )
            
            results = []
            for hit in search_result:
                results.append({
                    'content': hit.payload['content'],
                    'url': hit.payload['url'],
                    'title': hit.payload['title'],
                    'score': hit.score
                })
            
            logger.info(f"Found {len(results)} similar documents above threshold {SIMILARITY_THRESHOLD}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            return []

    def get_collection_info(self):
        """Get information about the collection"""
        try:
            info = self.client.get_collection(COLLECTION_NAME)
            return {
                'name': COLLECTION_NAME,
                'vectors_count': info.vectors_count,
                'status': info.status.value if hasattr(info.status, 'value') else str(info.status)
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None

    def url_exists(self, url):
        """Check if URL already exists in the collection"""
        try:
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, url))
            result = self.client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[doc_id]
            )
            return len(result) > 0
        except Exception as e:
            logger.debug(f"URL check failed (expected for new URLs): {e}")
            return False

    def check_storage_capacity(self, new_docs_count=0):
        """Check if we're approaching storage limits"""
        try:
            info = self.get_collection_info()
            if not info:
                return True
            
            current_count = info['vectors_count']
            estimated_total = current_count + new_docs_count
            
            # Assuming 1M vector limit for free tier
            FREE_TIER_LIMIT = 1000000
            usage_percent = (estimated_total / FREE_TIER_LIMIT) * 100
            
            logger.info(f"Storage usage: {usage_percent:.1f}% ({estimated_total:,}/{FREE_TIER_LIMIT:,})")
            
            if usage_percent > 90:
                logger.warning("⚠️ CRITICAL: >90% storage used!")
                return False
            elif usage_percent > 80:
                logger.warning("⚠️ WARNING: >80% storage used!")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking storage capacity: {e}")
            return True

    def cleanup_by_url_pattern(self, url_patterns):
        """Delete vectors matching URL patterns"""
        try:
            # Scroll through all points
            points, _ = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=10000,
                with_payload=True
            )
            
            ids_to_delete = []
            for point in points:
                url = point.payload.get('url', '')
                if any(pattern in url for pattern in url_patterns):
                    ids_to_delete.append(point.id)
            
            if ids_to_delete:
                self.client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=ids_to_delete
                )
                logger.info(f"Deleted {len(ids_to_delete)} vectors matching patterns: {url_patterns}")
                return len(ids_to_delete)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0

    def reset_collection(self):
        """Delete all vectors (nuclear option)"""
        try:
            self.client.delete_collection(COLLECTION_NAME)
            self.setup_collection()
            logger.info("Collection reset - all vectors deleted")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False

    def get_storage_stats(self):
        """Get detailed storage statistics"""
        try:
            info = self.get_collection_info()
            if not info:
                return None
            
            current_count = info['vectors_count']
            FREE_TIER_LIMIT = 1000000
            usage_percent = (current_count / FREE_TIER_LIMIT) * 100
            remaining = FREE_TIER_LIMIT - current_count
            
            return {
                'current_vectors': current_count,
                'limit': FREE_TIER_LIMIT,
                'usage_percent': usage_percent,
                'remaining': remaining,
                'status': 'critical' if usage_percent > 90 else 'warning' if usage_percent > 80 else 'good'
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return None

# Global instance
qdrant_service = QdrantService()