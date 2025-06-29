import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType
from langchain.schema import Document
import uuid

class QdrantVectorStore:
    """
    Qdrant vector store implementation with session-based filtering.
    
    This class handles all vector database operations including:
    - Collection management and indexing
    - Document storage with metadata
    - Similarity search with filtering
    - Session-based data isolation
    """
    
    def __init__(self):
        """Initialize Qdrant client and collection"""
        # Initialize Qdrant client with cloud credentials
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        # Collection configuration
        self.collection_name = os.getenv("COLLECTION_NAME", "rag_documents")
        self.vector_size = 768  # nomic-embed-text embedding dimension
        
        # Ensure collection exists with proper configuration
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist with proper indexing for efficient filtering"""
        try:
            # Check if collection already exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create new collection with vector configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,      # Embedding dimension
                        distance=Distance.COSINE    # Cosine similarity for semantic search
                    )
                )
                print(f"Created collection: {self.collection_name}")
            
            # Create payload indexes for efficient filtering
            # Session ID index for session-based isolation
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="session_id",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                print("Created session_id index")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    print(f"Note: Could not create session_id index: {e}")
            
            # Source type index for filtering by document/web content
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="source_type",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                print("Created source_type index")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    print(f"Note: Could not create source_type index: {e}")
                
        except Exception as e:
            print(f"Error ensuring collection exists: {e}")
            raise
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> bool:
        """
        Add documents with embeddings to the vector store.
        
        Args:
            documents: List of Document objects with content and metadata
            embeddings: List of embedding vectors corresponding to documents
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            points = []
            
            # Create point objects for each document-embedding pair
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                # Skip documents with invalid embeddings
                if not embedding or len(embedding) == 0:
                    print(f"Skipping document {i} due to invalid embedding")
                    continue
                
                # Create point with unique ID, vector, and metadata
                point = PointStruct(
                    id=str(uuid.uuid4()),  # Unique identifier
                    vector=embedding,       # Vector representation
                    payload={               # Metadata for filtering and retrieval
                        "content": doc.page_content,
                        "source_type": doc.metadata.get("source_type"),
                        "source_name": doc.metadata.get("source_name"),
                        "session_id": doc.metadata.get("session_id"),
                        "chunk_id": doc.metadata.get("chunk_id", i)
                    }
                )
                points.append(point)
            
            # Validate that we have points to upload
            if not points:
                print("No valid points to upload")
                return False
            
            # Upload points to Qdrant collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            print(f"Successfully added {len(points)} documents to vector store")
            return True
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            return False
    
    def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 5, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents with optional filtering.
        
        Args:
            query_embedding: Query vector for similarity search
            k: Number of similar documents to return
            filter_dict: Optional filters (e.g., {"session_id": "abc123"})
            
        Returns:
            List of Document objects with similarity scores
        """
        try:
            # Validate query embedding
            if not query_embedding or len(query_embedding) == 0:
                print("Invalid query embedding")
                return []
            
            # Build filter conditions if provided
            query_filter = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                query_filter = Filter(must=conditions)
            
            # Perform similarity search in vector space
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=k
            )
            
            # Convert search results to Document objects
            documents = []
            for result in search_results:
                doc = Document(
                    page_content=result.payload["content"],
                    metadata={
                        "source_type": result.payload.get("source_type"),
                        "source_name": result.payload.get("source_name"),
                        "session_id": result.payload.get("session_id"),
                        "chunk_id": result.payload.get("chunk_id"),
                        "score": result.score  # Similarity score
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error performing similarity search: {e}")
            return []
    
    def get_documents_by_session(self, session_id: str) -> List[dict]:
        """
        Get all documents for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of document dictionaries with metadata
        """
        try:
            # Use scroll to get all documents for the session
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=session_id)
                        )
                    ]
                ),
                limit=1000  # Adjust based on expected session size
            )
            
            # Convert points to document dictionaries
            documents = []
            for point in points:
                documents.append({
                    "id": point.id,
                    "content": point.payload["content"],
                    "source_type": point.payload.get("source_type"),
                    "source_name": point.payload.get("source_name"),
                    "chunk_id": point.payload.get("chunk_id")
                })
            
            return documents
            
        except Exception as e:
            print(f"Error getting documents by session: {e}")
            return []
    
    def delete_by_session(self, session_id: str) -> bool:
        """
        Delete all documents for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Delete all points matching the session ID
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=session_id)
                        )
                    ]
                )
            )
            print(f"Deleted all documents for session: {session_id}")
            return True
            
        except Exception as e:
            print(f"Error deleting documents by session: {e}")
            return False
    
    def get_collection_info(self) -> dict:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vector_size": info.config.params.vectors.size,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}