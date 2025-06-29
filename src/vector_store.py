import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType
from langchain.schema import Document
import uuid

class QdrantVectorStore:
    """Qdrant vector store implementation with session-based filtering"""
    
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.collection_name = os.getenv("COLLECTION_NAME", "rag_documents")
        self.vector_size = 768  # nomic-embed-text embedding size
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist with proper indexing"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection with vector config
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created collection: {self.collection_name}")
            
            # Create payload indexes for filtering
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
        """Add documents with embeddings to the vector store"""
        try:
            points = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                # Skip if embedding is empty or invalid
                if not embedding or len(embedding) == 0:
                    print(f"Skipping document {i} due to invalid embedding")
                    continue
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "content": doc.page_content,
                        "source_type": doc.metadata.get("source_type"),
                        "source_name": doc.metadata.get("source_name"),
                        "session_id": doc.metadata.get("session_id"),
                        "chunk_id": doc.metadata.get("chunk_id", i)
                    }
                )
                points.append(point)
            
            if not points:
                print("No valid points to upload")
                return False
            
            # Upload points to Qdrant
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
        """Search for similar documents with optional filtering"""
        try:
            # Skip if query embedding is invalid
            if not query_embedding or len(query_embedding) == 0:
                print("Invalid query embedding")
                return []
            
            # Build filter if provided
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
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=k
            )
            
            # Convert results to Document objects
            documents = []
            for result in search_results:
                doc = Document(
                    page_content=result.payload["content"],
                    metadata={
                        "source_type": result.payload.get("source_type"),
                        "source_name": result.payload.get("source_name"),
                        "session_id": result.payload.get("session_id"),
                        "chunk_id": result.payload.get("chunk_id"),
                        "score": result.score
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error performing similarity search: {e}")
            return []
    
    def get_documents_by_session(self, session_id: str) -> List[dict]:
        """Get all documents for a specific session"""
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
                limit=1000  # Adjust as needed
            )
            
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
        """Delete all documents for a specific session"""
        try:
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
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.config.params.vectors.size,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}