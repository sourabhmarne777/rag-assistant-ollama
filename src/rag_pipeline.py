import os
from typing import List, Tuple, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from .vector_store import QdrantVectorStore
from .llm_client import OllamaClient
from .embeddings import EmbeddingClient

class RAGPipeline:
    """Main RAG pipeline orchestrating all components"""
    
    def __init__(self):
        self.vector_store = QdrantVectorStore()
        self.llm_client = OllamaClient()
        self.embedding_client = EmbeddingClient()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def add_documents(
        self, 
        documents: List[str], 
        source_type: str, 
        source_name: str, 
        session_id: str
    ) -> bool:
        """Add documents to the vector store with metadata"""
        try:
            # Create Document objects with metadata
            docs = []
            for i, doc_text in enumerate(documents):
                metadata = {
                    "source_type": source_type,
                    "source_name": source_name,
                    "session_id": session_id,
                    "chunk_id": i
                }
                docs.append(Document(page_content=doc_text, metadata=metadata))
            
            # Generate embeddings
            texts = [doc.page_content for doc in docs]
            embeddings = self.embedding_client.embed_documents(texts)
            
            # Store in vector database
            return self.vector_store.add_documents(docs, embeddings)
            
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False
    
    def query(self, question: str, session_id: str, k: int = 5) -> Tuple[str, List[str]]:
        """Query the RAG system with session-based filtering"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_client.embed_query(question)
            
            # Search for relevant documents with session filter
            relevant_docs = self.vector_store.similarity_search(
                query_embedding, 
                k=k, 
                filter_dict={"session_id": session_id}
            )
            
            if not relevant_docs:
                return "I don't have any relevant information to answer your question. Please upload some documents first.", []
            
            # Prepare context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Generate response using LLM
            response = self.llm_client.generate_response(question, context)
            
            # Extract sources
            sources = []
            for doc in relevant_docs:
                source_info = f"{doc.metadata.get('source_name', 'Unknown')} ({doc.metadata.get('source_type', 'unknown')})"
                if source_info not in sources:
                    sources.append(source_info)
            
            return response, sources
            
        except Exception as e:
            print(f"Error querying RAG system: {e}")
            return f"An error occurred while processing your question: {str(e)}", []
    
    def get_session_documents(self, session_id: str) -> List[dict]:
        """Get all documents for a specific session"""
        try:
            return self.vector_store.get_documents_by_session(session_id)
        except Exception as e:
            print(f"Error retrieving session documents: {e}")
            return []
    
    def clear_session(self, session_id: str) -> bool:
        """Clear all documents for a specific session"""
        try:
            return self.vector_store.delete_by_session(session_id)
        except Exception as e:
            print(f"Error clearing session: {e}")
            return False