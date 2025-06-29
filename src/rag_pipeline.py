import os
from typing import List, Tuple, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from .vector_store import QdrantVectorStore
from .llm_client import OllamaClient
from .embeddings import EmbeddingClient

class RAGPipeline:
    """
    Main RAG (Retrieval-Augmented Generation) pipeline orchestrating all components.
    
    This class coordinates the entire RAG workflow:
    1. Document processing and chunking
    2. Vector embedding generation
    3. Vector storage with session isolation
    4. Similarity search for relevant context
    5. LLM response generation with context
    """
    
    def __init__(self):
        """Initialize all RAG pipeline components"""
        # Vector store for document embeddings with session-based filtering
        self.vector_store = QdrantVectorStore()
        
        # LLM client for generating responses
        self.llm_client = OllamaClient()
        
        # Embedding client for converting text to vectors
        self.embedding_client = EmbeddingClient()
        
        # Text splitter for breaking documents into manageable chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # Maximum characters per chunk
            chunk_overlap=200,    # Overlap between chunks to maintain context
            length_function=len,  # Function to measure chunk length
        )
    
    def add_documents(
        self, 
        documents: List[str], 
        source_type: str, 
        source_name: str, 
        session_id: str
    ) -> bool:
        """
        Add documents to the vector store with comprehensive metadata.
        
        Args:
            documents: List of text chunks to add
            source_type: Type of source ("document" or "web")
            source_name: Name/URL of the source
            session_id: Unique session identifier for isolation
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create Document objects with metadata for each chunk
            docs = []
            for i, doc_text in enumerate(documents):
                metadata = {
                    "source_type": source_type,    # document/web classification
                    "source_name": source_name,    # filename or URL
                    "session_id": session_id,      # session isolation
                    "chunk_id": i                  # chunk index within document
                }
                docs.append(Document(page_content=doc_text, metadata=metadata))
            
            # Generate vector embeddings for all document chunks
            texts = [doc.page_content for doc in docs]
            embeddings = self.embedding_client.embed_documents(texts)
            
            # Store documents and embeddings in vector database
            return self.vector_store.add_documents(docs, embeddings)
            
        except Exception as e:
            print(f"Error adding documents to RAG pipeline: {e}")
            return False
    
    def query(self, question: str, session_id: str, k: int = 5) -> Tuple[str, List[str]]:
        """
        Query the RAG system with session-based filtering.
        
        Args:
            question: User's question
            session_id: Session ID for document filtering
            k: Number of relevant documents to retrieve
            
        Returns:
            Tuple of (response_text, source_list)
        """
        try:
            # Generate embedding for the user's question
            query_embedding = self.embedding_client.embed_query(question)
            
            # Search for relevant documents within the current session
            relevant_docs = self.vector_store.similarity_search(
                query_embedding, 
                k=k, 
                filter_dict={"session_id": session_id}  # Session-based filtering
            )
            
            # Handle case where no relevant documents are found
            if not relevant_docs:
                return "I don't have any relevant information to answer your question. Please upload some documents first.", []
            
            # Prepare context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Generate response using LLM with retrieved context
            response = self.llm_client.generate_response(question, context)
            
            # Extract unique sources from retrieved documents
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
        """
        Get all documents for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of document dictionaries with metadata
        """
        try:
            return self.vector_store.get_documents_by_session(session_id)
        except Exception as e:
            print(f"Error retrieving session documents: {e}")
            return []
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear all documents for a specific session.
        
        Args:
            session_id: Session identifier to clear
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            return self.vector_store.delete_by_session(session_id)
        except Exception as e:
            print(f"Error clearing session: {e}")
            return False