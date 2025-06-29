import os
from typing import List, Tuple, Dict, Any
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.vectorstores import Qdrant
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """RAG System for document retrieval and question answering"""
    
    def __init__(self):
        self.qdrant_url = os.getenv('QDRANT_URL')
        self.qdrant_api_key = os.getenv('QDRANT_API_KEY')
        self.collection_name = os.getenv('COLLECTION_NAME', 'rag_documents')
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'mistral')
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')
        
        # RAG settings
        self.similarity_threshold = 0.7
        self.max_results = 5
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize Qdrant client, embeddings, and LLM"""
        try:
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key
            )
            
            # Initialize embeddings
            self.embeddings = OllamaEmbeddings(
                base_url=self.ollama_base_url,
                model=self.embedding_model
            )
            
            # Initialize LLM
            self.llm = Ollama(
                base_url=self.ollama_base_url,
                model=self.ollama_model,
                temperature=0.1
            )
            
            # Create collection if it doesn't exist
            self._create_collection_if_not_exists()
            
            # Initialize vector store
            self.vector_store = Qdrant(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embeddings=self.embeddings
            )
            
            # Create RAG chain
            self._create_rag_chain()
            
            logger.info("RAG System initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            raise
    
    def _create_collection_if_not_exists(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Get embedding dimension
                test_embedding = self.embeddings.embed_query("test")
                embedding_dim = len(test_embedding)
                
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise
    
    def _create_rag_chain(self):
        """Create the RAG chain with custom prompt"""
        prompt_template = """
        You are a helpful AI assistant. Use the following context to answer the user's question.
        If you cannot find the answer in the context, say so clearly and provide a general response if possible.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer: Provide a comprehensive and accurate answer based on the context. If the context doesn't contain enough information, mention this limitation.
        """
        
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": self.max_results,
                "score_threshold": self.similarity_threshold
            }
        )
        
        # Create RAG chain
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )
    
    def add_documents(self, documents: List[Dict[str, Any]], source: str):
        """Add documents to the vector store"""
        try:
            # Convert to LangChain Document objects
            langchain_docs = []
            for doc in documents:
                langchain_doc = Document(
                    page_content=doc['content'],
                    metadata={
                        'source': source,
                        'chunk_id': doc.get('chunk_id', ''),
                        'page': doc.get('page', 0)
                    }
                )
                langchain_docs.append(langchain_doc)
            
            # Add to vector store
            self.vector_store.add_documents(langchain_docs)
            
            logger.info(f"Added {len(langchain_docs)} documents from {source}")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def query(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Query the RAG system"""
        try:
            # Get response from RAG chain
            result = self.rag_chain({"query": question})
            
            answer = result['result']
            source_docs = result['source_documents']
            
            # Format sources
            sources = []
            for doc in source_docs:
                sources.append({
                    'content': doc.page_content[:500],  # Truncate for display
                    'source': doc.metadata.get('source', 'Unknown'),
                    'page': doc.metadata.get('page', 0),
                    'score': getattr(doc, 'score', 0.0)
                })
            
            return answer, sources
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            return f"I encountered an error while processing your question: {str(e)}", []
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            self.qdrant_client.delete_collection(self.collection_name)
            self._create_collection_if_not_exists()
            
            # Reinitialize vector store
            self.vector_store = Qdrant(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embeddings=self.embeddings
            )
            
            # Recreate RAG chain
            self._create_rag_chain()
            
            logger.info("Collection cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            raise
    
    def update_models(self, ollama_model: str, embedding_model: str):
        """Update the models used by the RAG system"""
        try:
            self.ollama_model = ollama_model
            self.embedding_model = embedding_model
            
            # Reinitialize components
            self._initialize_components()
            
            logger.info(f"Updated models: LLM={ollama_model}, Embedding={embedding_model}")
            
        except Exception as e:
            logger.error(f"Error updating models: {str(e)}")
            raise
    
    def update_settings(self, similarity_threshold: float, max_results: int):
        """Update RAG settings"""
        try:
            self.similarity_threshold = similarity_threshold
            self.max_results = max_results
            
            # Recreate RAG chain with new settings
            self._create_rag_chain()
            
            logger.info(f"Updated settings: threshold={similarity_threshold}, max_results={max_results}")
            
        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'vectors_count': collection_info.vectors_count,
                'status': collection_info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}