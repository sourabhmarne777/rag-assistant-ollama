from src.scraping.web_scraper import web_scraper
from src.vector_store.qdrant_client import qdrant_service
from src.embeddings.embedder import embedding_service
from src.llm.ollama_client import ollama_service
from config.settings import MAX_TEXT_LENGTH
import logging

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        # Setup collection on initialization
        qdrant_service.setup_collection()
        self.qdrant_service = qdrant_service  # Expose for UI access
        
    def process_urls(self, urls):
        """Process URLs: scrape, embed, and store in vector database"""
        try:
            # Check storage capacity first
            storage_stats = qdrant_service.get_storage_stats()
            if storage_stats and storage_stats['status'] == 'critical':
                logger.warning("Storage capacity critical - processing may fail")
            
            # Filter out already processed URLs
            new_urls = [url for url in urls if not qdrant_service.url_exists(url)]
            
            if not new_urls:
                logger.info("All URLs already processed")
                return True
            
            # Check if we can add new documents
            if not qdrant_service.check_storage_capacity(len(new_urls) * 5):  # Estimate 5 chunks per URL
                logger.warning("Approaching storage limit - consider cleanup")
            
            # Scrape URLs
            logger.info(f"Scraping {len(new_urls)} new URLs")
            documents = web_scraper.scrape_urls(new_urls)
            
            if not documents:
                logger.warning("No documents scraped successfully")
                return False
            
            # Generate embeddings
            logger.info("Generating embeddings")
            texts = [doc['content'] for doc in documents]
            embeddings = embedding_service.embed_texts(texts)
            
            if not embeddings:
                logger.error("Failed to generate embeddings")
                return False
            
            # Store in vector database
            logger.info("Storing documents in vector database")
            success = qdrant_service.add_documents(documents, embeddings)
            
            if success:
                logger.info(f"Successfully processed {len(new_urls)} URLs")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in URL processing: {e}")
            return False

    def query(self, question, urls=None):
        """Query the RAG system"""
        try:
            # Process URLs if provided
            if urls:
                if not self.process_urls(urls):
                    return "Failed to process some URLs. Proceeding with available data..."
            
            # Generate query embedding
            logger.info("Generating query embedding")
            query_vector = embedding_service.embed_query(question)
            
            if not query_vector:
                return "Failed to process your question. Please try again."
            
            # Search for relevant documents
            logger.info("Searching for relevant documents")
            relevant_docs = qdrant_service.search_similar(query_vector)
            
            if not relevant_docs:
                return "I couldn't find any relevant information to answer your question. Please try providing more specific URLs or rephrasing your question."
            
            # Prepare context with smart truncation
            context_parts = []
            total_length = 0
            max_context_length = MAX_TEXT_LENGTH // 2  # Reserve space for prompt
            
            for i, doc in enumerate(relevant_docs, 1):
                source_info = f"Source {i} (Relevance: {doc['score']:.3f}):\nTitle: {doc['title']}\nURL: {doc['url']}\nContent: {doc['content'][:800]}...\n"
                
                if total_length + len(source_info) > max_context_length:
                    break
                
                context_parts.append(source_info)
                total_length += len(source_info)
            
            context = "\n".join(context_parts)
            
            # Generate answer
            logger.info("Generating answer")
            prompt = ollama_service.create_rag_prompt(context, question)
            answer = ollama_service.generate_answer(prompt)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            return f"An error occurred while processing your question: {str(e)}"

    def get_stats(self):
        """Get pipeline statistics"""
        collection_info = qdrant_service.get_collection_info()
        storage_stats = qdrant_service.get_storage_stats()
        return {
            'collection_info': collection_info,
            'storage_stats': storage_stats
        }

# Global instance
rag_pipeline = RAGPipeline()