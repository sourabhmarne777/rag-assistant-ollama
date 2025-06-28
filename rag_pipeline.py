from web_scraper import web_scraper
from vector_store import vector_store
from embeddings import embedding_service
from llm import llm_service
from settings import MAX_TEXT_LENGTH
import logging

logger = logging.getLogger(__name__)

class RAGPipeline:
    def process_urls(self, urls):
        """Process URLs and store in vector database"""
        try:
            # Filter new URLs
            new_urls = [url for url in urls if not vector_store.url_exists(url)]
            
            if not new_urls:
                logger.info("All URLs already processed")
                return True
            
            # Scrape content
            documents = web_scraper.scrape_urls(new_urls)
            if not documents:
                logger.warning("No documents scraped successfully")
                return False
            
            # Generate embeddings
            texts = [doc['content'] for doc in documents]
            embeddings = embedding_service.embed_texts(texts)
            if not embeddings:
                logger.error("Failed to generate embeddings")
                return False
            
            # Store in vector database
            success = vector_store.add_documents(documents, embeddings)
            if success:
                logger.info(f"Successfully processed {len(documents)} documents")
            return success
            
        except Exception as e:
            logger.error(f"Error processing URLs: {e}")
            return False

    def query(self, question, urls=None):
        """Answer question using RAG"""
        try:
            # Process URLs if provided
            if urls:
                process_success = self.process_urls(urls)
                if not process_success:
                    logger.warning("Some URLs failed to process, continuing with available data")
            
            # Generate query embedding
            query_vector = embedding_service.embed_query(question)
            if not query_vector:
                return "Failed to process your question. Please try again."
            
            # Search for relevant documents
            relevant_docs = vector_store.search_similar(query_vector)
            
            # Debug: Show what we found
            logger.info(f"Search found {len(relevant_docs)} documents")
            for i, doc in enumerate(relevant_docs[:3]):
                logger.info(f"Doc {i+1}: {doc['title'][:50]}... (score: {doc['score']:.3f})")
            
            if not relevant_docs:
                # Debug: Check what's in the database
                all_docs = vector_store.get_all_documents()
                logger.info(f"Database contains {len(all_docs)} total documents")
                
                return f"""I couldn't find relevant information in the provided sources. 

**Debug Info:**
- Total documents in database: {len(all_docs)}
- Search threshold: 0.3 (very permissive)
- Your question was processed successfully

**Possible issues:**
1. The URLs don't contain content related to your question
2. The content couldn't be properly extracted from the URLs
3. The embedding models might not be matching well

**Try:**
- Asking more specific questions
- Using different URLs with clearer content
- Checking if the URLs are accessible"""
            
            # Prepare context with better formatting
            context_parts = []
            total_length = 0
            max_context_length = MAX_TEXT_LENGTH // 2  # Reserve space for prompt
            
            for i, doc in enumerate(relevant_docs, 1):
                # Include more context per document
                content_snippet = doc['content'][:1500]  # Increased snippet size
                source_info = f"Source {i} (Relevance: {doc['score']:.2f}):\nTitle: {doc['title']}\nURL: {doc['url']}\nContent: {content_snippet}\n"
                
                if total_length + len(source_info) > max_context_length:
                    break
                
                context_parts.append(source_info)
                total_length += len(source_info)
            
            context = "\n---\n".join(context_parts)
            
            # Generate answer with better context
            answer = llm_service.generate_answer(context, question)
            
            # Add source information to answer
            sources = [f"• {doc['title']} ({doc['url']})" for doc in relevant_docs[:3]]
            answer_with_sources = f"{answer}\n\n**Sources Used:**\n" + "\n".join(sources)
            
            return answer_with_sources
            
        except Exception as e:
            logger.error(f"Error in query: {e}")
            return f"An error occurred while processing your question: {str(e)}"

    def get_storage_stats(self):
        """Get storage statistics"""
        return vector_store.get_storage_stats()

rag_pipeline = RAGPipeline()