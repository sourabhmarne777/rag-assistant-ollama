from web_scraper import web_scraper
from vector_store import vector_store
from embeddings import embedding_service
from llm import llm_service
from settings import MAX_TEXT_LENGTH
import logging

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        self.processed_context_ids = set()  # Track processed context to avoid reuse after reset
    
    def reset_context(self):
        """Reset processed context tracking - called when user clicks Reset All"""
        self.processed_context_ids.clear()
        logger.info("RAG Pipeline context tracking reset - all previous context cleared")

    def process_urls(self, urls):
        """Process URLs and store in vector database"""
        try:
            if not urls:
                return True
                
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

    def process_file_contexts(self, file_contexts):
        """Process file contents and store in vector database with deduplication"""
        try:
            if not file_contexts:
                return True
            
            documents = []
            for i, file_content in enumerate(file_contexts):
                # Extract file name and content
                if file_content.startswith("File: "):
                    lines = file_content.split('\n', 1)
                    file_name = lines[0].replace("File: ", "")
                    content = lines[1] if len(lines) > 1 else ""
                else:
                    file_name = f"File {i+1}"
                    content = file_content
                
                if content.strip():
                    # Create unique ID based on content hash and filename
                    content_hash = hash(content + file_name)
                    context_id = f"file_{content_hash}_{i}"
                    
                    # Only add if not already processed (deduplication)
                    if context_id not in self.processed_context_ids:
                        documents.append({
                            'url': context_id,
                            'title': file_name,
                            'content': content
                        })
                        # Track this context
                        self.processed_context_ids.add(context_id)
                    else:
                        logger.info(f"Skipping duplicate file: {file_name}")
            
            if not documents:
                logger.info("No new file contexts to process")
                return True
            
            # Generate embeddings
            texts = [doc['content'] for doc in documents]
            embeddings = embedding_service.embed_texts(texts)
            if not embeddings:
                logger.error("Failed to generate embeddings for files")
                return False
            
            # Store in vector database
            success = vector_store.add_documents(documents, embeddings)
            if success:
                logger.info(f"Successfully processed {len(documents)} new file contexts")
            return success
            
        except Exception as e:
            logger.error(f"Error processing file contexts: {e}")
            return False

    def is_query_relevant_to_context(self, query, context_snippet, min_relevance=0.2):
        """Check if query is relevant to context using keyword matching"""
        query_words = set(query.lower().split())
        context_words = set(context_snippet.lower().split())
        
        # Calculate overlap
        overlap = len(query_words.intersection(context_words))
        relevance = overlap / len(query_words) if query_words else 0
        
        return relevance >= min_relevance

    def filter_relevant_sources(self, sources, question, max_sources=3):
        """Filter and return only truly relevant sources with better deduplication"""
        if not sources:
            return []
        
        relevant_sources = []
        seen_content = set()  # Track content to avoid duplicates
        
        for source in sources:
            # Check semantic similarity score
            if source.get('score', 0) > 0.12:  # Lower threshold for better recall
                # Additional keyword relevance check
                if self.is_query_relevant_to_context(question, source['content'][:500]):
                    # Check for content duplication (same document, different chunks)
                    content_signature = source['content'][:200].lower().strip()
                    if content_signature not in seen_content:
                        relevant_sources.append(source)
                        seen_content.add(content_signature)
        
        # Sort by score and return top sources
        relevant_sources.sort(key=lambda x: x.get('score', 0), reverse=True)
        return relevant_sources[:max_sources]

    def chat_query(self, question, urls=None, file_contexts=None, conversation_context=""):
        """Answer question using RAG with improved relevance filtering and deduplication"""
        try:
            has_external_context = bool(urls or file_contexts)
            
            # If no external context, use general AI chat
            if not has_external_context:
                logger.info("No external context - using general AI chat")
                if conversation_context:
                    enhanced_question = f"Previous conversation:\n{conversation_context}\n\nCurrent question: {question}"
                else:
                    enhanced_question = question
                
                answer = llm_service.generate_answer("", enhanced_question)
                return answer
            
            # Process URLs if provided
            if urls:
                process_success = self.process_urls(urls)
                if not process_success:
                    logger.warning("Some URLs failed to process, continuing with available data")
            
            # Process file contexts if provided
            if file_contexts:
                file_success = self.process_file_contexts(file_contexts)
                if not file_success:
                    logger.warning("Some files failed to process, continuing with available data")
            
            # Search for relevant documents
            query_vector = embedding_service.embed_query(question)
            relevant_docs = []
            
            if query_vector:
                # Get search results
                search_results = vector_store.search_similar(query_vector, limit=10)
                
                # Filter for truly relevant sources with deduplication
                relevant_docs = self.filter_relevant_sources(search_results, question, max_sources=3)
                
                logger.info(f"Search found {len(search_results)} documents, filtered to {len(relevant_docs)} relevant unique sources")
            
            # Prepare context from relevant documents only
            context_parts = []
            used_sources = []
            total_length = 0
            max_context_length = MAX_TEXT_LENGTH // 2
            
            # Add only the most relevant documents
            for i, doc in enumerate(relevant_docs, 1):
                content_snippet = doc['content'][:1000]  # Reasonable snippet size
                source_info = f"Source {i} ({doc['title']}):\n{content_snippet}\n"
                
                if total_length + len(source_info) > max_context_length:
                    break
                
                context_parts.append(source_info)
                used_sources.append(doc)
                total_length += len(source_info)
            
            # Combine all context
            document_context = "\n---\n".join(context_parts) if context_parts else ""
            
            # Create enhanced question with conversation context
            if conversation_context:
                enhanced_question = f"Previous conversation:\n{conversation_context}\n\nCurrent question: {question}"
            else:
                enhanced_question = question
            
            # Generate answer
            if document_context:
                # RAG mode with relevant context
                answer = llm_service.generate_answer(document_context, enhanced_question)
                
                # Add source information only for actually used sources
                if used_sources:
                    sources = []
                    for doc in used_sources:
                        if doc['url'].startswith('file_'):
                            sources.append(f"• {doc['title']}")
                        else:
                            sources.append(f"• {doc['title']} ({doc['url']})")
                    
                    if sources:
                        answer += f"\n\n**Sources Used:**\n" + "\n".join(sources)
            else:
                # No relevant documents found
                answer = llm_service.generate_answer("", enhanced_question)
                answer += "\n\n*Note: I couldn't find specific information in your uploaded sources relevant to this question, so I've provided a general response.*"
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in chat query: {e}")
            return f"I encountered an error while processing your question: {str(e)}"

rag_pipeline = RAGPipeline()