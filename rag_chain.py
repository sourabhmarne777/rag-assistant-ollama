from web_scraper import scrape_urls
from qdrant_handler import setup_collection, add_documents_to_qdrant, search_qdrant
from embedder import embed_texts, embed_query
from ollama_llm import get_answer

def run_rag_pipeline(query, urls):
    setup_collection()
    
    raw_texts = scrape_urls(urls)
    embeddings = embed_texts(raw_texts)
    add_documents_to_qdrant(raw_texts, embeddings)

    query_vec = embed_query(query)
    relevant_docs = search_qdrant(query_vec)

    context = "\n\n".join(relevant_docs)
    prompt = f"Answer the question based on the context below:\n\n{context}\n\nQuestion: {query}"
    return get_answer(prompt)
