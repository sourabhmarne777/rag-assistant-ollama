from langchain.embeddings import OllamaEmbeddings

embedder = OllamaEmbeddings(model="mistral")  # or "nomic-embed-text"

def embed_texts(texts):
    return embedder.embed_documents(texts)

def embed_query(query):
    return embedder.embed_query(query)
