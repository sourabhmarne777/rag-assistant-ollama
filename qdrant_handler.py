from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(path="qdrant_data")  # local Qdrant

def setup_collection(name="web_docs", dim=384):
    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

def add_documents_to_qdrant(texts, embeddings, collection="web_docs"):
    points = [PointStruct(id=i, vector=emb, payload={"text": t}) for i, (t, emb) in enumerate(zip(texts, embeddings))]
    client.upsert(collection_name=collection, points=points)

def search_qdrant(query_vector, top_k=5, collection="web_docs"):
    hits = client.search(collection_name=collection, query_vector=query_vector, limit=top_k)
    return [hit.payload["text"] for hit in hits]
