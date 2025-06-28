from rag_chain import run_rag_pipeline

if __name__ == "__main__":
    user_query = input("Ask your question: ")
    urls = [
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "https://www.pinecone.io/learn/retrieval-augmented-generation/"
    ]
    response = run_rag_pipeline(user_query, urls)
    print("\n--- Answer ---\n", response)
