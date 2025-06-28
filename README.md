# 🧠 RAG-based AI Assistant (LangChain + Ollama + Qdrant)

This is a Retrieval-Augmented Generation (RAG) assistant built using LangChain, Qdrant, and Ollama (Mistral model). It scrapes web content, stores embeddings in Qdrant, performs semantic search, and generates contextual answers using a local LLM.

## 🔧 Tech Stack
- LangChain
- Qdrant (local)
- Ollama (Mistral)
- Python
- BeautifulSoup (for scraping)
- Streamlit (optional UI)

## 🚀 Features
- Web scraping from user-defined URLs
- Text embedding using `OllamaEmbeddings`
- Vector search via Qdrant
- Contextual answering using local LLM
- Optional Streamlit-based frontend

## 🛠️ Setup
```bash
git clone https://github.com/yourusername/rag-assistant.git
cd rag-assistant
pip install -r requirements.txt
ollama run mistral
python app.py  # or streamlit run ui.py

## License
This project is licensed under the [MIT License](LICENSE).
