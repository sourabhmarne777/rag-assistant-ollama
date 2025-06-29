# RAG Assistant with Ollama & Qdrant

A production-grade Retrieval-Augmented Generation (RAG) assistant that combines local AI models via Ollama with cloud-hosted vector storage using Qdrant. Built with Streamlit for an intuitive web interface.

## 🚀 Features

- **📁 Document Processing**: Upload and process PDF, TXT, and CSV files
- **🌐 Web Scraping**: Extract content from web URLs using BeautifulSoup
- **🔍 Semantic Search**: Advanced vector similarity search with session-based filtering
- **💬 Interactive Chat**: Natural language querying with context-aware responses
- **🏷️ Smart Tagging**: Metadata-based document organization and retrieval
- **🔒 Session Isolation**: Each session maintains separate document contexts
- **⚡ Local AI**: Fast inference using Ollama with Mistral model
- **☁️ Cloud Vector Store**: Scalable vector storage with Qdrant Cloud

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│  RAG Pipeline    │────│  Qdrant Cloud   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                    ┌──────────────────┐
                    │  Ollama (Local)  │
                    │  - Mistral LLM   │
                    │  - Embeddings    │
                    └──────────────────┘
```

### Core Components

1. **Document Processor**: Handles PDF, TXT, and CSV file processing
2. **Web Scraper**: Extracts clean content from web URLs
3. **Vector Store**: Qdrant integration with session-based filtering
4. **LLM Client**: Ollama integration for local AI inference
5. **Embedding Client**: Generates vector embeddings using nomic-embed-text
6. **RAG Pipeline**: Orchestrates the entire retrieval-augmented generation flow

### Tag-Based Filtering Logic

Each document chunk is stored with metadata:
```json
{
  "source_type": "document" | "web",
  "source_name": "filename_or_url",
  "session_id": "unique_session_identifier",
  "chunk_id": "chunk_index"
}
```

When querying, only vectors matching the current session are retrieved, ensuring context isolation.

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Qdrant Cloud account (free tier available)

### Linux Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/sourabhmarne777/rag-assistant-ollama.git
   cd rag-assistant-ollama
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Ollama**
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama service
   ollama serve
   
   # In another terminal, pull required models
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

5. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your Qdrant credentials
   ```

6. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Windows Setup

1. **Clone the repository**
   ```cmd
   git clone https://github.com/sourabhmarne777/rag-assistant-ollama.git
   cd rag-assistant-ollama
   ```

2. **Create virtual environment**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```cmd
   pip install -r requirements.txt
   ```

4. **Setup Ollama**
   - Download and install Ollama from [ollama.ai](https://ollama.ai/)
   - Open Command Prompt and run:
   ```cmd
   ollama serve
   ```
   - In another Command Prompt:
   ```cmd
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

5. **Configure environment**
   ```cmd
   copy .env.example .env
   # Edit .env with your Qdrant credentials using notepad or your preferred editor
   ```

6. **Run the application**
   ```cmd
   streamlit run app.py
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Qdrant Cloud Configuration
QDRANT_URL=https://your-cluster-url.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
COLLECTION_NAME=rag_documents

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
EMBEDDING_MODEL=nomic-embed-text
```

### Qdrant Cloud Setup

1. Sign up at [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create a new cluster (free tier: 1GB storage, 100K vectors)
3. Get your cluster URL and API key
4. Update the `.env` file with your credentials

## 📖 Usage

1. **Start the application**: `streamlit run app.py`
2. **Upload documents**: Use the file uploader to add PDF, TXT, or CSV files
3. **Scrape web content**: Enter URLs to extract and process web content
4. **Chat with your data**: Ask questions about the uploaded/scraped content
5. **Session management**: Use "New Session" to start fresh with different documents

### Sample Documents and URLs

Try these examples to test the system:

**Sample URLs:**
- https://en.wikipedia.org/wiki/Artificial_intelligence
- https://docs.python.org/3/tutorial/
- https://streamlit.io/

**Sample Questions:**
- "What are the main topics discussed in the documents?"
- "Summarize the key points from the uploaded content"
- "What specific information is available about [topic]?"

## 🔧 Development

### Project Structure

```
rag-assistant-ollama/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # This file
├── LICENSE               # MIT License
└── src/                  # Source code modules
    ├── __init__.py
    ├── rag_pipeline.py    # Main RAG orchestration
    ├── vector_store.py    # Qdrant integration
    ├── llm_client.py      # Ollama LLM client
    ├── embeddings.py      # Embedding generation
    ├── document_processor.py # Document processing
    └── web_scraper.py     # Web content extraction
```

### Key Design Decisions

1. **Session-based Isolation**: Each user session maintains separate document contexts
2. **Modular Architecture**: Clear separation of concerns for maintainability
3. **Error Handling**: Comprehensive error handling and user feedback
4. **Scalable Storage**: Cloud-based vector storage for production use
5. **Local AI**: Privacy-focused local model inference

## 🚀 Deployment

For production deployment:

1. **Docker**: Create a Dockerfile for containerized deployment
2. **Cloud Hosting**: Deploy on AWS, GCP, or Azure
3. **Environment**: Use production-grade environment variable management
4. **Monitoring**: Add logging and monitoring for production use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local AI model serving
- [Qdrant](https://qdrant.tech/) for vector database technology
- [Streamlit](https://streamlit.io/) for the web interface framework
- [LangChain](https://langchain.com/) for RAG pipeline components

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the documentation
- Review the sample configurations

---

**Built with ❤️ for the AI/ML community**