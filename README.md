# 🧠 RAG Assistant

A production-ready Retrieval-Augmented Generation (RAG) system using **Qdrant Cloud**, **Ollama**, and **LangChain** for intelligent question-answering with web content.

## 🌟 What This Does

This RAG assistant helps you get answers to questions by:

1. **Web Scraping** - Extracts content from URLs you provide
2. **Vector Embedding** - Converts text to vectors using configurable embedding models
3. **Cloud Storage** - Stores vectors in Qdrant Cloud with smart caching
4. **Semantic Search** - Finds relevant content using cosine similarity
5. **AI Generation** - Uses your chosen Ollama model to generate contextual answers

## 🚀 One-Command Setup

```bash
# 1. Clone and setup
git clone <your-repo-url>
cd rag-assistant
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your Qdrant Cloud credentials

# 3. Start Ollama and pull models
ollama serve
ollama pull mistral
ollama pull nomic-embed-text

# 4. Launch the app
python run.py
```

## ⚙️ Configuration

### Required: Qdrant Cloud Setup

1. **Create Account**: Sign up at [qdrant.tech](https://qdrant.tech/)
2. **Create Cluster**: Set up a new cluster in the dashboard
3. **Get Credentials**: Copy your cluster URL and API key
4. **Configure .env**:

```env
# Qdrant Cloud Configuration (Required)
QDRANT_URL=https://your-cluster-url.qdrant.tech:6333
QDRANT_API_KEY=your_qdrant_api_key_here
COLLECTION_NAME=rag_documents

# Ollama Configuration (Customizable)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
EMBEDDING_MODEL=nomic-embed-text

# Text Processing
MAX_TEXT_LENGTH=10000
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Search Configuration
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
```

### Ollama Model Options

You can use any Ollama model by changing `OLLAMA_MODEL` in `.env`:

```bash
# Available models (pull first with ollama pull <model>)
OLLAMA_MODEL=mistral          # Default - Good balance (7B)
OLLAMA_MODEL=llama2           # Alternative option (7B)
OLLAMA_MODEL=codellama        # For code-related questions (7B)
OLLAMA_MODEL=neural-chat      # Conversational model (7B)
OLLAMA_MODEL=llama2:13b       # Larger model (needs more RAM)
OLLAMA_MODEL=llama2:70b       # Largest model (needs significant RAM)
OLLAMA_MODEL=phi              # Lightweight model (2.7B)
OLLAMA_MODEL=gemma           # Google's model (7B)
```

### Embedding Model Options

```bash
# Embedding models for vector generation
EMBEDDING_MODEL=nomic-embed-text    # Default - Good general purpose
EMBEDDING_MODEL=all-minilm          # Lightweight alternative
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Scraper   │───▶│   Embeddings     │───▶│  Qdrant Cloud   │
│  (BeautifulSoup)│    │ (configurable)   │    │  Vector Store   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│   User Query    │───▶│   Query Vector   │─────────────┘
└─────────────────┘    └──────────────────┘             │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Final Answer   │◀───│  Ollama LLM      │◀───│ Relevant Context│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🎯 Key Features

### **Smart Caching**
- URLs are processed once and stored permanently in Qdrant Cloud
- Subsequent queries reuse existing vectors for instant responses
- Efficient deduplication using URL-based UUIDs

### **Context Window Management**
- Automatically truncates content to fit model limits
- Prioritizes most relevant content by similarity score
- Maintains source attribution and relevance scores

### **Efficient Vector Operations**
- Uses cosine similarity for semantic matching
- Configurable similarity thresholds via environment variables
- Returns top-k most relevant documents with scores

### **LangChain Integration**
- Uses official `langchain` and `langchain-ollama` packages
- Proper embedding and LLM interfaces with better stability
- Structured prompt engineering for optimal results

### **Production Ready**
- Comprehensive error handling and logging
- Environment-based configuration
- Streamlit web interface with modern design
- One-command startup script

## 📁 Project Structure

```
rag-assistant/
├── config/
│   └── settings.py          # Environment-based configuration
├── src/
│   ├── embeddings/
│   │   └── embedder.py      # LangChain Ollama embeddings
│   ├── llm/
│   │   └── ollama_client.py # LangChain Ollama LLM
│   ├── scraping/
│   │   └── web_scraper.py   # Web content extraction
│   ├── vector_store/
│   │   └── qdrant_client.py # Qdrant Cloud operations
│   ├── rag/
│   │   └── pipeline.py      # Main RAG orchestration
│   └── ui/
│       └── streamlit_app.py # Web interface
├── .env.example             # Environment template
├── run.py                   # One-command launcher
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🔧 Usage Examples

### Basic Usage
1. Start the app: `python run.py`
2. Open http://localhost:8501
3. Enter your question
4. Provide relevant URLs
5. Get AI-generated answers with source attribution

### Example Query
- **Question**: "What are the key features of Python 3.12?"
- **URLs**: 
  - https://docs.python.org/3.12/whatsnew/3.12.html
  - https://realpython.com/python312-new-features/
- **Result**: Contextual answer with relevance scores and sources

### Advanced Configuration Examples

```env
# High-precision setup
SIMILARITY_THRESHOLD=0.8
TOP_K_RESULTS=3
MAX_TEXT_LENGTH=15000

# Fast processing setup
SIMILARITY_THRESHOLD=0.6
TOP_K_RESULTS=10
CHUNK_SIZE=500

# Large context setup (for bigger models)
MAX_TEXT_LENGTH=20000
CHUNK_SIZE=2000
CHUNK_OVERLAP=400
```

## 🛠️ Advanced Configuration

### Custom Collection Names
```env
# Use descriptive collection names for different projects
COLLECTION_NAME=company_docs
COLLECTION_NAME=research_papers
COLLECTION_NAME=blog_articles
```

### Performance Tuning
- **RAM**: 8GB minimum, 16GB recommended for larger models
- **Models**: Use smaller models (mistral, phi) for faster responses
- **Network**: Stable connection for Qdrant Cloud operations
- **Context**: Adjust `MAX_TEXT_LENGTH` based on your model's capabilities

## 🔍 Troubleshooting

### Common Issues

**Environment Variables**
```bash
# Check if .env is configured
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('QDRANT_URL:', os.getenv('QDRANT_URL'))"
```

**Ollama Connection**
```bash
# Check if Ollama is running
ollama list

# Start Ollama if needed
ollama serve

# Pull required models
ollama pull mistral
ollama pull nomic-embed-text
```

**Qdrant Cloud Connection**
- Verify cluster URL and API key in Qdrant dashboard
- Check if cluster is active and accessible
- Ensure API key has proper permissions

**Model Issues**
```bash
# List available models
ollama list

# Pull specific models
ollama pull llama2
ollama pull phi
```

**LangChain Issues**
```bash
# Reinstall dependencies
pip install --upgrade langchain langchain-ollama
```

## 📊 System Requirements

- **Python**: 3.8+
- **RAM**: 8GB minimum (16GB for larger models like llama2:13b)
- **Storage**: 5GB for models and cache
- **Network**: Stable internet for Qdrant Cloud and web scraping
- **Ollama**: Latest version with required models

## 🤝 Dependencies

- **langchain**: Core LangChain framework for LLM orchestration
- **langchain-ollama**: Official Ollama integration for LangChain
- **qdrant-client**: Vector database operations
- **beautifulsoup4**: Web scraping and content extraction
- **streamlit**: Modern web interface
- **requests**: HTTP operations
- **python-dotenv**: Environment variable management
- **numpy**: Numerical operations

## 🚀 What Makes This Better

### **LangChain Core Benefits**
- ✅ **Stability**: Official LangChain package, not community version
- ✅ **Performance**: Faster imports, smaller footprint
- ✅ **Reliability**: Better error handling and type safety
- ✅ **Future-proof**: Official support and updates

### **Qdrant Cloud Efficiency**
- ✅ **Smart Caching**: URLs processed once, reused forever
- ✅ **Cosine Similarity**: Efficient semantic search
- ✅ **Scalability**: Cloud-based, handles large datasets
- ✅ **Persistence**: Data survives application restarts

### **Context Window Optimization**
- ✅ **Intelligent Truncation**: Prioritizes relevant content
- ✅ **Source Attribution**: Tracks relevance scores
- ✅ **Memory Efficient**: Configurable limits via environment

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with ❤️ using LangChain, Qdrant Cloud, and Ollama**