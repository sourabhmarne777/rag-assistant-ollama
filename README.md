# RAG Chat Assistant

A professional, production-ready RAG (Retrieval-Augmented Generation) system that provides intelligent answers based on your documents and web sources. Perfect for research, documentation analysis, and knowledge management.

## Features

- **Document Upload** - Support for TXT, PDF, and DOCX files
- **Web Content** - Extract information from any URL
- **Smart AI Chat** - Context-aware responses using your sources
- **Efficient Storage** - Qdrant Cloud vector database with usage monitoring
- **Beautiful UI** - Clean, professional interface built with Streamlit
- **Context Management** - Smart deduplication and relevance filtering

## Quick Start

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- [Qdrant Cloud](https://qdrant.tech/) account (free tier available)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sourabhmarne777/rag-assistant-ollama.git
   cd rag-assistant-ollama
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Ollama models** (one-time setup)
   ```bash
   # Start Ollama service
   ollama serve
   
   # Pull required models
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

4. **Configure environment**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env with your settings (see Configuration section)
   nano .env
   ```

5. **Run the application**
   ```bash
   python run.py
   ```

6. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Start chatting with your AI assistant!

## Configuration

Create a `.env` file with the following settings:

### Required Settings

```env
# Qdrant Cloud Configuration
QDRANT_URL=https://your-cluster-url.qdrant.tech:6333
QDRANT_API_KEY=your_qdrant_api_key_here
COLLECTION_NAME=rag_documents
```

**Getting Qdrant Cloud credentials:**
1. Sign up at [qdrant.tech](https://qdrant.tech/)
2. Create a new cluster (free tier: 1GB storage, 1M vectors)
3. Copy your cluster URL and API key
4. Paste them into your `.env` file

### Optional Settings

```env
# Ollama Configuration (defaults work fine)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
EMBEDDING_MODEL=nomic-embed-text

# Text Processing
MAX_TEXT_LENGTH=15000
CHUNK_SIZE=1000
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.3

# Storage Management
FREE_TIER_LIMIT=1000000
STORAGE_WARNING_THRESHOLD=80
STORAGE_CRITICAL_THRESHOLD=90
```

## How to Use

### 1. Upload Documents
- Click "Choose files" in the sidebar
- Select TXT, PDF, or DOCX files
- Files are automatically processed and stored

### 2. Add Web Sources
- Enter URLs in the "Add URLs" section
- Click "Add URL" to process web content
- Content is extracted and stored for future queries

### 3. Chat with AI
- Type your questions in the chat input
- AI will use your uploaded sources for context-aware answers
- Sources are automatically cited in responses

### 4. Manage Context
- **Clear Chat**: Remove conversation history only
- **Reset All**: Clear everything (chat + uploaded sources)
- **Remove Items**: Click × next to files/URLs to remove them

## Architecture

```
rag-assistant-ollama/
├── app.py              # Main Streamlit application
├── rag_pipeline.py     # Core RAG logic and orchestration
├── web_scraper.py      # Web content extraction
├── vector_store.py     # Qdrant Cloud operations
├── embeddings.py       # Text-to-vector conversion
├── llm.py              # AI response generation
├── file_processor.py   # Document processing (PDF, DOCX, TXT)
├── settings.py         # Configuration management
├── styles.css          # UI styling
├── run.py              # Application launcher
└── requirements.txt    # Python dependencies
```

## How It Works

1. **Document Processing**: Files and URLs are processed to extract clean text
2. **Embedding Generation**: Text is converted to vectors using `nomic-embed-text`
3. **Vector Storage**: Embeddings are stored in Qdrant Cloud with metadata
4. **Semantic Search**: User queries are matched against stored vectors
5. **Context Assembly**: Relevant sources are combined for AI context
6. **Response Generation**: Mistral AI generates answers using retrieved context

## Key Features

### Smart Context Management
- **Deduplication**: Identical documents are processed only once
- **Relevance Filtering**: Only truly relevant sources are used for answers
- **Source Attribution**: All responses include source citations

### Storage Monitoring
- **Real-time Usage**: Track your Qdrant Cloud storage usage
- **Visual Indicators**: Progress bars and status alerts
- **Efficient Caching**: URLs and files are processed once and reused

### Professional UI
- **Clean Design**: Modern, responsive interface
- **Intuitive Controls**: Easy file upload and URL management
- **Context Awareness**: Clear indicators of active sources

## Customization

### Change AI Models
Edit `.env` to use different Ollama models:
```env
OLLAMA_MODEL=llama2          # or codellama, neural-chat, etc.
EMBEDDING_MODEL=all-minilm   # or other embedding models
```

### Adjust Processing Limits
```env
MAX_TEXT_LENGTH=20000        # Increase for longer documents
CHUNK_SIZE=1500             # Larger chunks for more context
TOP_K_RESULTS=10            # More search results
```

### Storage Thresholds
```env
STORAGE_WARNING_THRESHOLD=70    # Warning at 70% usage
STORAGE_CRITICAL_THRESHOLD=85   # Critical at 85% usage
```

## Privacy & Security

- **Local Processing**: All AI processing happens locally via Ollama
- **Secure Storage**: Documents are stored securely in Qdrant Cloud
- **No Data Sharing**: Your data never leaves your control
- **API Keys**: Stored locally in `.env` file (never committed to git)

## Performance Tips

### For Better Results
- **Quality Sources**: Use authoritative, well-written documents
- **Relevant Content**: Upload documents related to your questions
- **Clear Questions**: Ask specific, well-formed questions

### For Better Performance
- **Manage Storage**: Regularly clean up unused sources
- **Optimize Chunks**: Adjust `CHUNK_SIZE` for your document types
- **Monitor Usage**: Keep an eye on Qdrant storage limits

## Troubleshooting

### Common Issues

**Ollama not responding**
```bash
# Check if Ollama is running
ollama list

# Restart Ollama if needed
ollama serve
```

**Models not found**
```bash
# Pull required models
ollama pull mistral
ollama pull nomic-embed-text
```

**Qdrant connection failed**
- Verify your `QDRANT_URL` and `QDRANT_API_KEY` in `.env`
- Check your internet connection
- Ensure your Qdrant cluster is active

**File upload issues**
- Check file format (TXT, PDF, DOCX only)
- Ensure file size is reasonable (< 50MB recommended)
- Verify file is not corrupted

## Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and test thoroughly
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to the branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/rag-assistant-ollama.git
cd rag-assistant-ollama

# Install development dependencies
pip install -r requirements.txt

# Run in development mode
streamlit run app.py --server.runOnSave=true
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **[Ollama](https://ollama.ai/)** - Local AI model serving
- **[Qdrant](https://qdrant.tech/)** - Vector database and search
- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[LangChain](https://langchain.com/)** - AI application development

## Support

- **Issues**: [GitHub Issues](https://github.com/sourabhmarne777/rag-assistant-ollama/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sourabhmarne777/rag-assistant-ollama/discussions)
- **Documentation**: This README and inline code comments

---

**Built with care for the community**

*Transform your documents into intelligent conversations*