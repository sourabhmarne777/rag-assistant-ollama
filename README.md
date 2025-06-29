# 🤖 RAG Assistant with Ollama & Qdrant

A production-grade **Retrieval-Augmented Generation (RAG)** assistant that combines local AI models via **Ollama** with cloud-hosted vector storage using **Qdrant Cloud**. Built with **Streamlit** for an intuitive web interface.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![Ollama](https://img.shields.io/badge/ollama-latest-orange.svg)

## 🌟 Features

- **📁 Document Processing**: Upload and process PDF, TXT, and CSV files with intelligent text extraction
- **🌐 Web Scraping**: Extract content from web URLs using BeautifulSoup with smart content detection
- **🔍 Semantic Search**: Advanced vector similarity search with session-based filtering
- **💬 Interactive Chat**: Natural language querying with context-aware responses and source attribution
- **🏷️ Smart Tagging**: Metadata-based document organization and retrieval
- **🔒 Session Isolation**: Each session maintains separate document contexts for privacy
- **⚡ Local AI**: Fast inference using Ollama with Mistral model (privacy-focused)
- **☁️ Cloud Vector Store**: Scalable vector storage with Qdrant Cloud
- **🎨 Modern UI**: Clean, responsive interface with loading indicators and visual feedback
- **🚀 Real-time Processing**: Live document processing with progress indicators

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│  RAG Pipeline    │────│  Qdrant Cloud   │
│   - File Upload │    │  - Orchestration │    │  - Vector Store │
│   - Chat Interface│   │  - Session Mgmt  │    │  - Similarity   │
│   - URL Input   │    │  - Error Handling│    │    Search       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                    ┌──────────────────┐
                    │  Ollama (Local)  │
                    │  - Mistral LLM   │
                    │  - nomic-embed   │
                    │  - Privacy First │
                    └──────────────────┘
```

### 🔧 Core Components

1. **Document Processor** (`src/document_processor.py`): Handles PDF, TXT, and CSV file processing with multiple fallback methods
2. **Web Scraper** (`src/web_scraper.py`): Extracts clean content from web URLs with intelligent content detection
3. **Vector Store** (`src/vector_store.py`): Qdrant integration with session-based filtering and metadata indexing
4. **LLM Client** (`src/llm_client.py`): Ollama integration for local AI inference with error handling
5. **Embedding Client** (`src/embeddings.py`): Generates vector embeddings using nomic-embed-text model
6. **RAG Pipeline** (`src/rag_pipeline.py`): Orchestrates the entire retrieval-augmented generation flow

### 🏷️ Session-Based Architecture

Each document chunk is stored with comprehensive metadata:
```json
{
  "source_type": "document" | "web",
  "source_name": "filename_or_url",
  "session_id": "unique_session_identifier",
  "chunk_id": "chunk_index",
  "content": "actual_text_content"
}
```

When querying, only vectors matching the current session are retrieved, ensuring complete context isolation between different users or sessions.

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (Python 3.9+ recommended)
- **[Ollama](https://ollama.ai/)** installed and running
- **Qdrant Cloud** account (free tier available - 1GB storage, 100K vectors)
- **Git** for cloning the repository

## 📋 Installation Guide

### 🐧 Linux/macOS Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/rag-assistant-ollama.git
   cd rag-assistant-ollama
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   source venv/bin/activate  # Linux/macOS
   ```

3. **Install Python Dependencies**
   ```bash
   # Upgrade pip first
   pip install --upgrade pip
   
   # Install all required packages
   pip install -r requirements.txt
   ```

4. **Install and Setup Ollama**
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama service (keep this terminal open)
   ollama serve
   ```

5. **Download Required AI Models**
   ```bash
   # In a new terminal window, download the models
   ollama pull mistral          # Main language model (~4GB)
   ollama pull nomic-embed-text # Embedding model (~274MB)
   
   # Verify models are installed
   ollama list
   ```

6. **Configure Environment Variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit the .env file with your Qdrant credentials
   nano .env  # or use your preferred editor
   ```

7. **Run the Application**
   ```bash
   # Make sure your virtual environment is activated
   streamlit run app.py
   ```

### 🪟 Windows Setup

1. **Clone the Repository**
   ```cmd
   git clone https://github.com/yourusername/rag-assistant-ollama.git
   cd rag-assistant-ollama
   ```

2. **Create and Activate Virtual Environment**
   ```cmd
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   venv\Scripts\activate
   ```

3. **Install Python Dependencies**
   ```cmd
   # Upgrade pip first
   python -m pip install --upgrade pip
   
   # Install all required packages
   pip install -r requirements.txt
   ```

4. **Install and Setup Ollama**
   - Download Ollama from [ollama.ai](https://ollama.ai/download)
   - Install the downloaded executable
   - Open Command Prompt as Administrator and run:
   ```cmd
   ollama serve
   ```

5. **Download Required AI Models**
   ```cmd
   # In a new Command Prompt window
   ollama pull mistral
   ollama pull nomic-embed-text
   
   # Verify installation
   ollama list
   ```

6. **Configure Environment Variables**
   ```cmd
   # Copy the example file
   copy .env.example .env
   
   # Edit .env with Notepad or your preferred editor
   notepad .env
   ```

7. **Run the Application**
   ```cmd
   # Ensure virtual environment is activated
   streamlit run app.py
   ```

## ⚙️ Configuration

### 🔐 Environment Variables Setup

Create a `.env` file in the project root with the following configuration:

```env
# Qdrant Cloud Configuration (Required)
QDRANT_URL=https://your-cluster-url.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
COLLECTION_NAME=rag_documents

# Ollama Configuration (Local AI Models)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
EMBEDDING_MODEL=nomic-embed-text
```

### ☁️ Qdrant Cloud Setup

1. **Create Account**: Sign up at [Qdrant Cloud](https://cloud.qdrant.io/)
2. **Create Cluster**: 
   - Choose the **Free Tier** (1GB storage, 100K vectors)
   - Select your preferred region
   - Wait for cluster creation (usually 2-3 minutes)
3. **Get Credentials**:
   - Copy your **Cluster URL** (looks like: `https://xyz.qdrant.io`)
   - Copy your **API Key** from the cluster dashboard
4. **Update Configuration**: Add these credentials to your `.env` file

### 🤖 Ollama Model Configuration

The application uses two models:
- **mistral**: Main language model for generating responses (~4GB)
- **nomic-embed-text**: Embedding model for vector generation (~274MB)

You can change models by updating the `.env` file:
```env
OLLAMA_MODEL=llama2          # Alternative: llama2, codellama, etc.
EMBEDDING_MODEL=all-minilm   # Alternative embedding models
```

## 📖 Usage Guide

### 🚀 Starting the Application

1. **Activate Virtual Environment**:
   ```bash
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Start Ollama** (in separate terminal):
   ```bash
   ollama serve
   ```

3. **Launch Application**:
   ```bash
   streamlit run app.py
   ```

4. **Access Interface**: Open your browser to `http://localhost:8501`

### 📁 Adding Content

#### **Upload Documents**
- **Supported Formats**: PDF, TXT, CSV
- **Multiple Files**: Upload several files at once
- **Auto-Processing**: Files are automatically processed when uploaded
- **Progress Tracking**: Visual progress bar shows processing status

#### **Add Web Content**
- **Enter URL**: Paste any web URL in the input field
- **Click "Add URL"**: Button processes the content
- **Smart Extraction**: Automatically extracts main content, ignoring navigation and ads
- **Loading Indicator**: Shows progress while scraping and processing

### 💬 Chatting with Your Data

1. **Ask Questions**: Type natural language questions about your content
2. **Press Enter**: Submit questions using Enter key or the Send button
3. **View Sources**: See which documents contributed to each answer
4. **Session Context**: All questions are answered within your current session's context

### 🔄 Session Management

- **Automatic Sessions**: Each browser session gets a unique ID
- **Clear All**: Use the "Clear All" button to start fresh
- **Data Isolation**: Your documents are never mixed with other users' data

## 🎯 Example Use Cases

### 📚 Research Assistant
```
Upload: Research papers (PDFs), articles (URLs)
Ask: "What are the main findings across these studies?"
```

### 📊 Business Intelligence
```
Upload: Reports (PDFs), company data (CSV)
Ask: "What trends do you see in our quarterly data?"
```

### 📖 Learning Companion
```
Upload: Textbooks (PDFs), online tutorials (URLs)
Ask: "Explain the key concepts from chapter 3"
```

### 🔍 Content Analysis
```
Upload: Multiple documents on a topic
Ask: "Compare the different perspectives presented"
```

## 🛠️ Development

### 📁 Project Structure

```
rag-assistant-ollama/
├── app.py                    # Main Streamlit application with UI
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore patterns
├── README.md                # This comprehensive guide
├── LICENSE                  # MIT License
└── src/                     # Source code modules
    ├── __init__.py          # Package initialization
    ├── rag_pipeline.py      # Main RAG orchestration logic
    ├── vector_store.py      # Qdrant integration and vector operations
    ├── llm_client.py        # Ollama LLM client with error handling
    ├── embeddings.py        # Embedding generation using nomic-embed-text
    ├── document_processor.py # Document processing with multiple formats
    └── web_scraper.py       # Web content extraction and cleaning
```

### 🔧 Key Design Decisions

1. **Session-Based Isolation**: Each user session maintains separate document contexts using unique session IDs
2. **Modular Architecture**: Clear separation of concerns for maintainability and testing
3. **Comprehensive Error Handling**: Graceful degradation and user-friendly error messages
4. **Scalable Storage**: Cloud-based vector storage for production scalability
5. **Privacy-Focused**: Local AI model inference keeps your data private
6. **Responsive UI**: Modern, clean interface that works on all devices

### 🧪 Testing Your Setup

1. **Test Ollama Connection**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. **Test with Sample Content**:
   - Upload a simple text file
   - Add a Wikipedia URL
   - Ask: "What is this content about?"

3. **Verify Vector Storage**:
   - Check Qdrant Cloud dashboard for stored vectors
   - Verify session isolation by clearing and re-adding content

## 🚀 Production Deployment

### 🐳 Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### ☁️ Cloud Deployment Options

- **Streamlit Cloud**: Direct deployment from GitHub
- **Heroku**: Easy deployment with buildpacks
- **AWS/GCP/Azure**: Full control with container services
- **Railway/Render**: Simple deployment platforms

### 🔒 Production Considerations

1. **Environment Security**: Use proper secret management
2. **Rate Limiting**: Implement API rate limiting
3. **Monitoring**: Add logging and health checks
4. **Scaling**: Consider load balancing for multiple users
5. **Backup**: Regular backup of vector data

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the Repository**
2. **Create Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Make Changes**: Follow the existing code style
4. **Add Tests**: Ensure your changes are tested
5. **Commit Changes**: `git commit -m 'Add amazing feature'`
6. **Push to Branch**: `git push origin feature/amazing-feature`
7. **Open Pull Request**: Describe your changes clearly

### 📝 Development Guidelines

- **Code Style**: Follow PEP 8 for Python code
- **Comments**: Add clear comments for complex logic
- **Error Handling**: Include comprehensive error handling
- **Documentation**: Update README for new features
- **Testing**: Test with various file types and URLs

## 🐛 Troubleshooting

### Common Issues and Solutions

#### **Ollama Connection Issues**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve

# Check available models
ollama list
```

#### **PDF Processing Errors**
- **Issue**: "No text could be extracted from PDF"
- **Solution**: PDF might be image-based. Try converting to text-based PDF first
- **Alternative**: Use OCR tools like Tesseract for image-based PDFs

#### **Qdrant Connection Issues**
- **Check Credentials**: Verify URL and API key in `.env`
- **Network Issues**: Ensure internet connectivity
- **Quota Limits**: Check if you've exceeded free tier limits

#### **Memory Issues**
- **Large Files**: Break large documents into smaller chunks
- **Model Size**: Consider using smaller models like `llama2:7b`
- **System Resources**: Ensure adequate RAM (8GB+ recommended)

### 📊 Performance Optimization

1. **Chunk Size**: Adjust `chunk_size` in document processor for your use case
2. **Model Selection**: Use smaller models for faster inference
3. **Batch Processing**: Process multiple documents together
4. **Caching**: Implement caching for frequently accessed content

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- ✅ **Commercial Use**: Use in commercial projects
- ✅ **Modification**: Modify the code as needed
- ✅ **Distribution**: Share and distribute freely
- ✅ **Private Use**: Use privately without restrictions
- ❗ **Liability**: No warranty provided
- ❗ **Attribution**: Include original license and copyright

## 🙏 Acknowledgments

- **[Ollama](https://ollama.ai/)** - For making local AI models accessible and easy to use
- **[Qdrant](https://qdrant.tech/)** - For providing excellent vector database technology
- **[Streamlit](https://streamlit.io/)** - For the amazing web interface framework
- **[LangChain](https://langchain.com/)** - For RAG pipeline components and utilities
- **Open Source Community** - For the countless libraries that make this project possible

## 📞 Support & Community

### Getting Help
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Join community discussions on GitHub
- **Documentation**: Check this README and code comments
- **Examples**: Review the sample configurations and use cases

### Stay Updated
- **Star the Repository** ⭐ to stay updated with new features
- **Watch Releases** 👀 for important updates
- **Follow the Project** 📢 for announcements

### Contributing to the Community
- **Share Your Use Cases**: Help others learn from your implementations
- **Report Issues**: Help improve the project by reporting bugs
- **Suggest Features**: Share ideas for new functionality
- **Write Tutorials**: Create guides for specific use cases

---

**Built with ❤️ for the AI/ML community**

*This project demonstrates the power of combining local AI models with cloud vector storage to create privacy-focused, scalable RAG applications. Perfect for learning, research, and production use cases.*

---

### 🔗 Quick Links

- [Ollama Installation](https://ollama.ai/download)
- [Qdrant Cloud Signup](https://cloud.qdrant.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
- [Git Installation](https://git-scm.com/downloads)

### 📈 Project Stats

- **Language**: Python 3.8+
- **Framework**: Streamlit
- **AI Models**: Ollama (Mistral, nomic-embed-text)
- **Vector DB**: Qdrant Cloud
- **License**: MIT
- **Maintenance**: Actively maintained