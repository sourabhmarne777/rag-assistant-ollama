# 🧠 Simple RAG Assistant

A clean, educational RAG (Retrieval-Augmented Generation) system that answers questions based on web content. Perfect for learning how RAG works!

## ✨ What It Does

1. **Ask Questions** - Enter any question you want answered
2. **Provide URLs** - Add web sources with relevant information  
3. **Get Smart Answers** - AI reads your sources and provides intelligent responses
4. **Monitor Storage** - Track your Qdrant Cloud usage with a beautiful progress bar

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone <your-repo>
cd rag-assistant
pip install -r requirements.txt

# 2. Setup environment
cp .env.example .env
# Edit .env with your Qdrant Cloud credentials

# 3. Start Ollama and pull models
ollama serve
ollama pull mistral
ollama pull nomic-embed-text

# 4. Run the app
python run.py
```

## 🔧 Configuration

Create a `.env` file with your Qdrant Cloud settings:

```env
# Get these from https://qdrant.tech/
QDRANT_URL=https://your-cluster.qdrant.tech:6333
QDRANT_API_KEY=your_api_key_here
COLLECTION_NAME=rag_documents

# Ollama settings (defaults work fine)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
EMBEDDING_MODEL=nomic-embed-text
```

## 📁 Simple Structure

```
rag-assistant/
├── app.py              # Streamlit UI (main app)
├── rag_pipeline.py     # Core RAG logic
├── web_scraper.py      # URL content extraction
├── vector_store.py     # Qdrant Cloud operations
├── embeddings.py       # Text to vector conversion
├── llm.py              # AI answer generation
├── settings.py         # Configuration
├── run.py              # Launcher script
└── requirements.txt    # Dependencies
```

## 🎯 Key Features

### **Smart Caching**
- URLs are processed once and stored forever
- No need to re-process the same content
- Efficient deduplication

### **Storage Monitoring** 
- Real-time Qdrant usage tracking
- Beautiful progress bar in the UI
- Warnings when approaching limits
- One-click data cleanup

### **Simple & Educational**
- Clean, flat file structure
- Well-commented code
- Easy to understand and modify
- Perfect for learning RAG concepts

## 💡 How It Works

1. **Web Scraping** - Extracts clean text from your URLs
2. **Embedding** - Converts text to vectors using `nomic-embed-text`
3. **Storage** - Saves vectors in Qdrant Cloud with metadata
4. **Search** - Finds relevant content using cosine similarity
5. **Generation** - Creates answers using Mistral AI

## 🎨 Beautiful UI

- **Clean Design** - Simple, focused interface
- **Storage Monitor** - Real-time usage tracking with progress bar
- **Smart Feedback** - Clear status messages and error handling
- **Responsive** - Works great on desktop and mobile

## 📊 Storage Management

The app includes a beautiful storage monitor that shows:
- **Usage Percentage** - How much of your free tier you've used
- **Document Count** - Number of stored documents
- **Status Warnings** - Alerts when approaching limits
- **Quick Cleanup** - One-click data reset option

## 🔒 Free Tier Friendly

- **Efficient Caching** - Process URLs only once
- **Smart Limits** - Respects Qdrant free tier (1M vectors)
- **Usage Tracking** - Always know where you stand
- **Easy Cleanup** - Manage storage when needed

## 🤝 Contributing

This is an educational project! Feel free to:
- Add new features
- Improve the UI
- Add support for PDFs/files
- Enhance error handling
- Submit issues and PRs

## 📄 License

MIT License - Use this code to learn and build amazing things!

---

**Built with ❤️ for the community to learn RAG concepts**