# RAG Assistant Chatbot

A powerful RAG (Retrieval-Augmented Generation) chatbot built with Streamlit that can process documents, scrape web content, and provide contextual answers using Qdrant vector database and Ollama models.

## Features

- 📄 **Document Processing**: Upload and process PDF, TXT, and DOCX files
- 🌐 **Web Scraping**: Extract content from URLs and add to knowledge base
- 🤖 **AI-Powered Chat**: Chat with your documents using Ollama models
- 🔍 **Semantic Search**: Find relevant information using vector similarity
- 📊 **Knowledge Base Management**: Track and manage your document collection
- 🎨 **User-Friendly Interface**: Clean, intuitive Streamlit interface

## Technologies Used

- **Frontend**: Streamlit
- **LLM**: Ollama (Mistral, Llama2, etc.)
- **Vector Database**: Qdrant Cloud
- **Document Processing**: PyPDF2, python-docx
- **Web Scraping**: BeautifulSoup4, Requests
- **Embeddings**: Ollama embeddings (nomic-embed-text)
- **Framework**: LangChain

## Prerequisites

1. **Qdrant Cloud Account**: Sign up at [Qdrant Cloud](https://cloud.qdrant.io/)
2. **Ollama**: Install and run Ollama locally
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull required models
   ollama pull mistral
   ollama pull nomic-embed-text
   
   # Start Ollama server
   ollama serve
   ```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag-assistant-chatbot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   Update the `.env` file with your Qdrant credentials:
   ```env
   # Qdrant Cloud Configuration (Required)
   QDRANT_URL=your_qdrant_cluster_url
   QDRANT_API_KEY=your_qdrant_api_key
   COLLECTION_NAME=rag_documents

   # Ollama Configuration (Local AI Models)
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=mistral
   EMBEDDING_MODEL=nomic-embed-text
   ```

## Usage

1. **Start the application**:
   ```bash
   python run.py
   ```
   Or directly with Streamlit:
   ```bash
   streamlit run app.py
   ```

2. **Access the application**:
   Open your browser and go to `http://localhost:8501`

3. **Add content to your knowledge base**:
   - **Upload Documents**: Use the sidebar to upload PDF, TXT, or DOCX files
   - **Add Web Content**: Enter URLs to scrape and add web content

4. **Start chatting**:
   - Ask questions about your uploaded documents
   - Get contextual answers based on your knowledge base
   - View sources and relevance scores for each response

## Project Structure

```
rag-assistant-chatbot/
├── app.py                 # Main Streamlit application
├── run.py                 # Application runner with setup checks
├── requirements.txt       # Python dependencies
├── .env                  # Environment configuration
├── README.md             # This file
└── src/
    ├── __init__.py       # Package initialization
    ├── rag_system.py     # Core RAG system implementation
    ├── document_processor.py  # Document processing utilities
    ├── web_scraper.py    # Web scraping functionality
    └── chat_interface.py # Chat interface utilities
```

## Configuration Options

### Model Settings
- **Ollama Model**: Choose from mistral, llama2, codellama, neural-chat
- **Embedding Model**: Select embedding model for vector generation

### RAG Settings
- **Similarity Threshold**: Minimum similarity score for relevant documents (0.0-1.0)
- **Max Results**: Maximum number of relevant documents to retrieve (1-10)

### Document Processing
- **Chunk Size**: Size of text chunks for processing (default: 1000)
- **Chunk Overlap**: Overlap between chunks (default: 200)

## Supported File Types

- **PDF**: Portable Document Format files
- **TXT**: Plain text files
- **DOCX**: Microsoft Word documents

## Web Scraping

The application can extract content from web pages by:
- Removing navigation, ads, and other non-content elements
- Extracting main article/content areas
- Cleaning and normalizing text
- Handling various website structures

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**:
   - Ensure Ollama is running: `ollama serve`
   - Check if models are installed: `ollama list`
   - Verify OLLAMA_BASE_URL in .env file

2. **Qdrant Connection Error**:
   - Verify QDRANT_URL and QDRANT_API_KEY in .env file
   - Check Qdrant Cloud dashboard for cluster status

3. **Document Processing Error**:
   - Ensure uploaded files are not corrupted
   - Check file size limits
   - Verify file format is supported

4. **Web Scraping Issues**:
   - Some websites may block scraping
   - Check if URL is accessible
   - Verify internet connection

### Performance Tips

- Use smaller chunk sizes for more precise retrieval
- Adjust similarity threshold based on your use case
- Regularly clear unused documents from knowledge base
- Monitor Qdrant storage usage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [LangChain](https://langchain.com/) for RAG implementation tools
- [Qdrant](https://qdrant.tech/) for vector database capabilities
- [Ollama](https://ollama.ai/) for local LLM inference
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) for web scraping

## Support

If you encounter any issues or have questions, please:
1. Check the troubleshooting section above
2. Review the logs for error messages
3. Open an issue on the repository
4. Provide detailed error information and steps to reproduce