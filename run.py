import sys
import subprocess
import os
from dotenv import load_dotenv

def check_env_file():
    """Check if .env file exists and has required variables"""
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("Please copy .env.example to .env and configure your Qdrant Cloud settings:")
        print("cp .env.example .env")
        return False
    
    load_dotenv()
    
    required_vars = ['QDRANT_URL', 'QDRANT_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please configure your .env file with Qdrant Cloud credentials")
        return False
    
    return True

def check_ollama():
    """Check if Ollama is running"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def start_streamlit():
    """Start Streamlit UI"""
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'src/ui/streamlit_app.py',
            '--server.port=8501',
            '--server.address=0.0.0.0'
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

def main():
    print("🧠 RAG Assistant")
    print("=" * 30)
    
    # Check environment
    if not check_env_file():
        return
    
    # Check Ollama
    if not check_ollama():
        print("❌ Ollama is not running!")
        print("Please start Ollama and ensure models are available:")
        print("1. Start: ollama serve")
        print("2. Pull models: ollama pull mistral && ollama pull nomic-embed-text")
        return
    
    print("✅ Environment configured")
    print("✅ Ollama is running")
    print("✅ Qdrant Cloud ready")
    
    print("\n🚀 Starting RAG Assistant...")
    print("📱 Open http://localhost:8501 in your browser")
    start_streamlit()

if __name__ == "__main__":
    main()
