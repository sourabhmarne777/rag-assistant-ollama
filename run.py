#!/usr/bin/env python3
"""
Simple RAG Assistant Launcher
"""

import sys
import subprocess
import os
from dotenv import load_dotenv

def check_requirements():
    """Check if everything is set up correctly"""
    print("🧠 RAG Assistant - Simple Setup Check")
    print("=" * 40)
    
    # Check .env file
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("Please copy .env.example to .env and add your Qdrant credentials")
        return False
    
    load_dotenv()
    
    # Check required variables
    if not os.getenv('QDRANT_URL') or not os.getenv('QDRANT_API_KEY'):
        print("❌ Missing Qdrant credentials in .env file!")
        return False
    
    # Check Ollama
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Ollama is not running!")
            print("Please start: ollama serve")
            return False
    except FileNotFoundError:
        print("❌ Ollama not found!")
        return False
    
    print("✅ Environment configured")
    print("✅ Ollama is running")
    print("✅ Ready to start!")
    return True

def main():
    if not check_requirements():
        return
    
    print("\n🚀 Starting RAG Assistant...")
    print("📱 Open http://localhost:8501 in your browser")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port=8501',
            '--server.address=localhost'
        ])
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()