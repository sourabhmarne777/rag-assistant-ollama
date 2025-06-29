#!/usr/bin/env python3
"""
RAG Assistant Chatbot Runner
"""

import os
import sys
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import streamlit
        import langchain
        import qdrant_client
        import PyPDF2
        import requests
        import bs4
        logger.info("All required packages are available")
        return True
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        return False

def install_requirements():
    """Install required packages"""
    try:
        logger.info("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False

def check_environment():
    """Check environment variables"""
    required_vars = ['QDRANT_URL', 'QDRANT_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.info("Please check your .env file and ensure all required variables are set")
        return False
    
    logger.info("Environment variables are properly configured")
    return True

def run_streamlit_app():
    """Run the Streamlit application"""
    try:
        logger.info("Starting RAG Assistant Chatbot...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port=8501"])
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Error running application: {e}")

def main():
    """Main function"""
    print("🤖 RAG Assistant Chatbot")
    print("=" * 50)
    
    # Check if requirements are installed
    if not check_requirements():
        print("Installing missing requirements...")
        if not install_requirements():
            print("❌ Failed to install requirements. Please install manually using:")
            print("pip install -r requirements.txt")
            sys.exit(1)
    
    # Check environment configuration
    if not check_environment():
        print("❌ Environment configuration incomplete")
        print("Please ensure your .env file contains:")
        print("- QDRANT_URL")
        print("- QDRANT_API_KEY")
        print("- COLLECTION_NAME (optional)")
        print("- OLLAMA_BASE_URL (optional)")
        print("- OLLAMA_MODEL (optional)")
        print("- EMBEDDING_MODEL (optional)")
        sys.exit(1)
    
    print("✅ All checks passed!")
    print("🚀 Starting the application...")
    print("📱 The app will open in your browser at: http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    print("-" * 50)
    
    # Run the application
    run_streamlit_app()

if __name__ == "__main__":
    main()