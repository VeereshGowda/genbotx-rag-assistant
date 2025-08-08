"""
Setup script for GenBotX
Handles initial setup and testing of the RAG system.
"""

import sys
from pathlib import Path
import subprocess
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.wikipedia_scraper import WikipediaScraper
from src.document_processor import DocumentProcessor
from src.vector_store_manager import VectorStoreManager
from src.rag_engine import RAGEngine

def check_ollama_models():
    """Check if required Ollama models are available"""
    print("🔍 Checking Ollama models...")
    
    required_models = ["llama3.2", "mxbai-embed-large"]
    
    try:
        # Check installed models
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        installed_models = result.stdout.lower()
        
        missing_models = []
        for model in required_models:
            if model not in installed_models:
                missing_models.append(model)
        
        if missing_models:
            print(f"❌ Missing models: {missing_models}")
            print("📥 Please install missing models:")
            for model in missing_models:
                print(f"   ollama pull {model}")
            return False
        else:
            print("✅ All required models are available")
            return True
            
    except FileNotFoundError:
        print("❌ Ollama is not installed or not in PATH")
        print("📥 Please install Ollama from: https://ollama.ai/")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama models: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    directories = [
        "logs",
        "vector_store", 
        "documents"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    
    print("✅ Directories created")

def scrape_wikipedia_data():
    """Scrape Wikipedia data"""
    print("🌐 Scraping Wikipedia data...")
    
    try:
        scraper = WikipediaScraper()
        scraper.scrape_urls_from_file()
        print("✅ Wikipedia data scraped successfully")
        return True
    except Exception as e:
        print(f"❌ Error scraping Wikipedia data: {e}")
        return False

def process_documents():
    """Process all documents"""
    print("📄 Processing documents...")
    
    try:
        processor = DocumentProcessor()
        documents = processor.process_all_documents()
        
        if documents:
            print(f"✅ Processed {len(documents)} document chunks")
            return documents
        else:
            print("❌ No documents processed")
            return []
    except Exception as e:
        print(f"❌ Error processing documents: {e}")
        return []

def initialize_vector_store(documents):
    """Initialize vector store with documents"""
    print("🗃️ Initializing vector store...")
    
    try:
        vs_manager = VectorStoreManager()
        
        # Check embedding model
        if not vs_manager.check_embedding_model():
            print("❌ Embedding model check failed")
            return False
        
        # Add documents
        success = vs_manager.add_documents(documents)
        
        if success:
            print("✅ Vector store initialized successfully")
            return True
        else:
            print("❌ Failed to initialize vector store")
            return False
            
    except Exception as e:
        print(f"❌ Error initializing vector store: {e}")
        return False

def test_rag_engine():
    """Test the RAG engine with sample queries"""
    print("🧪 Testing RAG engine...")
    
    try:
        rag = RAGEngine()
        
        test_queries = [
            "Tell me about Krishnadevaraya",
            "What are the Transformers movies?",
            "What is the Kuru kingdom?"
        ]
        
        print("\n" + "="*60)
        print("🔍 TESTING QUERIES")
        print("="*60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🎯 Query {i}: {query}")
            print("-" * 40)
            
            result = rag.query(query)
            
            print(f"💬 Response: {result['response'][:200]}...")
            print(f"🎯 Confidence: {result['confidence_score']:.1%}")
            print(f"📚 Sources: {len(result['retrieved_documents'])}")
            
            if result['retrieved_documents']:
                print("📖 Retrieved from:")
                for doc in result['retrieved_documents']:
                    print(f"   • {doc['source']}")
        
        print("\n✅ RAG engine test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error testing RAG engine: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 GenBotX Setup")
    print("="*50)
    
    # Check prerequisites
    if not check_ollama_models():
        print("\n❌ Setup failed: Missing required models")
        return False
    
    # Setup directories
    setup_directories()
    
    # Scrape Wikipedia data
    if not scrape_wikipedia_data():
        print("\n⚠️ Warning: Wikipedia scraping failed, continuing with existing documents")
    
    # Process documents
    documents = process_documents()
    if not documents:
        print("\n❌ Setup failed: No documents to process")
        return False
    
    # Initialize vector store
    if not initialize_vector_store(documents):
        print("\n❌ Setup failed: Vector store initialization failed")
        return False
    
    # Test RAG engine
    if not test_rag_engine():
        print("\n❌ Setup failed: RAG engine test failed")
        return False
    
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📋 What's ready:")
    print("   ✅ Wikipedia data scraped and processed")
    print("   ✅ Documents processed and chunked")
    print("   ✅ Vector store initialized with embeddings")
    print("   ✅ RAG engine tested and working")
    print("\n🚀 You can now run the Streamlit app:")
    print("   streamlit run app.py")
    print("\n📚 Or test individual components:")
    print("   python -m src.wikipedia_scraper")
    print("   python -m src.document_processor")
    print("   python -m src.vector_store_manager")
    print("   python -m src.rag_engine")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
