"""
GenBotX - Main Entry Point
RAG-powered assistant for document-based question answering.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.rag_engine import RAGEngine

def main():
    """Main function - starts the RAG engine in CLI mode"""
    print("🤖 Welcome to GenBotX!")
    print("=" * 50)
    print("RAG-powered assistant for document-based Q&A")
    print("Built with LangGraph, ChromaDB, and Llama3.2")
    print("=" * 50)
    
    try:
        # Initialize RAG engine
        print("\n🚀 Initializing GenBotX...")
        rag = RAGEngine()
        
        # Check if knowledge base is ready
        stats = rag.get_stats()
        doc_count = stats.get("vector_store", {}).get("document_count", 0)
        
        if doc_count == 0:
            print("\n⚠️ Knowledge base is empty!")
            print("Please run: python setup.py")
            return
        
        print(f"✅ Knowledge base ready with {doc_count} documents")
        
        # Interactive CLI
        print("\n💬 Ask me anything about the documents (type 'quit' to exit):")
        print("-" * 50)
        
        while True:
            query = input("\n🎯 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print("\n🤔 Thinking...")
            result = rag.query(query)
            
            print(f"\n💬 GenBotX: {result['response']}")
            print(f"🎯 Confidence: {result['confidence_score']:.1%}")
            
            if result['retrieved_documents']:
                print(f"📚 Retrieved from {len(result['retrieved_documents'])} sources:")
                for doc in result['retrieved_documents']:
                    print(f"   • {doc['source']}")
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print("1. Ollama is installed and running")
        print("2. Required models are available (llama3.2, mxbai-embed-large)")
        print("3. Run: python setup.py")

if __name__ == "__main__":
    main()
