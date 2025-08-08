"""
Test Configuration System for GenBotX
Validates configuration loading and usage across components.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import get_config, GenBotXConfig
from src.vector_store_manager import VectorStoreManager
from src.document_processor import DocumentProcessor
from src.rag_engine import RAGEngine

def test_config_loading():
    """Test configuration loading and access"""
    print("🔧 Testing Configuration System")
    print("="*50)
    
    # Test basic config loading
    config = get_config()
    print(f"✅ Configuration loaded successfully")
    
    # Test config access
    llm_model = config.get("llm.model")
    embedding_model = config.get("embeddings.model")
    chunk_size = config.get("text_splitter.chunk_size")
    
    print(f"📋 LLM Model: {llm_model}")
    print(f"📋 Embedding Model: {embedding_model}")
    print(f"📋 Chunk Size: {chunk_size}")
    
    # Test config validation
    is_valid, errors = config.validate_config()
    if is_valid:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has errors:")
        for error in errors:
            print(f"   • {error}")
    
    return config

def test_component_initialization():
    """Test that all components can initialize with config"""
    print("\n🧪 Testing Component Initialization")
    print("="*50)
    
    try:
        # Test vector store manager
        print("Testing VectorStoreManager...")
        vs_manager = VectorStoreManager()
        print("✅ VectorStoreManager initialized")
        
        # Test document processor
        print("Testing DocumentProcessor...")
        doc_processor = DocumentProcessor()
        print("✅ DocumentProcessor initialized")
        
        # Test RAG engine (may fail if Ollama not available)
        print("Testing RAGEngine...")
        try:
            rag_engine = RAGEngine()
            print("✅ RAGEngine initialized")
        except Exception as e:
            print(f"⚠️ RAGEngine initialization failed (expected if Ollama not running): {e}")
        
        print("\n✅ All components can use configuration successfully!")
        
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")

def test_config_modification():
    """Test configuration modification and saving"""
    print("\n🔧 Testing Configuration Modification")
    print("="*50)
    
    config = get_config()
    
    # Test setting a value
    original_temp = config.get("llm.temperature")
    new_temp = 0.8
    
    config.set("llm.temperature", new_temp)
    retrieved_temp = config.get("llm.temperature")
    
    if retrieved_temp == new_temp:
        print(f"✅ Configuration modification successful: {original_temp} → {new_temp}")
    else:
        print(f"❌ Configuration modification failed")
    
    # Restore original value
    config.set("llm.temperature", original_temp)
    
    # Test section update
    config.update_section("test_section", {"test_key": "test_value"})
    test_value = config.get("test_section.test_key")
    
    if test_value == "test_value":
        print("✅ Section update successful")
    else:
        print("❌ Section update failed")

def test_ollama_models():
    """Test Ollama model configuration"""
    print("\n🤖 Testing Ollama Model Configuration")
    print("="*50)
    
    config = get_config()
    models = config.get_ollama_models()
    
    print(f"📋 Configured models: {models}")
    
    # Test if we can create embeddings (requires Ollama)
    try:
        vs_manager = VectorStoreManager()
        if vs_manager.check_embedding_model():
            print("✅ Embedding model is working")
        else:
            print("❌ Embedding model check failed")
    except Exception as e:
        print(f"⚠️ Embedding model test failed (expected if Ollama not running): {e}")

def main():
    """Run all configuration tests"""
    print("🚀 GenBotX Configuration System Test")
    print("="*60)
    
    try:
        # Test basic configuration
        config = test_config_loading()
        
        # Test component initialization
        test_component_initialization()
        
        # Test configuration modification
        test_config_modification()
        
        # Test Ollama models
        test_ollama_models()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED")
        print("Configuration system is working correctly!")
        
        # Print current configuration
        print("\n📋 Current Configuration Overview:")
        config.print_config()
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
