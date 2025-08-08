"""
Vector Store Manager for GenBotX
Manages ChromaDB vector store operations for the RAG system.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document
from loguru import logger
import time
from .config import get_config

class VectorStoreManager:
    def __init__(self, 
                 collection_name: Optional[str] = None,
                 persist_directory: Optional[str] = None,
                 embedding_model: Optional[str] = None):
        
        # Get configuration
        config = get_config()
        
        self.collection_name = collection_name or config.get("vector_store.collection_name")
        self.persist_directory = Path(persist_directory or config.get("vector_store.persist_directory"))
        embedding_model = embedding_model or config.get("embeddings.model")
        
        # Create directories
        self.persist_directory.mkdir(exist_ok=True)
        
        # Configure loguru logger
        log_config = config.get("logging")
        logger.add(
            f"{config.get('directories.logs')}/vector_store.log", 
            rotation=log_config.get("rotation", "10 MB"),
            level=log_config.get("level", "INFO"),
            format=log_config.get("format")
        )
        
        # Initialize Ollama embeddings
        try:
            self.embeddings = OllamaEmbeddings(
                model=embedding_model,
                base_url="http://localhost:11434"
            )
            logger.info(f"Initialized Ollama embeddings with model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama embeddings: {e}")
            raise
        
        # Initialize ChromaDB
        self.chroma_client = None
        self.vectorstore = None
        self._initialize_chroma()
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and vector store"""
        try:
            # Create ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_directory)
            )
            
            # Initialize Langchain Chroma vector store
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            
            logger.info(f"Initialized ChromaDB at {self.persist_directory}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store"""
        if not documents:
            logger.warning("No documents to add")
            return False
        
        try:
            logger.info(f"Adding {len(documents)} documents to vector store...")
            
            # Get batch size from configuration
            config = get_config()
            configured_batch_size = config.get("performance.batch_size", 20)
            
            # Use configured batch size for processing
            if len(documents) <= configured_batch_size:
                batch_size = len(documents)  # Process all at once for small batches
            else:
                batch_size = configured_batch_size  # Use configured batch size for larger sets
            
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                
                # Add batch to vector store
                self.vectorstore.add_documents(batch)
                
                # Minimal delay only for large batches
                if total_batches > 5:
                    time.sleep(0.01)  # Reduced from 0.05 to 0.01
            
            logger.info("Successfully added all documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            return False
    
    def similarity_search(self, 
                         query: str, 
                         k: Optional[int] = None, 
                         filter_metadata: Optional[Dict] = None) -> List[Document]:
        """Perform similarity search in the vector store"""
        try:
            config = get_config()
            k = k or config.get("vector_store.similarity_search_k", 5)
            
            if filter_metadata:
                results = self.vectorstore.similarity_search(
                    query=query, 
                    k=k, 
                    filter=filter_metadata
                )
            else:
                results = self.vectorstore.similarity_search(query=query, k=k)
            
            logger.info(f"Found {len(results)} similar documents for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, 
                                   query: str, 
                                   k: Optional[int] = None,
                                   filter_metadata: Optional[Dict] = None) -> List[tuple]:
        """Perform similarity search with relevance scores"""
        try:
            config = get_config()
            k = k or config.get("vector_store.similarity_search_k", 5)
            
            if filter_metadata:
                results = self.vectorstore.similarity_search_with_score(
                    query=query, 
                    k=k, 
                    filter=filter_metadata
                )
            else:
                results = self.vectorstore.similarity_search_with_score(query=query, k=k)
            
            logger.info(f"Found {len(results)} similar documents with scores for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search with score: {e}")
            return []
    
    def get_retriever(self, k: Optional[int] = None, search_type: str = "similarity"):
        """Get a retriever for the vector store"""
        config = get_config()
        k = k or config.get("vector_store.similarity_search_k", 5)
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
    
    def delete_collection(self) -> bool:
        """Delete the current collection"""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            
            # Reinitialize the vector store
            self._initialize_chroma()
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection"""
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            count = collection.count()
            
            stats = {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": "mxbai-embed-large",
                "persist_directory": str(self.persist_directory)
            }
            
            # Try to get a sample document to check metadata structure
            if count > 0:
                sample = collection.get(limit=1, include=["metadatas"])
                if sample and sample.get("metadatas"):
                    stats["sample_metadata"] = sample["metadatas"][0]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def list_collections(self) -> List[str]:
        """List all collections in the ChromaDB"""
        try:
            collections = self.chroma_client.list_collections()
            collection_names = [col.name for col in collections]
            logger.info(f"Found {len(collection_names)} collections")
            return collection_names
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def check_embedding_model(self) -> bool:
        """Check if the embedding model is available"""
        try:
            # Test embedding with a simple query
            test_text = "This is a test embedding"
            embedding = self.embeddings.embed_query(test_text)
            
            if embedding and len(embedding) > 0:
                logger.info(f"Embedding model working correctly. Dimension: {len(embedding)}")
                return True
            else:
                logger.error("Embedding model returned empty result")
                return False
                
        except Exception as e:
            logger.error(f"Embedding model check failed: {e}")
            return False

def main():
    """Test the vector store manager"""
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    try:
        # Initialize vector store manager
        vs_manager = VectorStoreManager()
        
        # Check embedding model
        if vs_manager.check_embedding_model():
            print("✓ Embedding model is working correctly")
        else:
            print("✗ Embedding model has issues")
        
        # Get collection stats
        stats = vs_manager.get_collection_stats()
        print(f"\nCollection Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # List collections
        collections = vs_manager.list_collections()
        print(f"\nAvailable collections: {collections}")
        
    except Exception as e:
        print(f"Error initializing vector store manager: {e}")

if __name__ == "__main__":
    main()
