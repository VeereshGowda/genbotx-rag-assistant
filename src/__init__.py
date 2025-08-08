"""
GenBotX - RAG-powered Assistant
A sophisticated document-based question-answering system using LangGraph and ChromaDB.
"""

from .rag_engine import RAGEngine
from .vector_store_manager import VectorStoreManager
from .document_processor import DocumentProcessor
from .wikipedia_scraper import WikipediaScraper
from .content_manager import ContentManager
from .config import GenBotXConfig, get_config

__version__ = "1.0.0"
__author__ = "GenBotX Team"

__all__ = [
    "RAGEngine",
    "VectorStoreManager", 
    "DocumentProcessor",
    "WikipediaScraper",
    "ContentManager",
    "GenBotXConfig",
    "get_config"
]
