"""
GenBotX Streamlit Application
A user-friendly interface for the RAG-powered assistant.
"""

import streamlit as st
import json
import time
import tempfile
import os
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent))

from src.rag_engine import RAGEngine
from src.wikipedia_scraper import WikipediaScraper
from src.document_processor import DocumentProcessor
from src.vector_store_manager import VectorStoreManager
from src.config import get_config
from src.config_ui import show_config_manager

# Get configuration
config = get_config()

# Page configuration
streamlit_config = config.get("streamlit")
st.set_page_config(
    page_title=streamlit_config.get("page_title"),
    page_icon=streamlit_config.get("page_icon"),
    layout=streamlit_config.get("layout"),
    initial_sidebar_state=streamlit_config.get("sidebar_state")
)

# Initialize session state
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "knowledge_base_initialized" not in st.session_state:
    st.session_state.knowledge_base_initialized = False

def check_ollama_status():
    """Check if Ollama is running and models are available"""
    try:
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            models = result.stdout.lower()
            required_models = ["llama3.2", "mxbai-embed-large"]
            available_models = []
            missing_models = []
            
            for model in required_models:
                if model in models:
                    available_models.append(model)
                else:
                    missing_models.append(model)
            
            return {
                "status": "running",
                "available_models": available_models,
                "missing_models": missing_models,
                "all_models_available": len(missing_models) == 0
            }
        else:
            return {"status": "error", "message": "Ollama command failed"}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "message": "Ollama request timed out"}
    except FileNotFoundError:
        return {"status": "not_installed", "message": "Ollama not found in PATH"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def initialize_rag_engine():
    """Initialize the RAG engine"""
    try:
        with st.spinner("Initializing GenBotX..."):
            rag_engine = RAGEngine()
            return rag_engine
    except Exception as e:
        st.error(f"Failed to initialize RAG engine: {e}")
        return None

def scrape_wikipedia_data():
    """Scrape Wikipedia data"""
    try:
        with st.spinner("Scraping Wikipedia data..."):
            scraper = WikipediaScraper()
            scraper.scrape_urls_from_file()
            return True
    except Exception as e:
        st.error(f"Failed to scrape Wikipedia data: {e}")
        return False

def initialize_knowledge_base(rag_engine, force_reinitialize: bool = False):
    """Initialize the knowledge base"""
    try:
        spinner_text = "Force reinitializing knowledge base..." if force_reinitialize else "Processing documents and building knowledge base..."
        with st.spinner(spinner_text):
            success = rag_engine.initialize_knowledge_base(force_reinitialize=force_reinitialize)
            return success
    except Exception as e:
        st.error(f"Failed to initialize knowledge base: {e}")
        return False

def display_chat_message(message: Dict[str, Any], is_user: bool = True):
    """Display a chat message"""
    if is_user:
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            # 🎯 DISPLAY CONFIDENCE SCORE PROMINENTLY AT THE TOP
            if "confidence_score" in message:
                confidence = message["confidence_score"]
                confidence_percent = confidence * 100
                
                # Create a more prominent confidence display
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if confidence > 0.8:
                        st.success(f"🎯 **Confidence Score: {confidence_percent:.1f}%** ✨", icon="✅")
                    elif confidence > 0.6:
                        st.info(f"🎯 **Confidence Score: {confidence_percent:.1f}%**", icon="ℹ️")
                    elif confidence > 0.4:
                        st.warning(f"🎯 **Confidence Score: {confidence_percent:.1f}%**", icon="⚠️")
                    else:
                        st.error(f"🎯 **Confidence Score: {confidence_percent:.1f}%**", icon="❌")
                
                st.divider()
            
            # Main response content
            st.write(message["content"])
            
            # Show reasoning mode indicator
            if "reasoning_enabled" in message:
                if message["reasoning_enabled"]:
                    st.caption("🧠 Generated with Chain-of-Thought reasoning")
                else:
                    st.caption("⚡ Generated in Quick Mode")
            
            # Show retrieved documents in an expander
            if "retrieved_documents" in message and message["retrieved_documents"]:
                with st.expander("📚 Source Documents"):
                    # Group documents by source file to avoid duplicates
                    sources_grouped = {}
                    for doc in message["retrieved_documents"]:
                        source = doc['source']
                        if source not in sources_grouped:
                            sources_grouped[source] = {
                                'chunks': [],
                                'previews': []
                            }
                        sources_grouped[source]['chunks'].append(doc.get('metadata', {}).get('chunk_id', 'Unknown'))
                        sources_grouped[source]['previews'].append(doc['content'])
                    
                    # Display unique sources
                    for i, (source, data) in enumerate(sources_grouped.items(), 1):
                        st.write(f"**Source {i}:** {source}")
                        
                        # Show chunk information if multiple chunks from same file
                        if len(data['chunks']) > 1:
                            chunk_info = [str(c) for c in data['chunks'] if c != 'Unknown']
                            if chunk_info:
                                st.write(f"*Chunks: {', '.join(chunk_info)} ({len(data['chunks'])} total)*")
                            else:
                                st.write(f"*{len(data['chunks'])} relevant sections*")
                        
                        # Show preview from the most relevant chunk (first one)
                        st.write(f"Preview: {data['previews'][0]}")
                        st.divider()
            
            # Show reasoning steps in an expander (only if reasoning was enabled)
            if message.get("reasoning_enabled", True) and "reasoning_steps" in message and message["reasoning_steps"]:
                with st.expander("🧠 Reasoning Process"):
                    for step in message["reasoning_steps"]:
                        st.write(f"• {step}")
            elif not message.get("reasoning_enabled", True):
                with st.expander("⚡ Quick Mode Info"):
                    st.info("This response was generated in Quick Mode - faster but without detailed reasoning steps. Enable Chain-of-Thought reasoning above for more detailed analysis.")

def main():
    """Main Streamlit application"""
    
    # Navigation
    page = st.sidebar.selectbox(
        "🧭 Navigation",
        ["💬 Chat Interface", "⚙️ Configuration Manager", "📊 System Stats"],
        index=0
    )
    
    if page == "💬 Chat Interface":
        show_chat_interface()
    elif page == "⚙️ Configuration Manager":
        show_config_manager()
    elif page == "📊 System Stats":
        show_system_stats()

def show_system_stats():
    """Show system statistics and diagnostics"""
    st.title("📊 System Statistics")
    st.markdown("*System overview and diagnostics*")
    
    if st.session_state.rag_engine:
        try:
            stats = st.session_state.rag_engine.get_stats()
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                doc_count = stats.get("vector_store", {}).get("document_count", 0)
                st.metric("Documents in Vector Store", doc_count)
            
            with col2:
                total_files = stats.get("documents", {}).get("total_files", 0)
                st.metric("Total Files", total_files)
            
            with col3:
                total_size = stats.get("documents", {}).get("total_size", 0)
                st.metric("Total Size", f"{total_size:,} bytes")
            
            with col4:
                messages = stats.get("conversation_memory", {}).get("total_messages", 0)
                st.metric("Conversation Messages", messages)
            
            # Detailed stats
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Vector Store Details")
                if "vector_store" in stats:
                    vs_stats = stats["vector_store"]
                    st.json(vs_stats)
            
            with col2:
                st.subheader("📄 Document Details")
                if "documents" in stats:
                    doc_stats = stats["documents"]
                    st.json(doc_stats)
            
            # Configuration overview
            st.divider()
            st.subheader("⚙️ Current Configuration")
            
            config = get_config()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**LLM Settings:**")
                st.write(f"Model: {config.get('llm.model')}")
                st.write(f"Temperature: {config.get('llm.temperature')}")
                st.write(f"Max Tokens: {config.get('llm.max_tokens')}")
            
            with col2:
                st.write("**Vector Store Settings:**")
                st.write(f"Collection: {config.get('vector_store.collection_name')}")
                st.write(f"Search K: {config.get('vector_store.similarity_search_k')}")
                st.write(f"Chunk Size: {config.get('text_splitter.chunk_size')}")
                
        except Exception as e:
            st.error(f"Error getting stats: {e}")
    else:
        st.warning("RAG Engine not initialized. Please go to Chat Interface and initialize the system.")

def show_chat_interface():
    """Show the main chat interface"""
    
    # Header with Ollama status
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🤖 GenBotX - RAG Assistant")
        st.markdown("*Powered by LangGraph, ChromaDB, and Llama3.2*")
    
    with col2:
        # Ollama Status in top right
        ollama_status = check_ollama_status()
        if ollama_status["status"] == "running":
            if ollama_status["all_models_available"]:
                st.success("🦙 Ollama: Ready")
            else:
                st.warning(f"🦙 Missing Models")
        else:
            st.error("🦙 Ollama: Offline")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Control")
        
        # System Status & Data Management (merged)
        st.subheader("🔧 System Status & Data Management")
        
        # RAG Engine Status
        if st.session_state.rag_engine is None:
            st.error("RAG Engine: Not Initialized")
            if st.button("🚀 Initialize System"):
                st.session_state.rag_engine = initialize_rag_engine()
                if st.session_state.rag_engine:
                    st.success("RAG Engine initialized!")
                    st.rerun()
        else:
            st.success("RAG Engine: Ready")
        
        # Knowledge Base Status and Initialization (placed closer together)
        if st.session_state.rag_engine:
            # Check current vector store status
            try:
                stats = st.session_state.rag_engine.get_stats()
                doc_count = stats.get("vector_store", {}).get("document_count", 0)
                
                if doc_count > 0:
                    st.success(f"Knowledge Base: Ready ({doc_count} documents)")
                    
                    # Single clear option - no need for force reinitialize
                    if st.button("🗑️ Clear & Reinitialize Knowledge Base", help="Completely clear the vector store and rebuild from documents folder"):
                        with st.spinner("Clearing and reinitializing knowledge base..."):
                            if st.session_state.rag_engine.reinitialize_knowledge_base():
                                st.success("Knowledge base cleared and reinitialized!")
                                st.rerun()
                            else:
                                st.error("Failed to clear and reinitialize knowledge base")
                else:
                    st.warning("Knowledge Base: Empty")
                    if st.button("📚 Initialize Sample Knowledge Base"):
                        with st.expander("ℹ️ About Sample Knowledge Base"):
                            st.info("This will process documents from the `documents/` folder including PDF, DOCX, and TXT files. If the knowledge base already contains documents, initialization will be skipped for faster startup.")
                        if initialize_knowledge_base(st.session_state.rag_engine):
                            st.session_state.knowledge_base_initialized = True
                            st.success("Sample knowledge base initialized!")
                            st.rerun()
                        
            except Exception as e:
                st.error(f"Error checking knowledge base status: {e}")
                # Fallback to old behavior
                if not st.session_state.knowledge_base_initialized:
                    st.warning("Knowledge Base: Not Ready")
                    if st.button("📚 Initialize Sample Knowledge Base"):
                        if initialize_knowledge_base(st.session_state.rag_engine):
                            st.session_state.knowledge_base_initialized = True
                            st.success("Sample knowledge base initialized!")
                            st.rerun()
                else:
                    st.success("Knowledge Base: Ready")
        else:
            st.info("Initialize system first to enable knowledge base setup")
        
        st.divider()
        
        # Wikipedia Scraping (separate since it's not included in sample KB)
        st.write("**Additional Data Sources:**")
        if st.button("🌐 Scrape Sample Wikipedia URLs"):
            with st.expander("ℹ️ About Wikipedia Scraping"):
                st.info("This will scrape the sample Wikipedia URLs from `documents/wikipedia_urls.txt` including Gupta Empire, Kuru Kingdom, and Transformers movie series.")
            if scrape_wikipedia_data():
                st.success("Wikipedia data scraped successfully!")
            else:
                st.error("Failed to scrape Wikipedia data")
        
        st.divider()
        
        # File Upload Section
        st.subheader("📁 Upload Files")
        
        uploaded_files = st.file_uploader(
            "Choose files to add to knowledge base",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files. Duplicate content will be automatically detected and skipped."
        )
        
        if uploaded_files and st.button("🔄 Process Uploaded Files"):
            if st.session_state.rag_engine:
                with st.spinner("📄 Processing uploaded files and adding to knowledge base..."):
                    # Save uploaded files temporarily
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            file_paths.append((tmp_file.name, uploaded_file.name))
                    
                    try:
                        # Process files
                        results = st.session_state.rag_engine.add_uploaded_files(file_paths)
                        
                        # Display results
                        if results["processed"] > 0:
                            st.success(f"✅ Processed {results['processed']} new files - You can now ask questions about them!")
                            if "chunks_added" in results:
                                st.info(f"Added {results['chunks_added']} document chunks to knowledge base")
                        
                        if results["skipped_duplicates"] > 0:
                            st.warning(f"⚠️ Skipped {results['skipped_duplicates']} duplicate files")
                        
                        if results["errors"] > 0:
                            st.error(f"❌ Failed to process {results['errors']} files")
                        
                        if results.get("vector_store_error"):
                            st.error(f"Vector store error: {results.get('vector_store_error', 'Unknown error')}")
                        
                        # Show processed files
                        if results["new_documents"]:
                            with st.expander("📋 Files Ready for Questions"):
                                for filename in results["new_documents"]:
                                    st.write(f"• {filename}")
                                st.info("💬 You can now ask questions about these files in the chat below!")
                    
                    finally:
                        # Clean up temporary files
                        for file_path, _ in file_paths:
                            try:
                                os.unlink(file_path)
                            except:
                                pass
            else:
                st.error("Please initialize the RAG engine first")
        
        st.divider()
        
        # URL Scraping Section
        st.subheader("🌐 Add Webpages")
        
        # URL input methods
        url_input_method = st.radio(
            "Choose input method:",
            ["Single URL", "Multiple URLs", "Upload URL file"],
            help="Add webpages to the knowledge base by providing URLs"
        )
        
        urls_to_process = []
        
        if url_input_method == "Single URL":
            single_url = st.text_input(
                "Enter webpage URL:",
                placeholder="https://example.com/article",
                help="Enter a single webpage URL to scrape and add to knowledge base"
            )
            if single_url.strip():
                urls_to_process = [single_url.strip()]
        
        elif url_input_method == "Multiple URLs":
            urls_text = st.text_area(
                "Enter URLs (one per line):",
                placeholder="https://example.com/page1\nhttps://example.com/page2\nhttps://example.com/page3",
                help="Enter multiple URLs, one per line"
            )
            if urls_text.strip():
                urls_to_process = [url.strip() for url in urls_text.split('\n') if url.strip()]
        
        else:  # Upload URL file
            url_file = st.file_uploader(
                "Upload text file with URLs",
                type=['txt'],
                help="Upload a text file containing URLs (one per line)"
            )
            if url_file:
                content = url_file.getvalue().decode('utf-8')
                urls_to_process = [url.strip() for url in content.split('\n') if url.strip() and not url.startswith('#')]
        
        # Display URLs to be processed
        if urls_to_process:
            st.write(f"**URLs to process:** {len(urls_to_process)}")
            with st.expander("📋 URL List"):
                for i, url in enumerate(urls_to_process, 1):
                    st.write(f"{i}. {url}")
        
        # Process URLs button
        if urls_to_process and st.button("🔄 Scrape and Add Webpages"):
            if st.session_state.rag_engine:
                with st.spinner(f"🌐 Scraping {len(urls_to_process)} webpages and adding to knowledge base..."):
                    try:
                        results = st.session_state.rag_engine.add_webpage_urls(urls_to_process)
                        
                        # Display results
                        if results["processed"] > 0:
                            st.success(f"✅ Successfully scraped {results['processed']} webpages - You can now ask questions about them!")
                            if "chunks_added" in results:
                                st.info(f"Added {results['chunks_added']} document chunks to knowledge base")
                        
                        if results["skipped_duplicates"] > 0:
                            st.warning(f"⚠️ Skipped {results['skipped_duplicates']} duplicate webpages")
                        
                        if results["errors"] > 0:
                            st.error(f"❌ Failed to scrape {results['errors']} webpages")
                        
                        if results.get("vector_store_error"):
                            st.error(f"Vector store error: {results.get('vector_store_error', 'Unknown error')}")
                        
                        # Show processed URLs
                        if results["new_documents"]:
                            with st.expander("📋 Webpages Ready for Questions"):
                                for url in results["new_documents"]:
                                    st.write(f"• {url}")
                                st.info("💬 You can now ask questions about these webpages in the chat below!")
                    
                    except Exception as e:
                        st.error(f"Error processing URLs: {e}")
            else:
                st.error("Please initialize the RAG engine first")
        
        st.divider()
        
        # System Stats
        if st.session_state.rag_engine:
            st.subheader("📈 System Statistics")
            
            if st.button("📊 Show Stats"):
                try:
                    stats = st.session_state.rag_engine.get_stats()
                    
                    # Vector Store Stats
                    if "vector_store" in stats:
                        vs_stats = stats["vector_store"]
                        st.metric("Documents in Vector Store", vs_stats.get("document_count", 0))
                    
                    # Document Stats
                    if "documents" in stats:
                        doc_stats = stats["documents"]
                        st.metric("Total Files", doc_stats.get("total_files", 0))
                        st.metric("Total Size", f"{doc_stats.get('total_size', 0)} bytes")
                    
                    # Content Manager Stats
                    if "content_manager" in stats:
                        content_stats = stats["content_manager"]
                        st.metric("Unique Content Items", content_stats.get("total_unique_content", 0))
                        content_types = content_stats.get("content_types", {})
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Uploaded Files", content_types.get("uploaded_file", 0))
                        with col2:
                            st.metric("Scraped Webpages", content_types.get("scraped_webpage", 0))
                        with col3:
                            st.metric("Wikipedia Pages", content_types.get("wikipedia", 0))
                    
                    # Memory Stats
                    if "conversation_memory" in stats:
                        mem_stats = stats["conversation_memory"]
                        st.metric("Conversation Messages", mem_stats.get("total_messages", 0))
                    
                    with st.expander("📋 Detailed Stats"):
                        st.json(stats)
                        
                except Exception as e:
                    st.error(f"Error getting stats: {e}")
        
        st.divider()
        
        st.divider()
        
        # Content Management
        st.subheader("🗂️ Content Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Duplicates Cache"):
                if st.session_state.rag_engine:
                    st.session_state.rag_engine.content_manager.clear_duplicates_cache()
                    st.success("Duplicates cache cleared!")
                else:
                    st.error("RAG engine not initialized")
        
        with col2:
            if st.button("📊 Refresh Stats"):
                st.rerun()
        
        # Clear Chat History
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main Chat Interface
    st.header("💬 Chat with GenBotX")
    
    # Initialize reasoning setting if not exists
    if "enable_reasoning" not in st.session_state:
        st.session_state.enable_reasoning = True  # Default enabled
    
    # Create two columns for the toggle and status
    col1, col2 = st.columns([2, 2])
    
    with col1:
        enable_reasoning = st.toggle(
            "Enable Chain-of-Thought Reasoning",
            value=st.session_state.enable_reasoning,
            help="When enabled, the AI will apply step-by-step reasoning for more accurate responses (takes longer). When disabled, responses are faster but may be less detailed."
        )
        
        # Update session state if changed
        if enable_reasoning != st.session_state.enable_reasoning:
            st.session_state.enable_reasoning = enable_reasoning
            st.rerun()
    
    with col2:
        # Show current reasoning status
        if st.session_state.enable_reasoning:
            st.success("🧠 Chain-of-Thought: **Enabled**  \n*More accurate, slower responses*")
        else:
            st.info("⚡ Quick Mode: **Enabled**  \n*Faster, less detailed responses*")
    
    st.divider()
    
    # 🔝 CHAT INPUT MOVED TO TOP FOR BETTER UX
    if st.session_state.rag_engine:
        st.subheader("💭 Ask Your Question")
        
        # Create a more prominent input area
        user_input = st.text_area(
            "What would you like to know?",
            height=100,
            placeholder="Ask me anything about the documents... \n\nExample: 'Tell me about the Kuru kingdom' or 'What were the Anglo-Mysore Wars?'",
            help="Type your question here and press Ctrl+Enter or click the 'Ask Question' button below"
        )
        
        # Create columns for the ask button and clear option
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            ask_button = st.button("🚀 Ask Question", type="primary", use_container_width=True)
        
        with col2:
            if st.button("🗑️ Clear Input", use_container_width=True):
                st.rerun()
        
        with col3:
            if st.button("📋 Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # Process the question
        if (ask_button or user_input.endswith('\n\n')) and user_input.strip():
            # Clean the input
            question = user_input.strip()
            
            # Add user message to history
            user_message = {
                "content": question,
                "is_user": True,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.chat_history.append(user_message)
            
            # Get response from RAG engine
            spinner_message = "🧠 GenBotX is applying chain-of-thought reasoning..." if st.session_state.enable_reasoning else "⚡ GenBotX is generating a quick response..."
            
            with st.spinner(spinner_message):
                try:
                    # Pass reasoning setting to RAG engine
                    result = st.session_state.rag_engine.query(
                        question, 
                        enable_reasoning=st.session_state.enable_reasoning
                    )
                    
                    # Add assistant message to history
                    assistant_message = {
                        "content": result["response"],
                        "is_user": False,
                        "confidence_score": result.get("confidence_score", 0.0),
                        "retrieved_documents": result.get("retrieved_documents", []),
                        "reasoning_steps": result.get("reasoning_steps", []),
                        "reasoning_enabled": st.session_state.enable_reasoning,
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.chat_history.append(assistant_message)
                    
                    # Rerun to clear the input and show new message
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing query: {e}")
        
    elif not st.session_state.rag_engine:
        st.info("⚠️ Please initialize the system before chatting.")
    
    # Example queries section (only show if no conversation history exists)
    if st.session_state.rag_engine and not st.session_state.chat_history:
        st.subheader("💡 Quick Start Examples")
        st.caption("Click any example below to try it out:")
        
        example_queries = [
            "Tell me about Krishnadevaraya",
            "What are the Transformers movies about?",
            "Describe the Kuru kingdom",
            "What were the Anglo-Mysore Wars?",
            "Compare different kingdoms in Indian history"
        ]
        
        cols = st.columns(2)
        for i, query in enumerate(example_queries):
            col = cols[i % 2]
            if col.button(f"📝 {query}", key=f"example_{i}", use_container_width=True):
                # Set the query as if user typed it
                st.session_state["example_query"] = query
                st.rerun()
        
        # Process example query if set
        if "example_query" in st.session_state:
            query = st.session_state.pop("example_query")
            
            # Add user message to history
            user_message = {
                "content": query,
                "is_user": True,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.chat_history.append(user_message)
            
            # Get response from RAG engine
            spinner_message = "🧠 GenBotX is applying chain-of-thought reasoning..." if st.session_state.enable_reasoning else "⚡ GenBotX is generating a quick response..."
            
            with st.spinner(spinner_message):
                try:
                    result = st.session_state.rag_engine.query(
                        query, 
                        enable_reasoning=st.session_state.enable_reasoning
                    )
                    
                    # Add assistant message to history
                    assistant_message = {
                        "content": result["response"],
                        "is_user": False,
                        "confidence_score": result.get("confidence_score", 0.0),
                        "retrieved_documents": result.get("retrieved_documents", []),
                        "reasoning_steps": result.get("reasoning_steps", []),
                        "reasoning_enabled": st.session_state.enable_reasoning,
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.chat_history.append(assistant_message)
                    
                    st.rerun()
                    
                except Exception as e:
                        st.error(f"Error processing example query: {e}")
    
    st.divider()    # 💬 CHAT HISTORY DISPLAY (now after input for better flow)
    if st.session_state.chat_history:
        st.subheader("💬 Conversation History")
        
        # Display chat history in chronological order (question first, then response)
        for message in st.session_state.chat_history:
            display_chat_message(message, is_user=message.get("is_user", False))
    else:
        st.info("💭 **Start a conversation!** Ask a question above or try one of the example queries.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>GenBotX v1.0 | Built with LangGraph, ChromaDB, and Streamlit</p>
        <p>Embedding Model: mxbai-embed-large | LLM: Llama3.2</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
