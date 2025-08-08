"""
Configuration UI for GenBotX Streamlit App
Provides an interface to view and modify configuration settings.
"""

import streamlit as st
import json
import yaml
from pathlib import Path
from src.config import get_config
from typing import Dict, Any

def show_config_manager():
    """Display configuration management interface"""
    config = get_config()
    
    st.header("⚙️ Configuration Manager")
    
    # Tabs for different config sections
    tabs = st.tabs([
        "🤖 LLM Settings", 
        "🔍 Vector Store", 
        "📄 Text Processing", 
        "🌐 Web Scraping",
        "📊 System",
        "💾 Export/Import"
    ])
    
    # LLM Settings Tab
    with tabs[0]:
        st.subheader("Large Language Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            llm_model = st.text_input(
                "LLM Model", 
                value=config.get("llm.model"),
                help="Ollama model name for text generation"
            )
            
            temperature = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=2.0, 
                value=float(config.get("llm.temperature")),
                step=0.1,
                help="Controls response creativity (0.0 = deterministic, 2.0 = very creative)"
            )
        
        with col2:
            max_tokens = st.number_input(
                "Max Tokens", 
                min_value=100, 
                max_value=8000, 
                value=config.get("llm.max_tokens"),
                help="Maximum length of generated responses"
            )
            
            timeout = st.number_input(
                "Timeout (seconds)", 
                min_value=10, 
                max_value=300, 
                value=config.get("llm.timeout"),
                help="Request timeout for LLM calls"
            )
        
        # Embedding settings
        st.subheader("Embedding Model Configuration")
        embedding_model = st.text_input(
            "Embedding Model", 
            value=config.get("embeddings.model"),
            help="Ollama model for text embeddings"
        )
        
        embedding_timeout = st.number_input(
            "Embedding Timeout", 
            min_value=10, 
            max_value=120, 
            value=config.get("embeddings.timeout"),
            help="Timeout for embedding requests"
        )
        
        if st.button("💾 Save LLM Settings"):
            config.update_section("llm", {
                "model": llm_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout
            })
            config.update_section("embeddings", {
                "model": embedding_model,
                "timeout": embedding_timeout
            })
            config.save_config()
            st.success("LLM settings saved!")
            st.rerun()
    
    # Vector Store Tab
    with tabs[1]:
        st.subheader("Vector Store Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            collection_name = st.text_input(
                "Collection Name", 
                value=config.get("vector_store.collection_name"),
                help="ChromaDB collection name"
            )
            
            similarity_k = st.number_input(
                "Similarity Search K", 
                min_value=1, 
                max_value=20, 
                value=config.get("vector_store.similarity_search_k"),
                help="Number of documents to retrieve"
            )
        
        with col2:
            persist_dir = st.text_input(
                "Persist Directory", 
                value=config.get("vector_store.persist_directory"),
                help="Directory to store vector database"
            )
            
            similarity_threshold = st.slider(
                "Similarity Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=float(config.get("vector_store.similarity_threshold")),
                step=0.05,
                help="Minimum similarity score for retrieval"
            )
        
        if st.button("💾 Save Vector Store Settings"):
            config.update_section("vector_store", {
                "collection_name": collection_name,
                "persist_directory": persist_dir,
                "similarity_search_k": similarity_k,
                "similarity_threshold": similarity_threshold
            })
            config.save_config()
            st.success("Vector store settings saved!")
            st.rerun()
    
    # Text Processing Tab
    with tabs[2]:
        st.subheader("Text Processing Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_size = st.number_input(
                "Chunk Size", 
                min_value=100, 
                max_value=2000, 
                value=config.get("text_splitter.chunk_size"),
                help="Characters per text chunk"
            )
            
            chunk_overlap = st.number_input(
                "Chunk Overlap", 
                min_value=0, 
                max_value=500, 
                value=config.get("text_splitter.chunk_overlap"),
                help="Overlap between chunks"
            )
        
        with col2:
            separators = st.text_area(
                "Separators (one per line)", 
                value="\n".join(config.get("text_splitter.separators")),
                help="Text splitting separators in order of preference"
            )
            
            max_file_size = st.number_input(
                "Max File Size (MB)", 
                min_value=1, 
                max_value=200, 
                value=config.get("content_manager.max_file_size_mb"),
                help="Maximum file size for uploads"
            )
        
        if st.button("💾 Save Text Processing Settings"):
            separator_list = [s.strip() for s in separators.split("\n") if s.strip()]
            config.update_section("text_splitter", {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "separators": separator_list
            })
            config.update_section("content_manager", {
                "max_file_size_mb": max_file_size
            })
            config.save_config()
            st.success("Text processing settings saved!")
            st.rerun()
    
    # Web Scraping Tab
    with tabs[3]:
        st.subheader("Web Scraping Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            request_delay = st.number_input(
                "Request Delay (seconds)", 
                min_value=0.1, 
                max_value=10.0, 
                value=float(config.get("web_scraping.request_delay")),
                step=0.1,
                help="Delay between web requests"
            )
            
            web_timeout = st.number_input(
                "Web Timeout", 
                min_value=5, 
                max_value=120, 
                value=config.get("web_scraping.timeout"),
                help="Timeout for web requests"
            )
        
        with col2:
            max_retries = st.number_input(
                "Max Retries", 
                min_value=1, 
                max_value=10, 
                value=config.get("web_scraping.max_retries"),
                help="Maximum retry attempts"
            )
            
            user_agent = st.text_input(
                "User Agent", 
                value=config.get("web_scraping.user_agent"),
                help="User agent for web requests"
            )
        
        if st.button("💾 Save Web Scraping Settings"):
            config.update_section("web_scraping", {
                "request_delay": request_delay,
                "timeout": web_timeout,
                "max_retries": max_retries,
                "user_agent": user_agent
            })
            config.save_config()
            st.success("Web scraping settings saved!")
            st.rerun()
    
    # System Tab
    with tabs[4]:
        st.subheader("System Configuration")
        
        # RAG Pipeline settings
        st.write("**RAG Pipeline Settings**")
        col1, col2 = st.columns(2)
        
        with col1:
            enable_reasoning = st.checkbox(
                "Enable Reasoning", 
                value=config.get("rag_pipeline.enable_reasoning"),
                help="Enable chain-of-thought reasoning"
            )
            
            enable_quality_check = st.checkbox(
                "Enable Quality Check", 
                value=config.get("rag_pipeline.enable_quality_check"),
                help="Enable response quality checking"
            )
        
        with col2:
            confidence_threshold = st.slider(
                "Confidence Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=float(config.get("rag_pipeline.confidence_threshold")),
                step=0.05,
                help="Minimum confidence for responses"
            )
            
            max_reasoning_steps = st.number_input(
                "Max Reasoning Steps", 
                min_value=1, 
                max_value=20, 
                value=config.get("rag_pipeline.max_reasoning_steps"),
                help="Maximum reasoning steps"
            )
        
        # Logging settings
        st.write("**Logging Configuration**")
        col3, col4 = st.columns(2)
        
        with col3:
            log_level = st.selectbox(
                "Log Level", 
                options=["DEBUG", "INFO", "WARNING", "ERROR"],
                index=["DEBUG", "INFO", "WARNING", "ERROR"].index(config.get("logging.level")),
                help="Logging verbosity level"
            )
        
        with col4:
            log_rotation = st.text_input(
                "Log Rotation", 
                value=config.get("logging.rotation"),
                help="Log file rotation size"
            )
        
        if st.button("💾 Save System Settings"):
            config.update_section("rag_pipeline", {
                "enable_reasoning": enable_reasoning,
                "enable_quality_check": enable_quality_check,
                "confidence_threshold": confidence_threshold,
                "max_reasoning_steps": max_reasoning_steps
            })
            config.update_section("logging", {
                "level": log_level,
                "rotation": log_rotation
            })
            config.save_config()
            st.success("System settings saved!")
            st.rerun()
    
    # Export/Import Tab
    with tabs[5]:
        st.subheader("Configuration Export/Import")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Export Configuration**")
            
            export_format = st.radio(
                "Export Format", 
                options=["YAML", "JSON"],
                help="Choose export format"
            )
            
            if st.button("📤 Export Config"):
                if export_format == "YAML":
                    config_str = yaml.dump(config.config, default_flow_style=False, indent=2)
                    filename = "genbotx_config.yaml"
                else:
                    config_str = json.dumps(config.config, indent=2)
                    filename = "genbotx_config.json"
                
                st.download_button(
                    label=f"Download {filename}",
                    data=config_str,
                    file_name=filename,
                    mime="text/plain"
                )
        
        with col2:
            st.write("**Import Configuration**")
            
            uploaded_config = st.file_uploader(
                "Upload Config File", 
                type=["yaml", "yml", "json"],
                help="Upload a configuration file to import settings"
            )
            
            if uploaded_config is not None:
                try:
                    content = uploaded_config.read().decode('utf-8')
                    
                    if uploaded_config.name.endswith(('.yaml', '.yml')):
                        imported_config = yaml.safe_load(content)
                    else:
                        imported_config = json.loads(content)
                    
                    st.write("**Preview of imported configuration:**")
                    st.json(imported_config)
                    
                    if st.button("✅ Apply Imported Config"):
                        config.config = config._deep_merge(config.config, imported_config)
                        config.save_config()
                        st.success("Configuration imported successfully!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error importing configuration: {e}")
        
        # Reset to defaults
        st.divider()
        st.write("**Reset Configuration**")
        st.warning("This will reset all settings to default values!")
        
        if st.button("🔄 Reset to Defaults", type="primary"):
            if st.session_state.get("confirm_reset", False):
                config.reset_to_defaults()
                st.success("Configuration reset to defaults!")
                st.session_state.confirm_reset = False
                st.rerun()
            else:
                st.session_state.confirm_reset = True
                st.warning("Click again to confirm reset!")
    
    # Configuration validation
    st.divider()
    st.subheader("🔍 Configuration Validation")
    
    if st.button("🔍 Validate Configuration"):
        is_valid, errors = config.validate_config()
        
        if is_valid:
            st.success("✅ Configuration is valid!")
        else:
            st.error("❌ Configuration has issues:")
            for error in errors:
                st.error(f"• {error}")
    
    # Current configuration display
    with st.expander("📋 View Current Configuration"):
        st.json(config.config)
