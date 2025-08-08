"""
Configuration file for GenBotX
Contains all configurable parameters for the RAG system.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

class GenBotXConfig:
    """Configuration manager for GenBotX"""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = Path(config_file)
        self.config = self._load_default_config()
        
        # Load from file if exists
        if self.config_file.exists():
            self.load_config()
        else:
            # Save default config
            self.save_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values"""
        return {
            "llm": {
                "model": "llama3.2",
                "temperature": 0.7,
                "max_tokens": 2000,
                "timeout": 60
            },
            "embeddings": {
                "model": "mxbai-embed-large",
                "timeout": 30
            },
            "vector_store": {
                "collection_name": "genbotx_documents",
                "persist_directory": "vector_store",
                "similarity_search_k": 5,
                "similarity_threshold": 0.7
            },
            "text_splitter": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "separators": ["\n\n", "\n", " ", ""]
            },
            "content_manager": {
                "uploads_directory": "uploads",
                "supported_formats": ["pdf", "docx", "txt"],
                "max_file_size_mb": 50,
                "duplicate_check_enabled": True
            },
            "web_scraping": {
                "request_delay": 1.0,
                "timeout": 30,
                "max_retries": 3,
                "user_agent": "GenBotX/1.0 Web Scraper"
            },
            "wikipedia": {
                "language": "en",
                "extract_format": "wiki",
                "user_agent": "GenBotX/1.0 Wikipedia Scraper"
            },
            "logging": {
                "level": "INFO",
                "rotation": "10 MB",
                "retention": "7 days",
                "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
            },
            "directories": {
                "documents": "documents",
                "logs": "logs",
                "vector_store": "vector_store",
                "uploads": "uploads"
            },
            "streamlit": {
                "page_title": "GenBotX - RAG Assistant",
                "page_icon": "🤖",
                "layout": "wide",
                "sidebar_state": "expanded"
            },
            "rag_pipeline": {
                "enable_reasoning": True,
                "enable_quality_check": True,
                "confidence_threshold": 0.3,
                "max_reasoning_steps": 10
            },
            "memory": {
                "type": "ConversationBufferMemory",
                "max_messages": 20,
                "return_messages": True
            }
        }
    
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            if self.config_file.suffix.lower() == '.yaml' or self.config_file.suffix.lower() == '.yml':
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
            else:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
            
            # Merge with default config (file config overrides defaults)
            self.config = self._deep_merge(self.config, file_config)
            logger.info(f"Configuration loaded from {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading config from {self.config_file}: {e}")
            logger.info("Using default configuration")
            return False
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            if self.config_file.suffix.lower() == '.yaml' or self.config_file.suffix.lower() == '.yml':
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
            else:
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config to {self.config_file}: {e}")
            return False
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'llm.model')"""
        try:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                value = value[key]
            
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> bool:
        """Set configuration value using dot notation"""
        try:
            keys = key_path.split('.')
            config_ref = self.config
            
            # Navigate to the parent of the final key
            for key in keys[:-1]:
                if key not in config_ref:
                    config_ref[key] = {}
                config_ref = config_ref[key]
            
            # Set the final value
            config_ref[keys[-1]] = value
            return True
            
        except Exception as e:
            logger.error(f"Error setting config value {key_path}: {e}")
            return False
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> bool:
        """Update an entire configuration section"""
        try:
            if section not in self.config:
                self.config[section] = {}
            
            self.config[section].update(updates)
            return True
            
        except Exception as e:
            logger.error(f"Error updating config section {section}: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to default values"""
        try:
            self.config = self._load_default_config()
            return self.save_config()
        except Exception as e:
            logger.error(f"Error resetting config to defaults: {e}")
            return False
    
    def validate_config(self) -> tuple[bool, list]:
        """Validate configuration and return (is_valid, errors)"""
        errors = []
        
        # Validate required sections
        required_sections = ['llm', 'embeddings', 'vector_store', 'text_splitter']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required section: {section}")
        
        # Validate LLM settings
        if 'llm' in self.config:
            llm_config = self.config['llm']
            if not llm_config.get('model'):
                errors.append("LLM model not specified")
            if not isinstance(llm_config.get('temperature', 0), (int, float)):
                errors.append("LLM temperature must be a number")
            if not isinstance(llm_config.get('max_tokens', 0), int):
                errors.append("LLM max_tokens must be an integer")
        
        # Validate text splitter settings
        if 'text_splitter' in self.config:
            ts_config = self.config['text_splitter']
            if not isinstance(ts_config.get('chunk_size', 0), int):
                errors.append("Text splitter chunk_size must be an integer")
            if not isinstance(ts_config.get('chunk_overlap', 0), int):
                errors.append("Text splitter chunk_overlap must be an integer")
        
        # Validate directories exist or can be created
        if 'directories' in self.config:
            for dir_name, dir_path in self.config['directories'].items():
                try:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create directory {dir_name} at {dir_path}: {e}")
        
        return len(errors) == 0, errors
    
    def get_ollama_models(self) -> Dict[str, str]:
        """Get the configured Ollama models"""
        return {
            "llm": self.get("llm.model", "llama3.2"),
            "embedding": self.get("embeddings.model", "mxbai-embed-large")
        }
    
    def print_config(self) -> None:
        """Print current configuration in a readable format"""
        print("="*60)
        print("GENBOTX CONFIGURATION")
        print("="*60)
        
        def print_section(section_name, section_data, indent=0):
            spaces = "  " * indent
            print(f"{spaces}{section_name.upper()}:")
            
            for key, value in section_data.items():
                if isinstance(value, dict):
                    print_section(key, value, indent + 1)
                else:
                    print(f"{spaces}  {key}: {value}")
            print()
        
        for section_name, section_data in self.config.items():
            if isinstance(section_data, dict):
                print_section(section_name, section_data)
            else:
                print(f"{section_name}: {section_data}")
        
        print("="*60)

# Global configuration instance
config = GenBotXConfig()

def get_config() -> GenBotXConfig:
    """Get the global configuration instance"""
    return config

def load_config(config_file: str = "config.yaml") -> GenBotXConfig:
    """Load configuration from a specific file"""
    return GenBotXConfig(config_file)
