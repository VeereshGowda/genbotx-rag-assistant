# GenBotX Configuration Guide

## Overview

GenBotX uses a YAML-based configuration system that allows you to customize the RAG pipeline without modifying code.

## Features

### File Upload & Management
- Multi-format support: PDF, DOCX, TXT files
- Drag & drop upload through web interface
- Duplicate detection using content hashing
- File validation and size limits

### Webpage Scraping
- Single URL or multiple URL input
- Content extraction using BeautifulSoup
- Automatic content cleaning
- Configurable request delays

### Configuration System
- YAML configuration file
- Web-based configuration manager
- Runtime validation
- Export/import capabilities

## Configuration Parameters

### LLM Settings
```yaml
llm:
  model: "llama3.2"              # Ollama model name
  temperature: 0.7               # Response creativity (0.0-1.0)
  max_tokens: 2000              # Maximum response length
  timeout: 60                   # Request timeout in seconds
```

### Vector Store
```yaml
vector_store:
  collection_name: "genbotx_documents"
  similarity_search_k: 5         # Number of documents to retrieve
  similarity_threshold: 0.7      # Minimum relevance score
  persist_directory: "vector_store"
```

### Text Processing
```yaml
text_splitter:
  chunk_size: 1000              # Characters per chunk
  chunk_overlap: 200            # Overlap between chunks
  separators: ["\n\n", "\n", " ", ""]
```

### Web Scraping
```yaml
web_scraping:
  request_delay: 1.0            # Delay between requests
  timeout: 30                   # Request timeout
  max_retries: 3                # Retry attempts
  user_agent: "GenBotX/1.0 Web Scraper"
```

## Usage

### Basic Setup
```bash
# Activate environment
.\.venv\Scripts\activate

# Test configuration
python test_config.py

# Run application
streamlit run app.py
```

### Upload Files
1. Open Streamlit app
2. Go to sidebar "Upload Files"
3. Select files (PDF, DOCX, TXT)
4. Click "Process Uploaded Files"

### Add Webpages
1. Go to sidebar "Add Webpages"
2. Enter URLs (single or multiple)
3. Click "Scrape and Add Webpages"

### Manage Configuration
1. Navigate to "Configuration Manager"
2. Modify settings through web interface
3. Export/import configurations
4. Validate settings

## Configuration Management

### Programmatic Access
```python
from src.config import get_config

config = get_config()

# Get values
llm_model = config.get("llm.model")
chunk_size = config.get("text_splitter.chunk_size")

# Set values
config.set("llm.temperature", 0.8)
config.save_config()
```

### Web Interface Features
- Live configuration updates
- Real-time validation
- Export/import YAML files
- Reset to defaults

## Testing

### Automated Validation
```bash
python test_config.py
```

### Manual Testing
1. Upload different file types
2. Test duplicate detection
3. Scrape various websites
4. Modify configuration settings
5. Query the knowledge base
