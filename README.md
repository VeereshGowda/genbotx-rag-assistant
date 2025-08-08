# GenBotX - RAG Document Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions from custom documents using LangGraph, ChromaDB, and Llama3.2.

![GenBotX Homepage](images/genbotx_homepage.png)

## Features

### Core RAG Capabilities
- **RAG Pipeline**: LangGraph workflow for document retrieval and answer generation
- **Multi-format Support**: PDF, DOCX, and TXT files
- **Vector Store**: ChromaDB with mxbai-embed-large embeddings
- **Session Memory**: Maintains conversation context
- **Chain-of-Thought**: Shows reasoning steps

### File Management
- **File Upload**: Upload documents through web interface
- **Web Scraping**: Add webpage content to knowledge base
- **Duplicate Detection**: Prevents processing identical content
- **Content Management**: Track all uploaded and scraped content

![File Upload Interface](images/facility_to_upload_custom_files_urls.png)

### Configuration Management
- **YAML Configuration**: External configuration file
- **Web UI**: Configure settings through Streamlit interface
- **Live Updates**: Apply changes without restart
- **Validation**: Automatic parameter checking

![Configuration Manager](images/Ui_config_manager_screen.png)

## Prerequisites

1. **Python 3.12+**
2. **Ollama** with models:
   - `llama3.2` (LLM)
   - `mxbai-embed-large` (embeddings)

Install Ollama models:
```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

## Installation

1. **Clone the project**
2. **Activate virtual environment**:
   ```bash
   .\.venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

## Project Structure

```
GenBotX/
├── src/
│   ├── __init__.py
│   ├── rag_engine.py           # LangGraph RAG pipeline
│   ├── vector_store_manager.py # ChromaDB operations
│   ├── document_processor.py   # Document processing and chunking
│   ├── content_manager.py      # File upload & web scraping
│   ├── wikipedia_scraper.py    # Wikipedia data extraction
│   ├── config.py              # Configuration management
│   └── config_ui.py           # Configuration web interface
├── documents/                  # Document storage
│   ├── Anglo-Mysore Wars- Sujatha kumari R.pdf
│   ├── Krishnadevaraya_Wikipedia.docx
│   └── wikipedia_urls.txt
├── uploads/                    # User uploaded files
├── vector_store/              # ChromaDB persistence
├── logs/                      # Application logs
├── images/                    # UI screenshots
├── config.yaml               # Main configuration file
├── app.py                     # Streamlit web application
├── main.py                    # CLI interface
├── setup.py                   # System initialization
├── test_config.py             # Configuration testing
├── CONFIGURATION_SUMMARY.md   # Configuration documentation
└── README.md
```

## Quick Start

### 1. Initialize System
```bash
python test_config.py  # Test configuration
python setup.py        # Initialize knowledge base
```

### 2. Launch Application
```bash
streamlit run app.py
```

The application includes:
- **Chat Interface**: Main Q&A interface
- **Configuration Manager**: Web-based settings management
- **System Stats**: Performance metrics

![System Statistics](images/genbotx_system_statistics.png)

### 3. CLI Interface
```bash
python main.py
```

## Using the System

### Upload Files
1. Launch Streamlit app
2. Use sidebar "Upload Files" section
3. Select PDF, DOCX, or TXT files
4. Click "Process Uploaded Files"

### Add Webpages
1. Use sidebar "Add Webpages" section
2. Enter URLs (single or multiple)
3. Click "Scrape and Add Webpages"

### Query the System
Ask questions about your uploaded content:

![Query Response Example](images/genbotx_query_response.png)

## Example Queries

- "What is the Gupta Empire?"
- "Tell me about Krishnadevaraya" 
- "What are the Transformers movies about?"
- "Describe the Anglo-Mysore Wars"

## Technical Architecture

### RAG Pipeline
1. **Query Analysis**: Extract key information from user questions
2. **Document Retrieval**: Semantic search using ChromaDB
3. **Context Formation**: Combine relevant document chunks
4. **Response Generation**: Generate answers using Llama3.2
5. **Quality Check**: Confidence scoring and validation

### Components
- **Vector Store**: ChromaDB with mxbai-embed-large embeddings
- **LLM**: Ollama Llama3.2 for response generation
- **Memory**: Conversation context tracking
- **Content Manager**: File upload and web scraping
- **Configuration**: YAML-based external configuration

## Configuration

The system uses `config.yaml` for all settings. Key parameters:

```yaml
llm:
  model: "llama3.2"
  temperature: 0.7
  max_tokens: 2000

vector_store:
  similarity_search_k: 5
  similarity_threshold: 0.7

text_splitter:
  chunk_size: 1000
  chunk_overlap: 200
```

Access the Configuration Manager through the web interface for easy management.

## Testing

```bash
python test_config.py  # Test configuration system
```

## Performance Metrics
- **Document Processing**: ~1000 docs/minute
- **Query Response**: ~2-5 seconds
- **Memory Usage**: ~500MB + documents
- **Vector Store**: Scales to 100k+ documents
