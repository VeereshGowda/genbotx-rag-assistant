# Contributing to GenBotX

Thank you for your interest in contributing to GenBotX! This guide will help you get started with contributing to this open-source RAG system project.

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Pull Request Process](#pull-request-process)
6. [Testing](#testing)
7. [Documentation](#documentation)
8. [Issue Reporting](#issue-reporting)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please read and follow our Code of Conduct:

- Be respectful and considerate in all interactions
- Welcome newcomers and help them get started
- Focus on constructive feedback and collaboration
- Respect different viewpoints and experiences
- Report unacceptable behavior to the maintainers

## Getting Started

### Prerequisites for Development

- Python 3.12 or higher
- Git for version control
- Ollama with required models (llama3.2, mxbai-embed-large)
- Understanding of RAG systems and LangChain framework
- Familiarity with Streamlit for web interface development

### Areas for Contribution

We welcome contributions in several areas:

**Core System Development:**
- RAG pipeline optimization and new features
- Document processing improvements
- Vector store management enhancements
- Query optimization and performance improvements

**User Interface Development:**
- Streamlit interface improvements
- Configuration management UI enhancements
- New visualization and analytics features
- Mobile-responsive design improvements

**Documentation and Examples:**
- Tutorial content and usage examples
- API documentation improvements
- Configuration guides and best practices
- Performance optimization guides

**Testing and Quality Assurance:**
- Unit test development and improvement
- Integration testing scenarios
- Performance benchmarking
- Security testing and validation

**Integrations and Extensions:**
- Support for additional document formats
- Integration with new LLM providers
- Database backend alternatives
- Authentication and authorization systems

## Development Setup

### 1. Fork and Clone Repository

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/genbotx-rag-assistant.git
cd genbotx-rag-assistant

# Add upstream remote
git remote add upstream https://github.com/VeereshGowda/genbotx-rag-assistant.git
```

### 2. Development Environment Setup

```bash
# Create development virtual environment
python -m venv .venv-dev

# Activate environment
# Windows:
.\.venv-dev\Scripts\activate
# Linux/Mac:
source .venv-dev/bin/activate

# Install development dependencies
pip install -e ".[dev]"
```

### 3. Pre-commit Hooks Setup

```bash
# Install pre-commit hooks for code quality
pre-commit install

# Run pre-commit on all files (optional)
pre-commit run --all-files
```

### 4. Development Configuration

```bash
# Copy development environment template
cp .env.example .env.dev

# Configure for development
# Edit .env.dev with development settings
```

### 5. Initialize Development System

```bash
# Run configuration tests
python test_config.py

# Initialize development system
python setup.py
```

## Contributing Guidelines

### Branch Naming Convention

Use descriptive branch names that indicate the type of contribution:

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `test/description` - Testing improvements
- `refactor/description` - Code refactoring

Examples:
- `feature/add-multilingual-support`
- `bugfix/fix-pdf-parsing-error`
- `docs/update-installation-guide`

### Commit Message Format

Use clear, descriptive commit messages following conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes

Examples:
```
feat(rag): add support for multilingual document processing

Add language detection and processing for documents in multiple languages.
Includes support for Spanish, French, and German text processing.

Closes #123
```

### Code Style and Standards

**Python Code Standards:**
- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Include docstrings for all functions and classes
- Maintain line length under 88 characters (Black formatter standard)

**Code Quality Tools:**
```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

**Example Code Style:**
```python
from typing import List, Optional, Dict, Any
from pathlib import Path

def process_documents(
    file_paths: List[Path], 
    chunk_size: int = 1000,
    overlap: int = 200
) -> Dict[str, Any]:
    """
    Process multiple documents for RAG system integration.
    
    Args:
        file_paths: List of document file paths to process
        chunk_size: Size of text chunks for processing
        overlap: Overlap size between consecutive chunks
        
    Returns:
        Dictionary containing processing results and metadata
        
    Raises:
        DocumentProcessingError: If document processing fails
    """
    # Implementation here
    pass
```

## Pull Request Process

### 1. Before Creating a Pull Request

- Ensure your branch is up to date with the main branch
- Run all tests and ensure they pass
- Update documentation if needed
- Add or update tests for new functionality
- Verify code style compliance

```bash
# Update your branch
git fetch upstream
git rebase upstream/main

# Run tests
python -m pytest tests/

# Check code style
black --check src/ tests/
flake8 src/ tests/
```

### 2. Creating the Pull Request

- Create a clear, descriptive title
- Include a detailed description of changes
- Reference related issues using GitHub keywords (Closes #123)
- Add screenshots for UI changes
- Mark as draft if work is in progress

**Pull Request Template:**
```markdown
## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Documentation
- [ ] Documentation updated
- [ ] Examples added/updated
- [ ] Configuration guide updated

## Related Issues
Closes #123
```

### 3. Review Process

- Maintainers will review your pull request
- Address feedback and make requested changes
- Ensure all CI checks pass
- Maintain a respectful dialogue during review

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test files
python -m pytest tests/test_document_processor.py

# Run with coverage
python -m pytest --cov=src tests/

# Run performance tests
python -m pytest tests/performance/ -v
```

### Writing Tests

**Unit Test Example:**
```python
import pytest
from pathlib import Path
from src.document_processor import DocumentProcessor

class TestDocumentProcessor:
    """Test suite for DocumentProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create DocumentProcessor instance for testing."""
        return DocumentProcessor()
    
    def test_process_pdf_document(self, processor, tmp_path):
        """Test PDF document processing functionality."""
        # Create test PDF file
        test_pdf = tmp_path / "test.pdf"
        # ... create test file
        
        result = processor.process_document(test_pdf)
        
        assert result is not None
        assert "chunks" in result
        assert len(result["chunks"]) > 0
```

**Integration Test Example:**
```python
import pytest
from src.rag_engine import RAGEngine
from src.config import GenBotXConfig

class TestRAGEngineIntegration:
    """Integration tests for RAG engine."""
    
    @pytest.fixture
    def rag_engine(self):
        """Create RAG engine for integration testing."""
        config = GenBotXConfig("test_config.yaml")
        return RAGEngine(config)
    
    def test_end_to_end_query(self, rag_engine):
        """Test complete query processing pipeline."""
        # Add test documents
        # Process query
        # Verify response quality
        pass
```

## Documentation

### Documentation Standards

- Use clear, concise language
- Include practical examples
- Update relevant sections when making changes
- Follow existing documentation structure

### Types of Documentation

**API Documentation:**
- Document all public functions and classes
- Include parameter descriptions and return values
- Provide usage examples

**User Documentation:**
- Installation and setup guides
- Configuration explanations
- Usage tutorials and examples

**Developer Documentation:**
- Architecture overviews
- Contributing guidelines
- Testing procedures

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs/
make html

# Preview documentation
open _build/html/index.html
```

## Issue Reporting

### Before Creating an Issue

- Search existing issues to avoid duplicates
- Gather relevant information about your environment
- Prepare minimal reproduction steps if reporting a bug

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug.

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Windows 11, Ubuntu 22.04]
- Python version: [e.g., 3.12.1]
- GenBotX version: [e.g., 0.1.0]
- Ollama version: [e.g., 0.5.2]

## Additional Context
Add any other context, screenshots, or logs.
```

### Feature Request Template

```markdown
## Feature Description
Clear description of the proposed feature.

## Use Case
Describe the problem this feature would solve.

## Proposed Solution
Describe your proposed implementation approach.

## Alternatives Considered
Describe alternative solutions you've considered.

## Additional Context
Add any other context or mockups.
```

## Development Best Practices

### Performance Considerations

- Profile code changes that may impact performance
- Include benchmark tests for performance-critical features
- Consider memory usage and scalability implications
- Test with realistic document sizes and query loads

### Security Considerations

- Follow security best practices for file handling
- Validate all user inputs
- Be mindful of potential injection vulnerabilities
- Review security implications of new features

### Compatibility Considerations

- Maintain backward compatibility when possible
- Document breaking changes clearly
- Consider impact on existing configurations
- Test across different operating systems

## Getting Help

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Request Comments**: For code review discussions

### Mentorship and Support

New contributors are welcome! If you need help getting started:

- Look for issues labeled "good first issue"
- Ask questions in GitHub Discussions
- Reach out to maintainers for guidance
- Join community discussions and code reviews

## Recognition

Contributors will be recognized through:

- Inclusion in the CONTRIBUTORS.md file
- Acknowledgment in release notes
- GitHub contributor statistics
- Community recognition for significant contributions

Thank you for contributing to GenBotX! Your efforts help make this project better for everyone.
