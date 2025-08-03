# Local RAG Assistant

A Retrieval Augmented Generation (RAG) system that runs locally using on uploaded documents or scraped web pages. Built with Flask, ChromaDB, and Ollama for local AI inference.

## Features

- **Web Scraping**: Automatically scrape websites and add to searchable database
- **Document Upload**: Upload PDFs, text files, and markdown documents  
- **Multiple Databases**: Switch between different knowledge bases
- **Temporary Documents**: Upload files for temporary analysis without permanent storage
- **Local AI**: Uses Ollama for privacy-focused local AI inference

## Prerequisites

### Required Software
- **Python 3.8+**
- **Ollama**: Download from [ollama.ai](https://ollama.ai)

### Python Dependencies
```bash
pip install flask scrapy requests beautifulsoup4 chromadb sentence-transformers 
pip install lxml readability-python markdown scikit-learn PyPDF2 w3lib
```

### Ollama Setup
1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a model (recommended):
```bash
ollama pull qwen2.5vl:3b
# or for larger/better model:
ollama pull llama3.1:8b
```

## Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo>
cd local_rag_assistant
pip install -r requirements.txt  # if you create one
```

### 2. Scrape Your First Website
```bash
# Scrape Python documentation (creates python_docs_db)
python scrape_web.py --url https://docs.python.org --max-pages 20 --verbose

# Scrape with custom database name
python scrape_web.py --url https://your-blog.com --dbname my_blog --max-pages 10
```

### 3. Start the Application
```bash
python app_refactored.py
```

### 4. Open Your Browser
Navigate to `http://localhost:5000`

## Usage Guide

### Web Scraping

#### Basic Website Scraping
```bash
# Automatic naming (recommended)
python scrape_web.py --url https://docs.python.org

# Creates:
# - Database: ./docs_python_org_db/
# - Collection: docs_python_org_db
```

#### Advanced Scraping Options
```bash
# Scrape more pages
python scrape_web.py --url https://site.com --max-pages 50

# Custom database name
python scrape_web.py --url https://site.com --dbname custom_name

# Custom collection name  
python scrape_web.py --url https://site.com --collection-name my_collection

# Save JSON copy of scraped data
python scrape_web.py --url https://site.com --output scraped_data.json

# Verbose output for debugging
python scrape_web.py --url https://site.com --verbose
```

#### Website Name Generation
URLs are automatically converted to clean database names:

| URL | Generated Database | Generated Collection |
|-----|-------------------|---------------------|
| `https://docs.python.org` | `docs_python_org_db` | `docs_python_org_db` |
| `https://www.github.com` | `github_db` | `github_db` |
| `https://blog.example.com` | `blog_example_db` | `blog_example_db` |
| `https://react.dev` | `react_db` | `react_db` |

### Using the Chat Interface

#### 1. Database Selection
- **Settings Panel** → **Database dropdown** → Select your scraped database
- **Collection dropdown** → Select collection or "🚫 No Database (Temp docs only)"
- **Click "Switch Database"** → Confirm the change

#### 2. Document Upload (Temporary)
- **Click 📄 button** → **Upload** PDF/TXT/MD files
- **Temporary documents** → Available immediately for search
- **Clear All button** → Remove temporary documents

#### 3. Chat Settings
- **Streaming Toggle**: 🔴 ON for real-time responses, ⚫ OFF for complete responses
- **Model Selection**: Choose from available Ollama models
- **Parameters**: Adjust temperature, max tokens, etc.

### Database Management

#### List Available Databases
The UI automatically discovers databases in your directory:
```
./chroma_db/           # Default database
./python_docs_db/      # Python documentation  
./react_db/           # React documentation
./my_blog_db/         # Custom blog database
```

#### Switch Between Databases
1. **Settings panel** → **Database dropdown**
2. **Select database** → **Select collection**  
3. **Click "Switch Database"**
4. **Start querying** the new database

## Advanced Usage

### Command Line RAG Queries
```bash
# Direct command line usage
python rag_system_refactored.py --query "How do I use pandas DataFrames?" --verbose
```


### Performance Tips

#### For Large Websites
```bash
# Limit pages for faster processing
python scrape_web.py --url https://large-site.com --max-pages 50

# Use smaller models for faster responses
# In UI: Select "qwen2.5vl:3b" instead of larger models
```


## File Structure

```
local_rag_assistant/
├── README.md                    # This file
├── app_refactored.py           # Flask web application
├── scrape_web.py               # Web scraping tool  
├── document_manager.py         # Database management class
├── rag_system_refactored.py    # RAG system implementation
├── temp_docs.py               # Temporary document handling
├── templates/
│   └── index_streaming.html   # Web interface
├── uploads/                   # Temporary upload directory
└── databases/                 # Your scraped databases
    ├── chroma_db/            # Default database
    ├── python_docs_db/       # Example: Python docs
    └── react_db/             # Example: React docs
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Ollama** for local AI inference
- **ChromaDB** for vector database
- **Scrapy** for web scraping
- **Flask** for web framework
- **Sentence Transformers** for embeddings

---
