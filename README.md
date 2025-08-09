# Local RAG Assistant

A local RAG system setup to learn how to implement a RAG with a local llm. 

## Goals
1. Encode web pages/documents into a database
2. Setup RAG system to extract relevant documents
3. Connect with locally run LLM to ask questions about documents
3. Build chatbot app to use system

## Usage

### Ollama Setup
1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a model (recommended):
```bash
ollama pull mistral
```

## Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/jcbayley/local_rag_assistant.git
cd local_rag_assistant
pip install . 
```

### 2. Add documents to database
```bash
rag-database -db database_name -c collection_name -d path/to/directory/of/documents -f path/to/specific/file
```

### 3. Start the Application
```bash
rag-assistant
```

### 4. Open Your Browser
```bash
rag-assistant-ui
```

Then navigate to `http://localhost:5000`

## Usage Guide

### Web Scraping

```bash
#### Advanced Scraping Options
```bash
# Scrape more pages
rag-scrape --url https://site.com --max-pages 50

# Custom database name
rag-scrape --url https://site.com --dbname custom_name

# Custom collection name  
rag-scrape --url https://site.com --collection-name my_collection

# Save JSON copy of scraped data
rag-scrape --url https://site.com --output scraped_data.json

# Verbose output for debugging
rag-scrape --url https://site.com --verbose
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
- **Collection dropdown** → Select collection or "No Database (Temp docs only)"
- **Click "Switch Database"** → Confirm the change

#### 2. Document Upload (Temporary)
- **Click 📄 button** → **Upload** PDF/TXT/MD/other files
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


## File Structure

```
local_rag_assistant/
├── README.md                    # This file
├── templates/
│   └── index_streaming.html   # Web interface
├── app/
│   └── app.py   # Web interface
├── rag/
│   └── rag_system.py  # Web interface
├── data/
│   └── docuemnt_handler.py   # Web interface
├── uploads/                   # Temporary upload directory
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Ollama** for local AI inference
- **ChromaDB** for vector database
- **Scrapy** for web scraping
- **Flask** for web framework
- **Sentence Transformers** for embeddings

---
