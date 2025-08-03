import chromadb
import json
import argparse
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import textwrap
import uuid
import PyPDF2
import os
from pathlib import Path

# Function to load a collection
def load_collection():
    # Initialize ChromaDB client with the same path used for persistence
    client = chromadb.PersistentClient(path="./chroma_db")

    # Load the collection from disk
    collection = client.get_collection(name="scraped_data")

    # Query the collection
    results = collection.query(query_texts=["This is a query"], n_results=2)
    return results

def chunk_text(text, max_tokens=500):
    # Naive splitting by sentence or characters
    # You can replace with token-based chunking if needed
    return textwrap.wrap(text, width=max_tokens)

def create_collection(input_data="scraped_data.json", output_path="./chroma_db"):
    # Initialize ChromaDB client with the same path used for persistence
    client = chromadb.PersistentClient(path=output_path)
    # Load the JSON file
    with open(input_data, 'r') as f:
        scraped_data = json.load(f)

    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    # Create a collection in ChromaDB
    collection = client.create_collection(name="chunked_scraped_data", embedding_function=embedding_fn)

    doc_count = 0
    for data in scraped_data:
        text = data['text']
        url = data['url']

        # Split into smaller chunks
        chunks = chunk_text(text, max_tokens=500)

        for chunk in chunks:
            # Generate a unique ID for each chunk
            chunk_id = f"{doc_count}-{uuid.uuid4()}"
            collection.add(
                documents=[chunk],
                metadatas=[{'url': url}],
                ids=[chunk_id]
            )
        doc_count += 1

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

def extract_text_from_file(file_path):
    """Extract text from text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error reading text file {file_path}: {e}")
        return ""

def add_document_to_collection(file_path, client_path="./chroma_db"):
    """Add a single document (PDF or text) to ChromaDB collection using same embedding function"""
    try:
        # Initialize client with same embedding function as create_collection
        client = chromadb.PersistentClient(path=client_path)
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        try:
            collection = client.get_collection("chunked_scraped_data")
        except:
            collection = client.create_collection("chunked_scraped_data", embedding_function=embedding_fn)
        
        file_path = Path(file_path)
        filename = file_path.name
        
        # Extract text based on file type
        if file_path.suffix.lower() == '.pdf':
            text = extract_text_from_pdf(file_path)
            source_type = "PDF"
        elif file_path.suffix.lower() in ['.txt', '.md']:
            text = extract_text_from_file(file_path)
            source_type = "Text"
        else:
            print(f"Unsupported file type: {file_path.suffix}")
            return False
        
        if not text:
            print(f"No text extracted from {filename}")
            return False
        
        # Split into chunks using same function
        chunks = chunk_text(text, max_tokens=500)
        
        print(f"Adding {len(chunks)} chunks from {filename}")
        
        # Add chunks to collection
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}-{i}-{uuid.uuid4()}"
            collection.add(
                documents=[chunk],
                metadatas=[{
                    'url': f"file://{file_path.absolute()}",
                    'filename': filename,
                    'source_type': source_type,
                    'chunk_index': i
                }],
                ids=[chunk_id]
            )
        
        print(f"Successfully added {filename} to ChromaDB")
        return True
        
    except Exception as e:
        print(f"Error adding document {file_path}: {e}")
        return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create or load a ChromaDB collection.")
    parser.add_argument("--input", "-i", type=str, default="scraped_data.json",help="Path to the input JSON file containing scraped data.")
    args = parser.parse_args()
    # Create the collection from the scraped data JSON file
    create_collection(args.input)


