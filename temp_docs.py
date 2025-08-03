import uuid
import textwrap
from sentence_transformers import SentenceTransformer
import PyPDF2
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Global storage for temporary documents
temp_documents = []
embedding_model = None

def get_embedding_model():
    """Get or initialize the embedding model"""
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model

def chunk_text(text, max_tokens=500):
    """Split text into chunks"""
    return textwrap.wrap(text, width=max_tokens)

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

def add_temp_document(file_path):
    """Add document to temporary storage with embeddings"""
    try:
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
            return False, f"Unsupported file type: {file_path.suffix}"
        
        if not text:
            return False, f"No text extracted from {filename}"
        
        # Split into chunks
        chunks = chunk_text(text, max_tokens=500)
        model = get_embedding_model()
        
        # Create embeddings for chunks
        doc_id = str(uuid.uuid4())
        chunk_data = []
        
        for i, chunk in enumerate(chunks):
            embedding = model.encode([chunk])[0]
            chunk_data.append({
                'id': f"{doc_id}-{i}",
                'text': chunk,
                'embedding': embedding,
                'metadata': {
                    'filename': filename,
                    'source_type': source_type,
                    'chunk_index': i,
                    'doc_id': doc_id
                }
            })
        
        # Add to global storage
        temp_doc = {
            'doc_id': doc_id,
            'filename': filename,
            'source_type': source_type,
            'chunks': chunk_data,
            'total_chunks': len(chunks)
        }
        
        temp_documents.append(temp_doc)
        print(f"Added temporary document: {filename} with {len(chunks)} chunks")
        
        return True, f"Successfully loaded {filename} as temporary document"
        
    except Exception as e:
        return False, f"Error processing document: {str(e)}"

def search_temp_documents(query, top_k=5):
    """Search through temporary documents"""
    if not temp_documents:
        return [], []
    
    try:
        model = get_embedding_model()
        query_embedding = model.encode([query])[0]
        
        # Collect all chunks with similarities
        all_chunks = []
        for doc in temp_documents:
            for chunk in doc['chunks']:
                similarity = cosine_similarity([query_embedding], [chunk['embedding']])[0][0]
                all_chunks.append({
                    'text': chunk['text'],
                    'metadata': chunk['metadata'],
                    'similarity': similarity
                })
        
        # Sort by similarity and get top results
        all_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        top_chunks = all_chunks[:top_k]
        
        # Format results similar to ChromaDB
        documents = [chunk['text'] for chunk in top_chunks]
        metadatas = [chunk['metadata'] for chunk in top_chunks]
        
        return documents, metadatas
        
    except Exception as e:
        print(f"Error searching temporary documents: {e}")
        return [], []

def get_temp_documents_info():
    """Get info about currently loaded temporary documents"""
    return [{
        'filename': doc['filename'],
        'source_type': doc['source_type'],
        'total_chunks': doc['total_chunks'],
        'doc_id': doc['doc_id']
    } for doc in temp_documents]

def clear_temp_documents():
    """Clear all temporary documents"""
    global temp_documents
    temp_documents = []
    return True

def remove_temp_document(doc_id):
    """Remove a specific temporary document"""
    global temp_documents
    temp_documents = [doc for doc in temp_documents if doc['doc_id'] != doc_id]
    return True