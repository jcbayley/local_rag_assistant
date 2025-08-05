"""
Document Manager Class for RAG System

Manages both persistent ChromaDB storage and temporary in-memory documents
with a unified interface for searching and document management.
Now using LangChain for document loading and text splitting.
"""

import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ChromaDB imports
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# LangChain imports
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, 
    CSVLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader, UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Embedding model for temporary documents
from sentence_transformers import SentenceTransformer


class DocumentManager:
    """
    Unified document management system for RAG applications.
    
    Handles both persistent ChromaDB storage and temporary in-memory documents
    with consistent embedding and search functionality.
    """
    
    def __init__(self, chromadb_path: str = "./chroma_db", 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 collection_name: str = "chunked_scraped_data"):
        """
        Initialize the DocumentManager.
        
        Args:
            chromadb_path: Path to ChromaDB persistent storage
            embedding_model: SentenceTransformer model name
            collection_name: ChromaDB collection name
        """
        self.chromadb_path = chromadb_path
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        
        # Initialize embedding model
        self.embedding_model = None
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        
        # ChromaDB components
        self.chromadb_client = None
        self.chromadb_collection = None
        
        # Temporary document storage
        self.temp_documents: List[Dict] = []
        
        # Initialize ChromaDB
        self._init_chromadb()
    
    def _init_chromadb(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            self.chromadb_client = chromadb.PersistentClient(path=self.chromadb_path)
            
            try:
                self.chromadb_collection = self.chromadb_client.get_collection(self.collection_name)
                print(f"Connected to existing ChromaDB collection: {self.collection_name}")
            except:
                # Create collection if it doesn't exist
                self.chromadb_collection = self.chromadb_client.create_collection(
                    self.collection_name, 
                    embedding_function=self.embedding_function
                )
                print(f"Created new ChromaDB collection: {self.collection_name}")
                
        except Exception as e:
            print(f"Warning: ChromaDB initialization failed: {e}")
            self.chromadb_client = None
            self.chromadb_collection = None
    
    def _get_embedding_model(self) -> SentenceTransformer:
        """Get or initialize the embedding model for temporary documents."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        return self.embedding_model
    
    def _get_text_splitter(self, chunk_size: int = 500, chunk_overlap: int = 50) -> RecursiveCharacterTextSplitter:
        """Get LangChain text splitter with configurable parameters."""
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def _load_document(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document using LangChain loaders with support for multiple file types."""
        try:
            file_path = Path(file_path)
            file_extension = file_path.suffix.lower()
            
            # Map file extensions to loaders and source types
            loader_map = {
                '.pdf': (PyPDFLoader, 'PDF'),
                '.txt': (TextLoader, 'Text'),
                '.md': (UnstructuredMarkdownLoader, 'Markdown'),
                '.docx': (UnstructuredWordDocumentLoader, 'Word Document'),
                '.doc': (UnstructuredWordDocumentLoader, 'Word Document'),
                '.csv': (CSVLoader, 'CSV'),
                '.html': (UnstructuredHTMLLoader, 'HTML'),
                '.htm': (UnstructuredHTMLLoader, 'HTML'),
                '.pptx': (UnstructuredPowerPointLoader, 'PowerPoint'),
                '.ppt': (UnstructuredPowerPointLoader, 'PowerPoint'),
                '.xlsx': (UnstructuredExcelLoader, 'Excel'),
                '.xls': (UnstructuredExcelLoader, 'Excel')
            }
            
            if file_extension not in loader_map:
                raise ValueError(f"Unsupported file type: {file_extension}. Supported types: {', '.join(loader_map.keys())}")
            
            loader_class, source_type = loader_map[file_extension]
            
            # Initialize loader with appropriate parameters
            if file_extension in ['.txt', '.md']:
                loader = loader_class(str(file_path), encoding='utf-8')
            elif file_extension == '.csv':
                loader = loader_class(str(file_path), encoding='utf-8')
            else:
                loader = loader_class(str(file_path))
            
            # Load documents
            documents = loader.load()
            
            if not documents:
                raise ValueError(f"No content could be extracted from {file_path.name}")
            
            # Add enhanced metadata to all documents
            for doc in documents:
                doc.metadata.update({
                    'filename': file_path.name,
                    'source_type': source_type,
                    'file_extension': file_extension,
                    'file_size': file_path.stat().st_size if file_path.exists() else 0
                })
            
            return documents
            
        except Exception as e:
            print(f"Error loading document {file_path}: {e}")
            return []
    
    # === ChromaDB Methods ===
    
    def add_to_chromadb(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Add document to persistent ChromaDB storage.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if self.chromadb_collection is None:
            return False, "ChromaDB not available"
        
        try:
            file_path = Path(file_path)
            filename = file_path.name
            
            # Load document using LangChain
            documents = self._load_document(file_path)
            if not documents:
                return False, f"No content loaded from {filename}"
            
            # Split documents into chunks
            text_splitter = self._get_text_splitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                return False, f"No chunks created from {filename}"
            
            # Add chunks to ChromaDB
            for i, chunk in enumerate(chunks):
                chunk_id = f"{filename}-{i}-{uuid.uuid4()}"
                
                # Prepare metadata
                metadata = chunk.metadata.copy()
                metadata.update({
                    'url': f"file://{file_path.absolute()}",
                    'chunk_index': i
                })
                
                self.chromadb_collection.add(
                    documents=[chunk.page_content],
                    metadatas=[metadata],
                    ids=[chunk_id]
                )
            
            return True, f"Successfully added {filename} to ChromaDB ({len(chunks)} chunks)"
            
        except Exception as e:
            return False, f"Error adding document to ChromaDB: {str(e)}"
    
    def search_chromadb(self, query: str, top_k: int = 5) -> Tuple[List[str], List[Dict]]:
        """
        Search ChromaDB for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Tuple of (documents: List[str], metadata: List[Dict])
        """
        if self.chromadb_collection is None:
            return [], []
        
        try:
            results = self.chromadb_collection.query(query_texts=[query], n_results=top_k)
            if results['documents'] and results['documents'][0]:
                return results['documents'][0], results['metadatas'][0]
            return [], []
        except Exception as e:
            print(f"Error searching ChromaDB: {e}")
            return [], []
    
    # === Temporary Document Methods ===
    
    def add_temp_document(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Add document to temporary in-memory storage.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            file_path = Path(file_path)
            filename = file_path.name
            
            # Load document using LangChain
            documents = self._load_document(file_path)
            if not documents:
                return False, f"No content loaded from {filename}"
            
            # Split documents into chunks
            text_splitter = self._get_text_splitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                return False, f"No chunks created from {filename}"
            
            # Create embeddings for chunks
            model = self._get_embedding_model()
            doc_id = str(uuid.uuid4())
            chunk_data = []
            
            for i, chunk in enumerate(chunks):
                embedding = model.encode([chunk.page_content])[0]
                
                # Prepare metadata
                metadata = chunk.metadata.copy()
                metadata.update({
                    'chunk_index': i,
                    'doc_id': doc_id
                })
                
                chunk_data.append({
                    'id': f"{doc_id}-{i}",
                    'text': chunk.page_content,
                    'embedding': embedding,
                    'metadata': metadata
                })
            
            # Add to temporary storage
            temp_doc = {
                'doc_id': doc_id,
                'filename': filename,
                'source_type': source_type,
                'chunks': chunk_data,
                'total_chunks': len(chunks)
            }
            
            self.temp_documents.append(temp_doc)
            
            return True, f"Successfully loaded {filename} as temporary document ({len(chunks)} chunks)"
            
        except Exception as e:
            return False, f"Error processing temporary document: {str(e)}"
    
    def search_temp_documents(self, query: str, top_k: int = 5) -> Tuple[List[str], List[Dict]]:
        """
        Search temporary documents for relevant content.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Tuple of (documents: List[str], metadata: List[Dict])
        """
        if not self.temp_documents:
            return [], []
        
        try:
            model = self._get_embedding_model()
            query_embedding = model.encode([query])[0]
            
            # Collect all chunks with similarities
            all_chunks = []
            for doc in self.temp_documents:
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
            
            documents = [chunk['text'] for chunk in top_chunks]
            metadatas = [chunk['metadata'] for chunk in top_chunks]
            
            return documents, metadatas
            
        except Exception as e:
            print(f"Error searching temporary documents: {e}")
            return [], []
    
    def get_temp_documents_info(self) -> List[Dict]:
        """Get information about currently loaded temporary documents."""
        return [{
            'filename': doc['filename'],
            'source_type': doc['source_type'],
            'total_chunks': doc['total_chunks'],
            'doc_id': doc['doc_id']
        } for doc in self.temp_documents]
    
    def clear_temp_documents(self) -> bool:
        """Clear all temporary documents."""
        self.temp_documents = []
        return True
    
    def remove_temp_document(self, doc_id: str) -> bool:
        """Remove a specific temporary document."""
        self.temp_documents = [doc for doc in self.temp_documents if doc['doc_id'] != doc_id]
        return True
    
    # === Unified Search Methods ===
    
    def search(self, query: str, top_k: int = 10, 
               use_chromadb: bool = True, 
               use_temp_docs: bool = True) -> Tuple[str, List[Dict]]:
        """
        Unified search across both ChromaDB and temporary documents.
        
        Args:
            query: Search query
            top_k: Total number of results to return
            use_chromadb: Whether to search ChromaDB
            use_temp_docs: Whether to search temporary documents
            
        Returns:
            Tuple of (formatted_context: str, metadata: List[Dict])
        """
        all_documents = []
        all_metadata = []
        
        # Search temporary documents
        if use_temp_docs:
            temp_k = top_k // 2 + 1 if use_chromadb else top_k
            temp_docs, temp_meta = self.search_temp_documents(query, temp_k)
            if temp_docs:
                all_documents.extend(temp_docs)
                all_metadata.extend(temp_meta)
                print(f"Found {len(temp_docs)} results from temporary documents")
        
        # Search ChromaDB
        if use_chromadb:
            remaining_k = max(1, top_k - len(all_documents))
            chromadb_docs, chromadb_meta = self.search_chromadb(query, remaining_k)
            if chromadb_docs:
                all_documents.extend(chromadb_docs)
                all_metadata.extend(chromadb_meta)
                print(f"Found {len(chromadb_docs)} results from ChromaDB")
        
        if not all_documents:
            return "No documents found", []
        
        # Limit to top_k results
        all_documents = all_documents[:top_k]
        all_metadata = all_metadata[:top_k]
        
        # Format context
        context = "\n\n".join(
            f"Document {i+1}\n"
            f"Source: {all_metadata[i].get('filename', all_metadata[i].get('url', 'Unknown'))}\n"
            f"Content: {all_documents[i]}" 
            for i in range(len(all_documents))
        )
        
        return context, all_metadata
    
    # === Status and Info Methods ===
    
    def switch_database(self, new_chromadb_path: str, new_collection_name: str = None) -> Tuple[bool, str]:
        """
        Switch to a different ChromaDB database and/or collection.
        
        Args:
            new_chromadb_path: Path to the new ChromaDB database
            new_collection_name: Name of collection (uses current if not provided)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Use current collection name if not provided
            if new_collection_name is None:
                new_collection_name = self.collection_name
            
            # Initialize new client
            new_client = chromadb.PersistentClient(path=new_chromadb_path)
            
            try:
                new_collection = new_client.get_collection(new_collection_name)
            except:
                # Create collection if it doesn't exist
                new_collection = new_client.create_collection(
                    new_collection_name, 
                    embedding_function=self.embedding_function
                )
            
            # Switch to new database
            self.chromadb_path = new_chromadb_path
            self.collection_name = new_collection_name
            self.chromadb_client = new_client
            self.chromadb_collection = new_collection
            
            return True, f"Switched to database: {new_chromadb_path}, collection: {new_collection_name}"
            
        except Exception as e:
            return False, f"Failed to switch database: {str(e)}"
    
    def list_available_databases(self, base_path: str = ".") -> List[str]:
        """
        List available ChromaDB databases in the given path.
        
        Args:
            base_path: Base directory to search for databases
            
        Returns:
            List of database directory names
        """
        import os
        databases = []
        
        try:
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path):
                    # Check if it looks like a ChromaDB directory
                    chroma_file = os.path.join(item_path, "chroma.sqlite3")
                    if os.path.exists(chroma_file) or item.endswith("_db") or item == "chroma_db":
                        databases.append(item)
        except Exception as e:
            print(f"Error listing databases: {e}")
        
        return sorted(databases)
    
    def list_collections_in_database(self, db_path: str = None) -> List[str]:
        """
        List collections in a ChromaDB database.
        
        Args:
            db_path: Database path (uses current if not provided)
            
        Returns:
            List of collection names
        """
        try:
            if db_path is None:
                client = self.chromadb_client
            else:
                client = chromadb.PersistentClient(path=db_path)
            
            if client is None:
                return []
                
            collections = client.list_collections()
            return [col.name for col in collections]
            
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []

    def get_status(self) -> Dict:
        """Get current status of the document manager."""
        return {
            'chromadb_available': self.chromadb_collection is not None,
            'chromadb_path': self.chromadb_path,
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model_name,
            'temp_documents_count': len(self.temp_documents),
            'temp_documents': self.get_temp_documents_info(),
            'available_databases': self.list_available_databases(),
            'available_collections': self.list_collections_in_database()
        }
    
    def __repr__(self) -> str:
        status = self.get_status()
        return (f"DocumentManager(chromadb={'✓' if status['chromadb_available'] else '✗'}, "
                f"temp_docs={status['temp_documents_count']}, "
                f"model='{self.embedding_model_name}')")