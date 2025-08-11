#!/usr/bin/env python3
"""
Command Line Interface for Local RAG Assistant

Interactive CLI tool that works alongside the web app, providing
command-line access to the RAG system with database switching capabilities.
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

from local_rag_assistant.data.document_manager import DocumentManager
from local_rag_assistant.rag.rag_system import RAGSystem
from local_rag_assistant.backends.ollama_backend import OllamaBackend
from local_rag_assistant.backends.transformers_backend import TransformersBackend


class RAGAssistantCLI:
    """Interactive command-line interface for RAG Assistant."""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "chunked_scraped_data", 
                 backend: str = "ollama", ollama_url: str = "http://localhost:11434",
                 hf_model_name: str = "microsoft/DialoGPT-small"):
        """Initialize CLI with specified database, collection, and backend."""
        self.document_manager = DocumentManager(chromadb_path=db_path, collection_name=collection_name)
        self.backend_type = backend
        self.ollama_url = ollama_url
        self.hf_model_name = hf_model_name
        
        # Initialize RAG system with selected backend
        self.rag_system = RAGSystem(
            self.document_manager, 
            backend=backend,
            ollama_url=ollama_url,
            hf_model_name=hf_model_name
        )
        self.running = True
        
        # Default query parameters
        self.default_params = {
            'model': 'smollm2:135m',
            'temperature': 0.4,
            'max_tokens': 100,
            'repeat_penalty': 1.4,
            'top_p': 0.9,
            'top_k': 10,
            'use_chromadb': True,
            'use_temp_docs': True
        }
        
    def display_welcome(self):
        """Display welcome message and current status."""
        print("\n" + "="*60)
        print("🤖 RAG Assistant CLI")
        print("="*60)
        status = self.document_manager.get_status()
        rag_status = self.rag_system.get_status()
        
        print(f"Database: {status['chromadb_path']}")
        print(f"Collection: {status['collection_name']}")
        print(f"ChromaDB Available: {'✓' if status['chromadb_available'] else '✗'}")
        print(f"Temp Documents: {status['temp_documents_count']}")
        print(f"Embedding Model: {status['embedding_model']}")
        
        # Backend information
        backend_status = "✓" if rag_status['backend_available'] else "✗"
        print(f"Backend: {self.backend_type} {backend_status}")
        if self.backend_type == "ollama":
            print(f"Ollama URL: {self.ollama_url}")
        elif self.backend_type == "transformers":
            print(f"HF Model: {self.hf_model_name}")
        
        print("\nType '/help' for available commands or enter a query to start.")
        print("="*60)
    
    def display_help(self):
        """Display available commands."""
        help_text = """
Available Commands:
===================

Query Commands:
  <your question>           - Ask a question using the RAG system
  /query <question>         - Alternative way to ask questions

Document Commands:
  /add-document <path>      - Add document to temporary storage
  /add-permanent <path>     - Add document to permanent ChromaDB
  /list-temp               - List temporary documents
  /clear-temp              - Clear all temporary documents
  /status                  - Show system status

Database Commands:
  /switch-database <path>           - Switch ChromaDB database
  /switch-collection <name>         - Switch to different collection
  /list-databases                   - List available databases
  /list-collections                 - List collections in current database
  /list-collections <db_path>       - List collections in specified database

Backend Commands:
  /switch-backend <backend>         - Switch between 'ollama' and 'transformers'
  /list-backends                    - Show available backends and their status
  /backend-status                   - Show current backend status

Configuration Commands:
  /set <param> <value>     - Set query parameter (model, temperature, etc.)
  /show-params             - Show current parameters
  /reset-params            - Reset parameters to defaults

System Commands:
  /help                    - Show this help message
  /exit, /quit, /q         - Exit the application

Parameters you can set:
- model: LLM model name (default: smollm2:135m)
- temperature: Response randomness (0.0-1.0, default: 0.4)
- max_tokens: Maximum response length (default: 100)
- top_k: Number of documents to retrieve (default: 10)
- use_chromadb: Use ChromaDB search (true/false, default: true)
- use_temp_docs: Use temporary documents (true/false, default: true)
"""
        print(help_text)
    
    def handle_command(self, user_input: str) -> bool:
        """
        Handle user commands and queries.
        
        Args:
            user_input: User's input string
            
        Returns:
            bool: True to continue, False to exit
        """
        user_input = user_input.strip()
        
        if not user_input:
            return True
        
        # Exit commands
        if user_input.lower() in ['/exit', '/quit', '/q']:
            print("👋 Goodbye!")
            return False
        
        # Help command
        elif user_input.lower() == '/help':
            self.display_help()
        
        # Status command
        elif user_input.lower() == '/status':
            self.show_status()
        
        # Document commands
        elif user_input.lower().startswith('/add-document '):
            path = user_input[14:].strip()
            self.add_document(path, permanent=False)
        
        elif user_input.lower().startswith('/add-permanent '):
            path = user_input[15:].strip()
            self.add_document(path, permanent=True)
        
        elif user_input.lower() == '/list-temp':
            self.list_temp_documents()
        
        elif user_input.lower() == '/clear-temp':
            self.clear_temp_documents()
        
        # Database commands
        elif user_input.lower().startswith('/switch-database '):
            path = user_input[17:].strip()
            self.switch_database(path)
        
        elif user_input.lower().startswith('/switch-collection '):
            name = user_input[19:].strip()
            self.switch_collection(name)
        
        elif user_input.lower() == '/list-databases':
            self.list_databases()
        
        elif user_input.lower().startswith('/list-collections'):
            parts = user_input.split()
            db_path = parts[1] if len(parts) > 1 else None
            self.list_collections(db_path)
        
        # Backend commands
        elif user_input.lower().startswith('/switch-backend '):
            backend = user_input[16:].strip()
            self.switch_backend(backend)
        
        elif user_input.lower() == '/list-backends':
            self.list_backends()
        
        elif user_input.lower() == '/backend-status':
            self.show_backend_status()
        
        # Configuration commands
        elif user_input.lower().startswith('/set '):
            self.set_parameter(user_input[5:])
        
        elif user_input.lower() == '/show-params':
            self.show_parameters()
        
        elif user_input.lower() == '/reset-params':
            self.reset_parameters()
        
        # Query commands
        elif user_input.lower().startswith('/query '):
            query = user_input[7:].strip()
            if query:
                self.process_query(query)
        elif user_input.startswith("/"):
            print(f"Command {user_input} is not available, please see /help")
        # Direct query (anything else)
        else:
            self.process_query(user_input)
        
        return True
    
    def show_status(self):
        """Show current system status."""
        status = self.document_manager.get_status()
        print("\n📊 System Status:")
        print(f"  Database Path: {status['chromadb_path']}")
        print(f"  Collection: {status['collection_name']}")
        print(f"  ChromaDB Available: {'✓' if status['chromadb_available'] else '✗'}")
        print(f"  Temp Documents: {status['temp_documents_count']}")
        print(f"  Available Databases: {len(status['available_databases'])}")
        print(f"  Available Collections: {len(status['available_collections'])}")
        
        if status['temp_documents']:
            print("\n  Temporary Documents:")
            for doc in status['temp_documents']:
                print(f"    - {doc['filename']} ({doc['total_chunks']} chunks, {doc['source_type']})")
    
    def add_document(self, path: str, permanent: bool = False):
        """Add document to storage."""
        if not path:
            print("❌ Please provide a file path")
            return
        
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            return
        
        print(f"📄 Adding document: {path}")
        
        if permanent:
            success, message = self.document_manager.add_to_chromadb(path)
            storage_type = "ChromaDB"
        else:
            success, message = self.document_manager.add_temp_document(path)
            storage_type = "temporary storage"
        
        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ Failed to add to {storage_type}: {message}")
    
    def list_temp_documents(self):
        """List temporary documents."""
        docs = self.document_manager.get_temp_documents_info()
        if not docs:
            print("📭 No temporary documents loaded")
            return
        
        print(f"\n📄 Temporary Documents ({len(docs)}):")
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. {doc['filename']} ({doc['total_chunks']} chunks, {doc['source_type']})")
    
    def clear_temp_documents(self):
        """Clear all temporary documents."""
        self.document_manager.clear_temp_documents()
        print("🗑️ Cleared all temporary documents")
    
    def switch_database(self, path: str):
        """Switch to different database."""
        if not path:
            print("❌ Please provide a database path")
            return
        
        success, message = self.document_manager.switch_database(path)
        if success:
            print(f"✅ {message}")
            # Reinitialize RAG system with new document manager
            self.rag_system = RAGSystem(
                self.document_manager,
                backend=self.backend_type,
                ollama_url=self.ollama_url,
                hf_model_name=self.hf_model_name
            )
        else:
            print(f"❌ {message}")
    
    def switch_collection(self, name: str):
        """Switch to different collection in current database."""
        if not name:
            print("❌ Please provide a collection name")
            return
        
        current_db = self.document_manager.chromadb_path
        success, message = self.document_manager.switch_database(current_db, name)
        if success:
            print(f"✅ {message}")
            # Reinitialize RAG system with new document manager
            self.rag_system = RAGSystem(
                self.document_manager,
                backend=self.backend_type,
                ollama_url=self.ollama_url,
                hf_model_name=self.hf_model_name
            )
        else:
            print(f"❌ {message}")
    
    def list_databases(self):
        """List available databases."""
        databases = self.document_manager.list_available_databases()
        if not databases:
            print("📭 No ChromaDB databases found in current directory")
            return
        
        print(f"\n💾 Available Databases ({len(databases)}):")
        current_db = os.path.basename(self.document_manager.chromadb_path)
        for db in databases:
            marker = " ← current" if db == current_db else ""
            print(f"  - {db}{marker}")
    
    def list_collections(self, db_path: Optional[str] = None):
        """List collections in database."""
        if db_path:
            collections = self.document_manager.list_collections_in_database(db_path)
            print(f"\n📚 Collections in {db_path}:")
        else:
            collections = self.document_manager.list_collections_in_database()
            print(f"\n📚 Collections in current database:")
        
        if not collections:
            print("  No collections found")
            return
        
        current_collection = self.document_manager.collection_name
        for collection in collections:
            marker = " ← current" if collection == current_collection and db_path is None else ""
            print(f"  - {collection}{marker}")
    
    def switch_backend(self, backend: str):
        """Switch to different backend."""
        if not backend:
            print("❌ Please provide a backend name")
            return
        
        backend = backend.lower()
        if backend not in ['ollama', 'transformers']:
            print("❌ Backend must be 'ollama' or 'transformers'")
            return
        
        if backend == self.backend_type:
            print(f"ℹ️ Already using {backend} backend")
            return
        
        print(f"🔄 Switching to {backend} backend...")
        
        try:
            # Update backend type
            self.backend_type = backend
            
            # Reinitialize RAG system with new backend
            self.rag_system = RAGSystem(
                self.document_manager,
                backend=backend,
                ollama_url=self.ollama_url,
                hf_model_name=self.hf_model_name
            )
            
            # Check if new backend is available
            status = self.rag_system.get_status()
            if status['backend_available']:
                print(f"✅ Successfully switched to {backend} backend")
            else:
                print(f"⚠️ Switched to {backend} backend, but it's not available")
                
        except Exception as e:
            print(f"❌ Failed to switch backend: {e}")
    
    def list_backends(self):
        """List available backends and their status."""
        print("\n🔧 Available Backends:")
        
        # Check Ollama backend
        print("\n1. Ollama Backend:")
        try:
            ollama_backend = OllamaBackend(base_url=self.ollama_url)
            status = "✓ Available" if ollama_backend.is_available() else "✗ Not available"
            marker = " ← current" if self.backend_type == "ollama" else ""
            print(f"   Status: {status}{marker}")
            print(f"   URL: {self.ollama_url}")
        except Exception as e:
            print(f"   Status: ✗ Error - {e}")
        
        # Check Transformers backend
        print("\n2. Transformers Backend:")
        try:
            transformers_backend = TransformersBackend(model_name=self.hf_model_name)
            status = "✓ Available" if transformers_backend.is_available() else "✗ Not available"
            marker = " ← current" if self.backend_type == "transformers" else ""
            print(f"   Status: {status}{marker}")
            print(f"   Model: {self.hf_model_name}")
            info = transformers_backend.get_model_info()
            print(f"   Device: {info['device']}")
            print(f"   CUDA Available: {info['cuda_available']}")
        except Exception as e:
            print(f"   Status: ✗ Error - {e}")
    
    def show_backend_status(self):
        """Show current backend status."""
        status = self.rag_system.get_status()
        print(f"\n🔧 Current Backend: {self.backend_type}")
        print(f"Status: {'✓ Available' if status['backend_available'] else '✗ Not available'}")
        
        if self.backend_type == "ollama":
            print(f"URL: {self.ollama_url}")
        elif self.backend_type == "transformers":
            if 'model_info' in status:
                info = status['model_info']
                print(f"Model: {info['model_name']}")
                print(f"Device: {info['device']}")
                print(f"Loaded: {info['loaded']}")
                print(f"CUDA Available: {info['cuda_available']}")
    
    def set_parameter(self, param_str: str):
        """Set configuration parameter."""
        try:
            parts = param_str.split(None, 1)
            if len(parts) != 2:
                print("❌ Usage: /set <parameter> <value>")
                return
            
            param, value = parts
            param = param.lower()
            
            if param not in self.default_params:
                print(f"❌ Unknown parameter: {param}")
                print(f"Available: {', '.join(self.default_params.keys())}")
                return
            
            # Type conversion based on parameter
            if param in ['temperature', 'top_p', 'repeat_penalty']:
                value = float(value)
            elif param in ['max_tokens', 'top_k']:
                value = int(value)
            elif param in ['use_chromadb', 'use_temp_docs']:
                value = value.lower() in ['true', '1', 'yes', 'on']
            # model stays as string
            
            self.default_params[param] = value
            print(f"✅ Set {param} = {value}")
            
        except ValueError as e:
            print(f"❌ Invalid value for parameter: {e}")
    
    def show_parameters(self):
        """Show current parameters."""
        print("\n⚙️ Current Parameters:")
        for param, value in self.default_params.items():
            print(f"  {param}: {value}")
    
    def reset_parameters(self):
        """Reset parameters to defaults."""
        self.default_params = {
            'model': 'smollm2:135m',
            'temperature': 0.4,
            'max_tokens': 100,
            'repeat_penalty': 1.4,
            'top_p': 0.9,
            'top_k': 10,
            'use_chromadb': True,
            'use_temp_docs': True
        }
        print("✅ Parameters reset to defaults")
    
    def process_query(self, query: str):
        """Process a user query through the RAG system."""
        if not query.strip():
            print("❌ Please provide a query")
            return
        
        print(f"\n🤔 Query: {query}")
        print("🔍 Searching documents...")
        
        try:
            # Prepare model kwargs
            model_kwargs = {
                'model': self.default_params['model'],
                'temperature': self.default_params['temperature'],
                'max_tokens': self.default_params['max_tokens'],
                'repeat_penalty': self.default_params['repeat_penalty'],
                'top_p': self.default_params['top_p']
            }
            
            # Get response from RAG system
            response, metadata = self.rag_system.query(
                query=query,
                top_k=self.default_params['top_k'],
                verbose=True,
                use_chromadb=self.default_params['use_chromadb'],
                use_temp_docs=self.default_params['use_temp_docs'],
                model_kwargs=model_kwargs
            )
            
            print(f"\n🤖 Response:")
            print("-" * 50)
            print(response)
            
            # Show sources
            if metadata:
                print(f"\n📚 Sources ({len(metadata)}):")
                for i, meta in enumerate(metadata, 1):
                    if 'url' in meta:
                        print(f"  {i}. {meta['url']}")
                    elif 'filename' in meta:
                        print(f"  {i}. {meta['filename']} ({meta.get('source_type', 'Document')})")
                    else:
                        print(f"  {i}. Unknown source")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    def run(self):
        """Main interactive loop."""
        self.display_welcome()
        
        try:
            while self.running:
                try:
                    user_input = input("\n💬 > ").strip()
                    if not self.handle_command(user_input):
                        break
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye! (Ctrl+C)")
                    break
                except EOFError:
                    print("\n👋 Goodbye!")
                    break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return 1
        
        return 0


def main():
    """Main entry point for CLI application."""
    parser = argparse.ArgumentParser(
        description="RAG Assistant Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rag-assistant-cli                                      # Use defaults (ollama backend)
  rag-assistant-cli -b transformers                      # Use transformers backend
  rag-assistant-cli -db ./my_db -c my_docs -b ollama     # Custom database with ollama
  rag-assistant-cli --backend transformers --hf-model microsoft/DialoGPT-medium
  rag-assistant-cli --ollama-url http://localhost:11434 --backend ollama
        """
    )
    
    parser.add_argument(
        '-db', '--database',
        default='./chroma_db',
        help='ChromaDB database path (default: ./chroma_db)'
    )
    
    parser.add_argument(
        '-c', '--collection',
        default='chunked_scraped_data',
        help='Collection name (default: chunked_scraped_data)'
    )
    
    parser.add_argument(
        '-b', '--backend',
        choices=['ollama', 'transformers'],
        default='ollama',
        help='Backend to use for text generation (default: ollama)'
    )
    
    parser.add_argument(
        '--ollama-url',
        default='http://localhost:11434',
        help='Ollama base URL (default: http://localhost:11434)'
    )
    
    parser.add_argument(
        '--hf-model',
        default='HuggingFaceTB/SmolLM2-360M-Instruct',
        help='HuggingFace model name for transformers backend (default: microsoft/DialoGPT-small)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='RAG Assistant CLI v0.1.0'
    )
    
    args = parser.parse_args()
    
    try:
        # Create CLI instance and run
        cli = RAGAssistantCLI(
            db_path=args.database, 
            collection_name=args.collection,
            backend=args.backend,
            ollama_url=args.ollama_url,
            hf_model_name=args.hf_model
        )
        return cli.run()
    except Exception as e:
        print(f"❌ Failed to start CLI: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())