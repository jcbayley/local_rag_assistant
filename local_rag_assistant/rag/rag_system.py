"""
Refactored RAG System using DocumentManager class

Clean, maintainable implementation of the RAG system with proper separation of concerns.
Now using LangChain for enhanced RAG capabilities.
"""

import requests
import json
import argparse
from typing import Union, Optional
from local_rag_assistant.data.document_manager import DocumentManager
from local_rag_assistant.backends.ollama_backend import OllamaBackend
from local_rag_assistant.backends.transformers_backend import TransformersBackend

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


class RAGSystem:
    """
    RAG (Retrieval Augmented Generation) system using DocumentManager for context retrieval
    and configurable backends for text generation, enhanced with LangChain components.
    """
    
    def __init__(self, document_manager: DocumentManager, 
                 backend: str = "ollama",
                 ollama_url: str = "http://localhost:11434",
                 hf_model_name: str = "microsoft/DialoGPT-small"):
        """
        Initialize RAG system.
        
        Args:
            document_manager: DocumentManager instance for handling documents
            backend: Backend to use ("ollama" or "transformers")
            ollama_url: Base URL for Ollama API
            hf_model_name: HuggingFace model name for transformers backend
        """
        self.doc_manager = document_manager
        self.backend_type = backend
        
        # Initialize the selected backend
        if backend == "ollama":
            self.backend = OllamaBackend(base_url=ollama_url)
        elif backend == "transformers":
            self.backend = TransformersBackend(model_name=hf_model_name)
        else:
            raise ValueError(f"Unsupported backend: {backend}. Use 'ollama' or 'transformers'")
            
        # Keep ollama_url for backward compatibility
        self.ollama_url = f"{ollama_url}/api/generate" if backend == "ollama" else None
        self._setup_prompt_template()
    
    def _setup_prompt_template(self):
        """Setup LangChain prompt template for consistent prompting."""
        self.prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template="""You are a helpful assistant searching a document archive. Use the provided context to answer the query. 
You must cite the source document and chunk number directly after each fact! 
Note that multiple chunks may come from the same document - always specify both the document name and chunk number.
Do not guess or make up information not found in the context.

Context: {context}

Query: {query}

Provide a bulleted answer with source references (document name and chunk number) and format in markdown. Answer with references to sources:"""
        )

    def _stream_response(self, response, include_logprobs=False):
        """
        Generator that yields streaming response chunks.
        
        Args:
            response: requests.Response object from Ollama
            include_logprobs: Whether to include token probabilities
            
        Yields:
            dict or str: Response data with optional logprobs, or just text chunks
        """
        try:
            for line in response.iter_lines():
                if line:
                    try:
                        data = line.decode("utf-8")
                        json_obj = json.loads(data)
                        
                        if include_logprobs and "response" in json_obj:
                            # Debug: print the full json_obj to see what Ollama returns
                            print(f"Debug - Full Ollama response: {json_obj}")
                            
                            # Return full data including logprobs if available
                            logprobs_data = json_obj.get("logprobs", None)
                            
                            yield {
                                "text": json_obj["response"],
                                "logprobs": logprobs_data
                            }
                        elif "response" in json_obj:
                            # Backward compatibility - just return text
                            yield json_obj["response"]
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"Warning: Failed to parse response line: {e}")
                        continue
        except Exception as e:
            if include_logprobs:
                yield {"text": f"Error in streaming: {str(e)}", "logprobs": None}
            else:
                yield f"Error in streaming: {str(e)}"
    
    def get_model_response(self, prompt: str, model: str = "smollm2:135m", 
                          temperature: float = 0.4, max_tokens: int = 500, 
                          repeat_penalty: float = 1.3, top_p: float = 0.9,
                          timeout: int = 120, stream: bool = False, 
                          include_logprobs: bool = False):
        """
        Get response from the configured backend.
        
        Args:
            prompt: Input prompt for the model
            model: Model name to use (for Ollama backend)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            repeat_penalty: Repetition penalty (Ollama only)
            top_p: Top-p sampling parameter
            timeout: Request timeout in seconds (Ollama only)
            stream: If True, returns generator for streaming; if False, returns complete string
            include_logprobs: If True, request token probabilities (Ollama only)
            
        Returns:
            Generated text response (str) or generator for streaming
        """
        try:
            if self.backend_type == "ollama":
                return self._get_ollama_response(
                    prompt, model, temperature, max_tokens, repeat_penalty, 
                    top_p, timeout, stream, include_logprobs
                )
            else:  # transformers
                return self.backend.generate_response(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=stream
                )
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            if stream:
                return self._error_generator(error_msg)
            return error_msg
    
    def _get_ollama_response(self, prompt, model, temperature, max_tokens, 
                           repeat_penalty, top_p, timeout, stream, include_logprobs):
        """Get response from Ollama backend with legacy support."""
        return self.backend.generate_response(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            repeat_penalty=repeat_penalty,
            top_p=top_p
        )
    
    def _error_generator(self, error_msg: str):
        """Generate error message for streaming."""
        yield error_msg
    
    def query(self, query: str, top_k: int = 10, verbose: bool = False,
              use_chromadb: bool = True, use_temp_docs: bool = True,
              model_kwargs: dict = None) -> tuple:
        """
        Process a query using RAG (Retrieval Augmented Generation).
        
        Args:
            query: User's question/query
            top_k: Number of documents to retrieve
            verbose: Enable verbose logging
            use_chromadb: Whether to search ChromaDB
            use_temp_docs: Whether to search temporary documents
            model_kwargs: Parameters for the language model
            
        Returns:
            Tuple of (response: str, metadata: List[Dict])
        """
        if model_kwargs is None:
            model_kwargs = {
                "model": "smollm2:135m",
                "temperature": 0.1,
                "max_tokens": 1000,
                "repeat_penalty": 1.2,
                "top_p": 0.9
            }
        
        if verbose:
            print(f"Query: {query}")
            print(f"Search settings - ChromaDB: {use_chromadb}, Temp docs: {use_temp_docs}")
        
        # Retrieve relevant documents
        context, metadata = self.doc_manager.search(
            query=query,
            top_k=top_k,
            use_chromadb=use_chromadb,
            use_temp_docs=use_temp_docs
        )
        
        if verbose:
            print(f"Retrieved {len(metadata)} documents")
        
        # Generate response using LangChain prompt template
        prompt = self.prompt_template.format(context=context, query=query)
        response = self.get_model_response(prompt, **model_kwargs)
        
        return response, metadata[:5]  # Limit metadata for consistency
    
    def query_stream(self, query: str, top_k: int = 10, verbose: bool = False,
                    use_chromadb: bool = True, use_temp_docs: bool = True,
                    model_kwargs: dict = None, include_logprobs: bool = False):
        """
        Process a query using RAG with streaming response.
        
        Args:
            query: User's question/query
            top_k: Number of documents to retrieve
            verbose: Enable verbose logging
            use_chromadb: Whether to search ChromaDB
            use_temp_docs: Whether to search temporary documents
            model_kwargs: Parameters for the language model
            include_logprobs: Whether to include token probabilities
            
        Yields:
            dict: Streaming response chunks with metadata and optional logprobs
        """
        if model_kwargs is None:
            model_kwargs = {
                "model": "smollm2:135m",
                "temperature": 0.1,
                "max_tokens": 1000,
                "repeat_penalty": 1.2,
                "top_p": 0.9
            }
        
        if verbose:
            print(f"Query: {query}")
            print(f"Search settings - ChromaDB: {use_chromadb}, Temp docs: {use_temp_docs}")
        
        # Retrieve relevant documents
        context, metadata = self.doc_manager.search(
            query=query,
            top_k=top_k,
            use_chromadb=use_chromadb,
            use_temp_docs=use_temp_docs
        )
        
        if verbose:
            print(f"Retrieved {len(metadata)} documents")
        
        # Send initial metadata
        yield {
            "type": "metadata",
            "sources_count": len(metadata),
            "metadata": metadata[:5]
        }
        
        # Generate response using LangChain prompt template with streaming
        prompt = self.prompt_template.format(context=context, query=query)
        
        # Stream the response
        try:
            response_stream = self.get_model_response(
                prompt, 
                stream=True, 
                include_logprobs=include_logprobs,
                **model_kwargs
            )
            for chunk in response_stream:
                if include_logprobs and isinstance(chunk, dict):
                    yield {
                        "type": "content",
                        "chunk": chunk["text"],
                        "logprobs": chunk["logprobs"]
                    }
                else:
                    yield {
                        "type": "content",
                        "chunk": chunk if isinstance(chunk, str) else chunk["text"]
                    }
        except Exception as e:
            yield {
                "type": "error",
                "message": f"Error generating response: {str(e)}"
            }
    
    def get_status(self) -> dict:
        """Get current status of the RAG system."""
        status = {
            'rag_system': 'active',
            'backend_type': self.backend_type,
            'backend_available': self.backend.is_available(),
            'document_manager': self.doc_manager.get_status()
        }
        
        # Add backend-specific info
        if self.backend_type == "ollama":
            status['ollama_url'] = self.ollama_url
        elif self.backend_type == "transformers":
            status['model_info'] = self.backend.get_model_info()
            
        return status



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG system with a query.")
    parser.add_argument("--query", "-q", type=str, required=True, help="The query to ask the RAG system.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output.")
    parser.add_argument("--top-k", "-k", type=int, default=10, help="Number of documents to retrieve.")
    parser.add_argument("--no-chromadb", action="store_true", help="Disable ChromaDB search.")
    parser.add_argument("--no-temp", action="store_true", help="Disable temporary document search.")
    parser.add_argument("--backend", "-b", type=str, choices=["ollama", "transformers"], 
                       default="ollama", help="Backend to use for text generation: 'ollama' for Ollama API or 'transformers' for local HuggingFace models.")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                       help="Ollama base URL (used with --backend ollama).")
    parser.add_argument("--hf-model", type=str, default="microsoft/DialoGPT-small",
                       help="HuggingFace model name for transformers backend (used with --backend transformers). Examples: 'microsoft/DialoGPT-medium', 'microsoft/DialoGPT-large'.")
    parser.add_argument("--list-backends", action="store_true", 
                       help="Show available backends and their status, then exit.")
    
    args = parser.parse_args()

    # Handle --list-backends option
    if args.list_backends:
        print("Available backends:")
        print("\n1. Ollama Backend:")
        try:
            ollama_backend = OllamaBackend(base_url=args.ollama_url)
            status = "✓ Available" if ollama_backend.is_available() else "✗ Not available"
            print(f"   Status: {status}")
            print(f"   URL: {args.ollama_url}")
        except Exception as e:
            print(f"   Status: ✗ Error - {e}")
        
        print("\n2. Transformers Backend:")
        try:
            transformers_backend = TransformersBackend(model_name=args.hf_model)
            status = "✓ Available" if transformers_backend.is_available() else "✗ Not available"
            print(f"   Status: {status}")
            print(f"   Model: {args.hf_model}")
            info = transformers_backend.get_model_info()
            print(f"   Device: {info['device']}")
            print(f"   CUDA Available: {info['cuda_available']}")
        except Exception as e:
            print(f"   Status: ✗ Error - {e}")
        
        print(f"\nCurrently selected: {args.backend}")
        

    # Create RAG system
    doc_manager = DocumentManager()
    rag = RAGSystem(
        doc_manager, 
        backend=args.backend,
        ollama_url=args.ollama_url,
        hf_model_name=args.hf_model
    )
    
    # Print status
    if args.verbose:
        print("RAG System Status:")
        status = rag.get_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        print()
    
    # Run query
    response, metadata = rag.query(
        query=args.query,
        top_k=args.top_k,
        verbose=args.verbose,
        use_chromadb=not args.no_chromadb,
        use_temp_docs=not args.no_temp
    )
    
    print(f"Query: {args.query}")
    print(f"Response: {response}")