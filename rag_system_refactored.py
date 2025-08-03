"""
Refactored RAG System using DocumentManager class

Clean, maintainable implementation of the RAG system with proper separation of concerns.
"""

import requests
import json
import argparse
from document_manager import DocumentManager


class RAGSystem:
    """
    RAG (Retrieval Augmented Generation) system using DocumentManager for context retrieval
    and Ollama for text generation.
    """
    
    def __init__(self, document_manager: DocumentManager, 
                 ollama_url: str = "http://localhost:11434/api/generate"):
        """
        Initialize RAG system.
        
        Args:
            document_manager: DocumentManager instance for handling documents
            ollama_url: URL for Ollama API endpoint
        """
        self.doc_manager = document_manager
        self.ollama_url = ollama_url
    
    def _stream_response(self, response):
        """
        Generator that yields streaming response chunks.
        
        Args:
            response: requests.Response object from Ollama
            
        Yields:
            str: Individual response chunks
        """
        try:
            for line in response.iter_lines():
                if line:
                    try:
                        data = line.decode("utf-8")
                        json_obj = json.loads(data)
                        if "response" in json_obj:
                            yield json_obj["response"]
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"Warning: Failed to parse response line: {e}")
                        continue
        except Exception as e:
            yield f"Error in streaming: {str(e)}"
    
    def get_model_response(self, prompt: str, model: str = "qwen2.5vl:3b", 
                          temperature: float = 0.1, max_tokens: int = 500, 
                          repeat_penalty: float = 1.3, top_p: float = 0.9,
                          timeout: int = 120, stream: bool = False):
        """
        Get response from Ollama language model.
        
        Args:
            prompt: Input prompt for the model
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            repeat_penalty: Repetition penalty
            top_p: Top-p sampling parameter
            timeout: Request timeout in seconds
            stream: If True, returns generator for streaming; if False, returns complete string
            
        Returns:
            Generated text response (str) or generator for streaming
        """
        try:
            response = requests.post(self.ollama_url, json={
                "model": model,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "repeat_penalty": repeat_penalty,
                    # "top_p": top_p,  # Commented as in original
                }
            }, stream=True, timeout=timeout)
            
            response.raise_for_status()
            
            if stream:
                # Return generator for streaming
                return self._stream_response(response)
            else:
                # Return complete response as before
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            data = line.decode("utf-8")
                            json_obj = json.loads(data)
                            if "response" in json_obj:
                                full_response += json_obj["response"]
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            print(f"Warning: Failed to parse response line: {e}")
                            continue

                return full_response
            
        except requests.exceptions.Timeout:
            print("Error: Request to Ollama timed out (consider reducing max_tokens or checking Ollama status)")
            return "Error: Request timed out. The query may be too complex or Ollama may be overloaded."
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to Ollama. Is it running?")
            return "Error: Could not connect to Ollama"
        except requests.exceptions.RequestException as e:
            print(f"Error: Request failed: {e}")
            return "Error: Request failed"
        except Exception as e:
            print(f"Unexpected error in get_model_response: {e}")
            return "Error: Unexpected error occurred"
    
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
                "model": "qwen2.5vl:3b",
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
        
        # Generate response with single LLM call
        prompt = f"""You are a helpful assistant searching a document archive. Use the provided context to answer the query. 
You must cite the source from each document directly after the fact it supports! 
Do not guess or make up information not found in the context.

Context: {context}

Query: {query}

Provide a bulleted answer with source references and format in markdown. Answer with references to sources:"""

        response = self.get_model_response(prompt, **model_kwargs)
        
        return response, metadata[:5]  # Limit metadata for consistency
    
    def query_stream(self, query: str, top_k: int = 10, verbose: bool = False,
                    use_chromadb: bool = True, use_temp_docs: bool = True,
                    model_kwargs: dict = None):
        """
        Process a query using RAG with streaming response.
        
        Args:
            query: User's question/query
            top_k: Number of documents to retrieve
            verbose: Enable verbose logging
            use_chromadb: Whether to search ChromaDB
            use_temp_docs: Whether to search temporary documents
            model_kwargs: Parameters for the language model
            
        Yields:
            dict: Streaming response chunks with metadata
        """
        if model_kwargs is None:
            model_kwargs = {
                "model": "qwen2.5vl:3b",
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
        
        # Generate response with streaming
        prompt = f"""You are a helpful assistant searching a document archive. Use the provided context to answer the query. 
You must cite the source from each document directly after the fact it supports! 
Do not guess or make up information not found in the context.

Context: {context}

Query: {query}

Provide a bulleted answer with source references and format in markdown. Answer with references to sources:"""

        # Stream the response
        try:
            response_stream = self.get_model_response(prompt, stream=True, **model_kwargs)
            for chunk in response_stream:
                yield {
                    "type": "content",
                    "chunk": chunk
                }
        except Exception as e:
            yield {
                "type": "error",
                "message": f"Error generating response: {str(e)}"
            }
    
    def get_status(self) -> dict:
        """Get current status of the RAG system."""
        return {
            'rag_system': 'active',
            'ollama_url': self.ollama_url,
            'document_manager': self.doc_manager.get_status()
        }


# Global instance for backward compatibility
document_manager = DocumentManager()
rag_system_instance = RAGSystem(document_manager)


def rag_system(query: str, top_k: int = 10, verbose: bool = False,
               model_kwargs: dict = None, use_chromadb: bool = True) -> tuple:
    """
    Backward compatible function that uses the class-based implementation.
    
    This maintains compatibility with existing code while using the new architecture.
    """
    return rag_system_instance.query(
        query=query,
        top_k=top_k,
        verbose=verbose,
        use_chromadb=use_chromadb,
        use_temp_docs=True,  # Always use temp docs for backward compatibility
        model_kwargs=model_kwargs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG system with a query.")
    parser.add_argument("--query", "-q", type=str, required=True, help="The query to ask the RAG system.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output.")
    parser.add_argument("--top-k", "-k", type=int, default=10, help="Number of documents to retrieve.")
    parser.add_argument("--no-chromadb", action="store_true", help="Disable ChromaDB search.")
    parser.add_argument("--no-temp", action="store_true", help="Disable temporary document search.")
    
    args = parser.parse_args()

    # Create RAG system
    doc_manager = DocumentManager()
    rag = RAGSystem(doc_manager)
    
    # Print status
    if args.verbose:
        print("RAG System Status:")
        print(f"Document Manager: {doc_manager}")
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