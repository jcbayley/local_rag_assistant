"""
Refactored RAG System using DocumentManager class

Clean, maintainable implementation of the RAG system with proper separation of concerns.
Now using LangChain for enhanced RAG capabilities.
"""

import requests
import json
import argparse
from local_rag_assistant.data.document_manager import DocumentManager

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


class RAGSystem:
    """
    RAG (Retrieval Augmented Generation) system using DocumentManager for context retrieval
    and Ollama for text generation, enhanced with LangChain components.
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
        self._setup_prompt_template()
    
    def _setup_prompt_template(self):
        """Setup LangChain prompt template for consistent prompting."""
        self.prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template="""You are a helpful assistant searching a document archive. Use the provided context to answer the query. 
You must cite the source from each document directly after the fact it supports! 
Do not guess or make up information not found in the context.

Context: {context}

Query: {query}

Provide a bulleted answer with source references and format in markdown. Answer with references to sources:"""
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
                          temperature: float = 0.1, max_tokens: int = 500, 
                          repeat_penalty: float = 1.3, top_p: float = 0.9,
                          timeout: int = 120, stream: bool = False, 
                          include_logprobs: bool = False):
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
            include_logprobs: If True, request token probabilities from Ollama
            
        Returns:
            Generated text response (str) or generator for streaming (with optional logprobs)
        """
        try:
            # Build request payload
            payload = {
                "model": model,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "repeat_penalty": repeat_penalty,
                    # "top_p": top_p,  # Commented as in original
                }
            }
            
            # Add logprobs request if needed
            if include_logprobs:
                payload["options"]["logprobs"] = True
                payload["options"]["top_logprobs"] = 5  # Get top 5 token probabilities
            
            response = requests.post(self.ollama_url, json=payload, stream=True, timeout=timeout)
            
            response.raise_for_status()
            
            if stream:
                # Return generator for streaming
                return self._stream_response(response, include_logprobs=include_logprobs)
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
        return {
            'rag_system': 'active',
            'ollama_url': self.ollama_url,
            'document_manager': self.doc_manager.get_status()
        }



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