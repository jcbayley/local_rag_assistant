import chromadb
import requests
import json
import argparse 
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from temp_docs import search_temp_documents

# Initialize ChromaDB client
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    try:
        collection = client.get_collection("chunked_scraped_data")
    except:
        # If collection doesn't exist, create it with the embedding function
        collection = client.create_collection("chunked_scraped_data", embedding_function=embedding_fn)
        
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    client = None
    collection = None

def retrieve_relevant_docs(query, top_k=3, use_chromadb=True):
    try:
        all_documents = []
        all_metadata = []
        
        # Search temporary documents first
        temp_docs, temp_meta = search_temp_documents(query, top_k//2 + 1 if use_chromadb else top_k)
        if temp_docs:
            all_documents.extend(temp_docs)
            all_metadata.extend(temp_meta)
            print(f"Found {len(temp_docs)} results from temporary documents")
        
        # Search ChromaDB only if enabled and available
        if use_chromadb:
            remaining_k = max(1, top_k - len(all_documents))
            if collection is not None:
                results = collection.query(query_texts=[query], n_results=remaining_k)
                if results['documents'] and results['documents'][0]:
                    all_documents.extend(results['documents'][0])
                    all_metadata.extend(results['metadatas'][0])
                    print(f"Found {len(results['documents'][0])} results from ChromaDB")
            elif not all_documents:
                print("Warning: ChromaDB not available and no temporary documents")
                return "No documents available", []
        elif not all_documents:
            print("Warning: ChromaDB disabled and no temporary documents")
            return "No documents available", []
        
        if not all_documents:
            return "No documents found", []
        
        # Limit to top_k results
        all_documents = all_documents[:top_k]
        all_metadata = all_metadata[:top_k]
        
        # Format context
        context = "\n\n".join(f"Document {i+1}\nSource: {all_metadata[i].get('filename', all_metadata[i].get('url', 'Unknown'))}\nContent: {all_documents[i]}" 
                             for i in range(len(all_documents)))
        
        return context, all_metadata
        
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return "Error retrieving documents", []

def get_model_response(prompt, model="qwen2.5vl:3b", temperature=0.1, max_tokens=500, repeat_penalty=1.3, top_p=0.9):
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": model,
            "prompt": prompt,
            "options":{
                "temperature": temperature,
                "max_tokens": max_tokens,
                "repeat_penalty": repeat_penalty,
                #"top_p": top_p,
            }
        }, stream=True, timeout=120)
        
        response.raise_for_status()
        
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


def rag_system(query, top_k=10, verbose=False, model_kwargs={"model": "qwen2.5vl:3b", "temperature": 0.1, "max_tokens": 1000, "repeat_penalty": 1.2, "top_p": 0.9}, use_chromadb=True):
    
    if verbose:
        print(f"Query: {query}")
    
    # Retrieve relevant documents using the original query directly
    context, metadata = retrieve_relevant_docs(query, top_k=top_k, use_chromadb=use_chromadb)
    
    if verbose:
        print(f"Retrieved {len(metadata)} documents")
    
    # Generate response with single LLM call
    prompt = f"""You are a helpful assistant searching a web archive. Use the provided context to answer the query. 
You must cite the source URL from each document directly after the fact it supports! 
Do not guess or make up information not found in the context.

Context: {context}

Query: {query}

Provide a bulleted answer with links to sources and format in markdown. Answer with references to URLs:"""

    response = get_model_response(prompt, **model_kwargs)
    
    return response, metadata[:5]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG system with a query.")
    parser.add_argument("--query","-q", type=str, help="The query to ask the RAG system.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    # Run the RAG system with the provided query
    response = rag_system(args.query, top_k=10, verbose=args.verbose)
    print(f"Query: {args.query}\nResponse: {response}")
