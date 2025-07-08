import chromadb
import requests
import json
import argparse 

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("glasgow_scraped_data")

def retrieve_relevant_docs(query, top_k=3):
    # Retrieve relevant documents from ChromaDB
    results = collection.query(query_texts=[query], n_results=top_k)
    return results['documents'][0], results['metadatas'][0]

def get_keywords(query, model = "qwen2.5vl:3b"):
    # Prepare the prompt for the LLM
    prompt = f"Context: Can you turn this query into a short set of keywords for a RAG vector database, no more than 15 words. \n Query: {query}\n"
    # Send the prompt to the Ollama LLM
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt
    }, stream=True)

    full_response = ""
    for line in response.iter_lines():
        if line:
            data = line.decode("utf-8")

            json_obj = json.loads(data)
            if "response" in json_obj:
                full_response += json_obj["response"]

    return full_response

def check_result(query, response, context, metadata, model = "qwen2.5vl:3b"):
    # Prepare the prompt for the LLM
    prompt = f"Context:{context}\n metadata: {metadata}\n Query: {query} \n Response: {response} \n Can you improve the response based on the query, i.e. fix bad links or include missing ones. Return only the better answer. \n Answer: \n"
    # Send the prompt to the Ollama LLM
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt
    }, stream=True)

    full_response = ""
    for line in response.iter_lines():
        if line:
            data = line.decode("utf-8")

            json_obj = json.loads(data)
            if "response" in json_obj:
                full_response += json_obj["response"]

    return full_response

def generate_response(query, context, metadata, model = "qwen2.5vl:3b"):
    # Prepare the prompt for the LLM
    prompt = f"Context: {context}\n Metadata: {metadata} \n Query: {query} \n Please provide relevant links. \n Answer:"
    # Send the prompt to the Ollama LLM
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt
    }, stream=True)

    full_response = ""
    for line in response.iter_lines():
        if line:
            data = line.decode("utf-8")

            json_obj = json.loads(data)
            if "response" in json_obj:
                full_response += json_obj["response"]

    return full_response

def generate_better_query(query, context, metadata, model = "qwen2.5vl:3b"):
    # Prepare the prompt for the LLM
    prompt = f"Context: {context}\n Metadata: {metadata} \n Query: {query} \n Can you generate a better query for a language model given what you have accessed from the metadata and context? Try to include a larger number of relevent key words and increase the length of the query.\n Better Query: \n"
    # Send the prompt to the Ollama LLM
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt
    }, stream=True)

    full_response = ""
    for line in response.iter_lines():
        if line:
            data = line.decode("utf-8")

            json_obj = json.loads(data)
            if "response" in json_obj:
                full_response += json_obj["response"]

    return full_response

def rag_system(query, top_k=10, verbose=False, model = "qwen2.5vl:3b"):

    keywords = get_keywords(query, model=model)
    if verbose:
        print(f"Query: {query}")
        print(f"Keywords: {keywords}")
    # Retrieve relevant documents
    context, metadata = retrieve_relevant_docs(keywords, top_k=top_k)
    if verbose:
        print(metadata)
    
    query = generate_better_query(query, context, metadata, model=model)
    if verbose:
        print(f"Better Query: {query}")
    keywords = query#get_keywords(query, model=model)
    if verbose:
        print(f"Keywords: {keywords}")
    # Retrieve relevant documents
    context, metadata = retrieve_relevant_docs(keywords, top_k=top_k)
    
    # Generate a response using the LLM
    response = generate_response(query, context, metadata, model=model)

    #response = check_result(query, response, context, metadata)

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG system with a query.")
    parser.add_argument("--query","-q", type=str, help="The query to ask the RAG system.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    # Run the RAG system with the provided query
    response = rag_system(args.query, top_k=10, verbose=args.verbose)
    print(f"Query: {args.query}\nResponse: {response}")
