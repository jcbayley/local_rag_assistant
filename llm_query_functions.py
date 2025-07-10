import chromadb
import requests
import json
import argparse 

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("chunked_scraped_data")

def retrieve_relevant_docs(query, top_k=3):
    # Retrieve relevant documents from ChromaDB
    results = collection.query(query_texts=[query], n_results=top_k)
    context = "\n\n".join(f"Document {i} \n URL: {results['metadatas'][0][i]}) \n Content: {results['documents'][0][i]}" for i in range(top_k))
    return context, results['metadatas'][0]

def get_model_response(prompt, model="qwen2.5vl:3b", temperature=0.1, max_tokens=500, repeat_penalty=1.3, top_p=0.9):
    # Send the prompt to the Ollama LLM
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "options":{
            "temperature": temperature,
            "max_tokens": max_tokens,
            "repeat_penalty": repeat_penalty,
            #"top_p": top_p,
        }
    }, stream=True)

    full_response = ""
    for line in response.iter_lines():
        if line:
            data = line.decode("utf-8")
            json_obj = json.loads(data)
            if "response" in json_obj:
                full_response += json_obj["response"]

    return full_response


def rag_system(query,  top_k=10, verbose=False, model_kwargs={"model": "qwen2.5vl:3b", "temperature": 0.1, "max_tokens": 1000, "repeat_penalty": 1.2, "top_p": 0.9}):

    keyword_prompt = f"""Context: Extract the most relevant keywords from this question 
                        to help retrieve documents from a database. Just return a space separated list of a maximum of 15 keywords. \n Query: {query}\n"""
    keywords = get_model_response(keyword_prompt, **model_kwargs)
    if verbose:
        print(f"Query: {query}")
        print(f"Keywords: {keywords}")

    # Retrieve relevant documents
    all_words = keywords #+ query
    context, metadata = retrieve_relevant_docs(all_words, top_k=top_k)
    #if verbose:
    #    print(context)
    
    query_prompt = f"""Context: {context}\n Metadata: {metadata} \n Query: {query} \n Can you generate a better 
                        query for a language model given what you have accessed from the metadata and context? 
                        Try to include a larger number of relevent key words and increase the length of the query.\n
                        Better Query: \n"""
    #query = get_model_response(query_prompt, **model_kwargs)
    #context, metadata = retrieve_relevant_docs(keywords + query, top_k=top_k)

    context_prompt = f"""Your job is to find the paragraphs of text (with document URLs) most relevant to the query 
                    and repeat them below for input to another method. 
                    Repeat the document headers and paragraphs of relevant text, dont miss details or summarise sentances. 
                    Context: {context}\n Query: {query} \n Repeated context:\n"""

    #smaller_context = get_model_response(context_prompt, **model_kwargs)

    think_prompt = f"""Think about this query Query: {query} and describe what you need to do to find the answer given from a set 
                a set of documents which will be provided with the urls: {metadata}. \n """

    think_query = get_model_response(think_prompt, **model_kwargs)
    print(think_query)
    # Generate a response using the LLM
    #prompt = f"""You are a helpful assistant searching a web archive. Use the provided context to answer the 
    #            query. You must cite the source URL from each document directly after the fact it supports! 
    #            Do not guess. \n 
    #            Context: {context} \n \n Query: {query} \n Answer:"""
    prompt = f"""This is what you must do give the context: {think_query}. Context: {context} \n \n Original query: {query} \n 
                Provide a bulleted answer with links to sources and format in markdown. Answer with references to URLs:"""
    #prompt = f"""Context: {context} \n \n Query: {query} \n You must include links to sources. Answer:"""

    response = get_model_response(prompt, **model_kwargs)

    review_prompt = f"""You are a reviewer, your job is to check the response from the query and fix any issues 
                        like wrong or missing links and search context again for missing information. 
                        Context: {context}\n Metadata: {metadata}\n Query: {query} \n Response: {response} \n 
                        Answer: \n"""
    #response = get_model_response(review_prompt, **model_kwargs)

    review_prompt = f"""You are a reviewer, your job is to check the response from the query and fix any issues 
                        like wrong or missing links and search context again for missing information. Reformat in markdown, and make sure source links are cited! 
                        Context: {context}\n Metadata: {metadata}\n Query: {query} \n Response: {response} \n 
                        Only provide the corrected answer no other comments: \n"""

    #response = get_model_response(review_prompt, **model_kwargs)


    return response, metadata[:5]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG system with a query.")
    parser.add_argument("--query","-q", type=str, help="The query to ask the RAG system.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    # Run the RAG system with the provided query
    response = rag_system(args.query, top_k=10, verbose=args.verbose)
    print(f"Query: {args.query}\nResponse: {response}")
