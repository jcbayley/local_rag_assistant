import chromadb
import json
import argparse
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import textwrap
import uuid

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create or load a ChromaDB collection.")
    parser.add_argument("--input", "-i", type=str, default="scraped_data.json",help="Path to the input JSON file containing scraped data.")
    args = parser.parse_args()
    # Create the collection from the scraped data JSON file
    create_collection(args.input)


