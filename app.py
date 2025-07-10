from flask import Flask, request, jsonify, render_template_string, render_template
from llm_query_functions import rag_system
import markdown
import subprocess
import json
app = Flask(__name__)


def get_ollama_models():
    try:
        result = subprocess.run(["ollama", "list",], capture_output=True, text=True)
        lines = result.stdout.strip().split("\n")

        # Skip the header line and parse model names
        models = [line.split()[0] for line in lines[1:] if line.strip()]
        return models
    except Exception as e:
        raise Exception(f"Error fetching ollama models: {e}") from e

@app.route('/', methods=['GET'])
def index():
    models = get_ollama_models()
    return render_template("./index.html", models=models)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query')
    top_k = int(data.get('top_k', 10))
    if not query:
        return jsonify({"error": "Missing 'query'"}), 400
    kwargs = {
        "model": data.get('model_name', "qwen2.5vl:3b"),
        "temperature": float(data.get('temperature', 0.1)),
        "max_tokens": int(data.get('max_tokens', 100)),
        "repeat_penalty": float(data.get('repeat_penalty', 1.4)),
        "top_p": float(data.get('top_p', 0.9))
    }
    response, top_urls = rag_system(query, top_k=top_k, verbose=True, model_kwargs=kwargs)

    source_links = "\n\n**Sources:**\n" + "\n".join(f"- [{url['url']}]({url['url']})" for url in top_urls if url)

    full_response = markdown.markdown(response + source_links) 

    return jsonify({"response": full_response})

if __name__ == '__main__':
    app.run(debug=True)