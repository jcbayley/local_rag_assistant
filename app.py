from flask import Flask, request, jsonify, render_template_string, render_template
from llm_query_functions import rag_system
from temp_docs import add_temp_document, get_temp_documents_info, clear_temp_documents
import markdown
import subprocess
import json
import os
from werkzeug.utils import secure_filename
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def get_ollama_models():
    try:
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True, 
            timeout=10,
            check=False
        )
        
        if result.returncode != 0:
            print(f"Warning: ollama command failed with return code {result.returncode}")
            print(f"stderr: {result.stderr}")
            return ["qwen2.5vl:3b"]  # fallback model
        
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            print("Warning: No models found in ollama output")
            return ["qwen2.5vl:3b"]  # fallback model

        # Skip the header line and parse model names
        models = [line.split()[0] for line in lines[1:] if line.strip()]
        
        if not models:
            print("Warning: No valid models parsed from ollama output")
            return ["qwen2.5vl:3b"]  # fallback model
            
        return models
        
    except Exception as e:
        print(f"Error fetching ollama models: {e}")
        return ["qwen2.5vl:3b"]  # fallback model

@app.route('/', methods=['GET'])
def index():
    models = get_ollama_models()
    return render_template("./index.html", models=models)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        query = data.get('query')
        if not query:
            return jsonify({"error": "Missing 'query'"}), 400
            
        top_k = int(data.get('top_k', 10))
        use_chromadb = data.get('use_chromadb', True)
        kwargs = {
            "model": data.get('model_name', "qwen2.5vl:3b"),
            "temperature": float(data.get('temperature', 0.1)),
            "max_tokens": int(data.get('max_tokens', 100)),
            "repeat_penalty": float(data.get('repeat_penalty', 1.4)),
            "top_p": float(data.get('top_p', 0.9))
        }
        
        response, top_urls = rag_system(query, top_k=top_k, verbose=True, model_kwargs=kwargs, use_chromadb=use_chromadb)
        
        # Handle different metadata formats (ChromaDB vs temp docs)
        source_links = "\n\n**Sources:**\n"
        for url_data in top_urls:
            if isinstance(url_data, dict):
                if 'url' in url_data:
                    # ChromaDB format
                    source_links += f"- [{url_data['url']}]({url_data['url']})\n"
                elif 'filename' in url_data:
                    # Temporary document format
                    source_links += f"- {url_data['filename']} ({url_data.get('source_type', 'Document')})\n"
        
        full_response = markdown.markdown(response + source_links) 

        return jsonify({"response": full_response})
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": "Failed to process request"}), 500

@app.route('/api/upload', methods=['POST'])
def upload_document():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check file type
        allowed_extensions = {'.pdf', '.txt', '.md'}
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"}), 400
        
        # Save file temporarily
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Add to temporary document storage
        success, message = add_temp_document(file_path)
        
        # Clean up temporary file
        try:
            os.remove(file_path)
        except:
            pass
        
        if success:
            return jsonify({"message": message})
        else:
            return jsonify({"error": message}), 500
            
    except Exception as e:
        print(f"Error in upload endpoint: {e}")
        return jsonify({"error": "Failed to upload document"}), 500

@app.route('/api/temp-docs', methods=['GET'])
def get_temp_docs():
    """Get list of temporary documents"""
    try:
        docs = get_temp_documents_info()
        return jsonify({"documents": docs})
    except Exception as e:
        print(f"Error getting temp docs: {e}")
        return jsonify({"error": "Failed to get documents"}), 500

@app.route('/api/temp-docs/clear', methods=['POST'])
def clear_temp_docs():
    """Clear all temporary documents"""
    try:
        clear_temp_documents()
        return jsonify({"message": "All temporary documents cleared"})
    except Exception as e:
        print(f"Error clearing temp docs: {e}")
        return jsonify({"error": "Failed to clear documents"}), 500

if __name__ == '__main__':
    app.run(debug=True)