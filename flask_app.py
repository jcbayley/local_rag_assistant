from flask import Flask, request, jsonify, render_template_string
from llm_query_functions import rag_system
import markdown
app = Flask(__name__)

HTML_TEMPLATE = '''
<!doctype html>
<html>
<head>
  <title>Flask Chat</title>
  <style>
    body {
      font-family: sans-serif;
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      height: 100vh;
      background: #f8f9fa;
    }
    #chat {
        flex: 1;
        overflow-y: auto;
        display: flex;
        flex-direction: column-reverse; /* Bottom-up */
        padding: 1rem;
        }
    .message {
      margin-bottom: 1rem;
    }
    .user {
      font-weight: bold;
      color: #007bff;
    }
    .bot {
      font-weight: bold;
      color: #28a745;
    }
    #input-form {
      display: flex;
      padding: 0.75rem;
      background: #fff;
      border-top: 1px solid #ccc;
    }
    #query {
      flex: 1;
      padding: 0.5rem;
      font-size: 1rem;
    }
    button {
      padding: 0.5rem 1rem;
      margin-left: 0.5rem;
      font-size: 1rem;
    }
    #loading {
      font-style: italic;
      color: gray;
    }
    .md {
        background: #f1f1f1;
        padding: 0.75rem;
        border-radius: 6px;
        margin-top: 0.25rem;
        font-size: 0.95rem;
    }

    .md p {
    margin: 0.5rem 0;
    }

    .md code {
    background: #eee;
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    font-family: monospace;
    }

    .md pre {
    background: #eee;
    padding: 1rem;
    overflow-x: auto;
    border-radius: 4px;
    }
  </style>
</head>
<!-- Settings Panel -->
<div id="settings" style="position: fixed; top: 1rem; right: 1rem; background: #fff; padding: 1rem; border: 1px solid #ccc; border-radius: 8px;">
  <h4>Settings</h4>
  <label>Model:
    <select id="model_name">
      <option value="default-model">default-model</option>
    </select>
  </label>
  <br><br>
  <label>Top K:
    <input type="number" id="top_k" value="5" min="1" max="100">
  </label>
</div>
<body>
  <div id="chat"></div>
  <div id="input-form">
    <input type="text" id="query" placeholder="Type your message..." autofocus>
    <button onclick="sendMessage()">Send</button>
  </div>

  <script>
    const chat = document.getElementById("chat");
    const input = document.getElementById("query");

    function addMessage(sender, text) {
        const div = document.createElement("div");
        div.className = "message";

        if (sender === "bot") {
            div.innerHTML = `<span class="${sender}">${sender}:</span><br><div class="md">${text}</div>`;
        } else {
            div.innerHTML = `<span class="${sender}">${sender}:</span> ${text}`;
        }

        chat.prepend(div);
        }

    function sendMessage() {
        const text = input.value.trim();
        if (!text) return;

        const modelName = document.getElementById("model_name").value;
        const topK = document.getElementById("top_k").value;

        addMessage("user", text);
        input.value = "";

        const loadingMsg = document.createElement("div");
        loadingMsg.id = "loading";
        loadingMsg.innerText = "Sending...";
        chat.prepend(loadingMsg);

        fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
            query: text,
            model_name: modelName,
            top_k: topK
            })
        })
        .then(res => res.json())
        .then(data => {
            document.getElementById("loading").remove();
            addMessage("bot", data.response);
        })
        .catch(err => {
            document.getElementById("loading").remove();
            addMessage("bot", "Error: " + err.message);
        });
        }

    // Press Enter to send
    input.addEventListener("keypress", function(e) {
      if (e.key === "Enter") {
        sendMessage();
        e.preventDefault();
      }
    });
  </script>
</body>
</html>
'''

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query')
    model_name = data.get('model_name', "qwen2.5vl:3b")
    if model_name == "default-model":
        model_name = "qwen2.5vl:3b"
    top_k = data.get('top_k', 10)
    if not query:
        return jsonify({"error": "Missing 'query'"}), 400
    print(f"Received query: {query}")
    print(f"Using model: {model_name}, Top K: {top_k}")
    response = rag_system(query, top_k=int(top_k), verbose=False, model=model_name)
    return jsonify({"response": markdown.markdown(response)})

if __name__ == '__main__':
    app.run(debug=True)