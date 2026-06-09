"""
Python FAQ Chatbot — Flask web app
Run from the project root (after activating .venv):
  source .venv/bin/activate
  python 'Task 2/chatbot.py'
Then open http://localhost:5001 in your browser
"""

import re

import nltk

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)

import numpy as np
from flask import Flask, jsonify, render_template_string, request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ_DATA: list[tuple[str, str]] = [
    ("What is Python?", "Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple paradigms including procedural, object-oriented, and functional programming."),
    ("How do I install Python?", "Visit python.org and download the latest installer for your OS. On Windows, check 'Add Python to PATH' during installation."),
    ("What is a variable in Python?", "A variable is a named container that stores data. Create one by assigning a value: x = 10 or name = 'Alice'. Python infers the type automatically."),
    ("What are Python data types?", "Built-in types: int, float, str, bool, list, tuple, dict, set, and NoneType. Use type(x) to inspect the type of any object."),
    ("What is a list in Python?", "A list is an ordered, mutable collection: [1, 2, 3]. It supports indexing, slicing, append, pop, sort, and many more methods."),
    ("What is a dictionary?", "A dict is an unordered collection of key-value pairs: {'name': 'Alice', 'age': 25}. Keys must be unique and immutable."),
    ("How do I define a function?", "Use the def keyword: def greet(name): return f'Hello, {name}'. Call it with greet('Alice')."),
    ("What is a class?", "A class is a blueprint for objects. Define it with class MyClass: and create instances with obj = MyClass(). Classes bundle data (attributes) and behavior (methods)."),
    ("What is inheritance?", "Inheritance lets a child class reuse code from a parent: class Dog(Animal): ... Dog gets all Animal methods and can override or extend them."),
    ("What is a lambda function?", "A lambda is a small anonymous function: square = lambda x: x**2. Good for short, one-time-use operations."),
    ("What is list comprehension?", "A concise way to build lists: [x*2 for x in range(5)] gives [0,2,4,6,8]. You can add a filter: [x for x in range(10) if x % 2 == 0]."),
    ("What is pip?", "pip is Python's package manager. Install packages with: pip install package_name. It fetches from PyPI (Python Package Index)."),
    ("What is a virtual environment?", "An isolated Python environment for project-specific dependencies. Create: python -m venv env. Activate: source env/bin/activate (Mac/Linux) or env\\Scripts\\activate (Windows)."),
    ("List vs tuple — what's the difference?", "Lists are mutable (can be changed) and use []. Tuples are immutable and use (). Tuples are slightly faster and used for fixed data."),
    ("How does exception handling work?", "Use try/except: try: risky() except ValueError as e: handle(e). Add else (runs if no exception) and finally (always runs)."),
    ("What is a module?", "A module is a .py file you can import. import math gives access to math.sqrt(), math.pi, etc. Use from math import sqrt to import specific items."),
    ("What is the difference between == and is?", "== checks value equality. is checks identity (same object in memory). [1,2] == [1,2] is True, but [1,2] is [1,2] is False."),
    ("What are *args and **kwargs?", "*args collects extra positional arguments as a tuple. **kwargs collects extra keyword arguments as a dict. def f(*args, **kwargs): lets you accept any input."),
    ("What is a decorator?", "A decorator wraps a function to add behavior: @my_decorator above def my_func(). Commonly used for logging, authentication, and caching."),
    ("What is a generator?", "A function that yields values one at a time using yield. Memory-efficient for large sequences — it doesn't build the whole list at once."),
    ("How do I read a file?", "Use: with open('file.txt', 'r') as f: content = f.read(). The with statement auto-closes the file."),
    ("How do I write to a file?", "with open('file.txt', 'w') as f: f.write('Hello'). Use 'a' mode to append, 'w' to overwrite."),
    ("What is JSON in Python?", "JSON is a text data format. json.dumps(obj) converts Python to JSON string. json.loads(string) parses JSON back to Python objects."),
    ("What is an API?", "An API is an interface that lets programs communicate. Web APIs accept HTTP requests and return data (usually JSON). Use the requests library to call them."),
    ("What is machine learning?", "ML is a branch of AI where algorithms learn patterns from data to make predictions. Common Python libraries: scikit-learn, TensorFlow, PyTorch."),
    ("Supervised vs unsupervised learning?", "Supervised: learns from labeled input-output pairs to predict outputs. Unsupervised: finds patterns in unlabeled data, e.g., clustering similar items."),
    ("What is overfitting?", "Overfitting happens when a model memorizes training data including noise and performs poorly on new data. Fix with regularization, dropout, or more training data."),
    ("What is a neural network?", "A neural network is a model of stacked layers of neurons. Each layer learns progressively more abstract features. Deep learning uses many hidden layers."),
    ("AI vs ML vs Deep Learning?", "AI is the broad goal of intelligent machines. ML is a subset using data-driven learning. Deep Learning is a subset of ML using multi-layer neural networks."),
    ("How do I exit Python?", "Type exit() or quit() in the REPL, or press Ctrl+D on Mac/Linux or Ctrl+Z+Enter on Windows."),
]

_lemmatizer = WordNetLemmatizer()
_stop_words: set[str] = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [_lemmatizer.lemmatize(t) for t in tokens if t not in _stop_words and len(t) > 1]
    return " ".join(tokens)


_questions = [q for q, _ in FAQ_DATA]
_answers = [a for _, a in FAQ_DATA]
_processed_questions = [preprocess(q) for q in _questions]
_vectorizer = TfidfVectorizer()
_tfidf_matrix = _vectorizer.fit_transform(_processed_questions)

GREETINGS: set[str] = {"hi", "hello", "hey", "howdy", "greetings", "sup", "yo"}
FAREWELLS: set[str] = {"bye", "goodbye", "exit", "quit", "see you", "later", "farewell"}


def get_answer(user_input: str, threshold: float = 0.15) -> str:
    lower = user_input.lower().strip()
    if any(g in lower for g in GREETINGS):
        return "Hello! I'm your Python FAQ Chatbot. Ask me anything about Python or machine learning!"
    if any(f in lower for f in FAREWELLS):
        return "Goodbye! Happy coding!"
    if "thank" in lower:
        return "You're welcome! Feel free to ask more questions."

    processed = preprocess(user_input)
    if not processed.strip():
        return "I didn't understand that. Could you rephrase?"

    user_vec = _vectorizer.transform([processed])
    sims = cosine_similarity(user_vec, _tfidf_matrix).flatten()
    best_idx: int = int(np.argmax(sims))
    best_score: float = float(sims[best_idx])

    if best_score < threshold:
        return (
            "I'm not sure I have an answer for that. "
            "Try asking about Python basics, data types, functions, classes, file I/O, or machine learning."
        )
    return _answers[best_idx]


app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Python FAQ Chatbot</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #1e1e2e;
    color: #cdd6f4;
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    padding: 28px 16px;
  }
  h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 20px; }
  .chat-container {
    background: #181825;
    border-radius: 16px;
    width: 100%;
    max-width: 700px;
    display: flex;
    flex-direction: column;
    height: 70vh;
    box-shadow: 0 8px 32px #00000060;
    overflow: hidden;
  }
  #chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .msg { display: flex; flex-direction: column; max-width: 80%; }
  .msg.user { align-self: flex-end; align-items: flex-end; }
  .msg.bot  { align-self: flex-start; align-items: flex-start; }
  .msg-label { font-size: 0.75rem; font-weight: 600; margin-bottom: 4px; color: #6c7086; }
  .msg.user .msg-label { color: #89b4fa; }
  .msg.bot  .msg-label { color: #a6e3a1; }
  .bubble {
    padding: 10px 15px;
    border-radius: 12px;
    font-size: 0.95rem;
    line-height: 1.5;
    max-width: 100%;
    word-break: break-word;
  }
  .msg.user .bubble { background: #313244; color: #cdd6f4; border-bottom-right-radius: 3px; }
  .msg.bot  .bubble { background: #1e3a2e; color: #a6e3a1; border-bottom-left-radius: 3px; }
  .input-row {
    display: flex;
    gap: 10px;
    padding: 14px 16px;
    background: #11111b;
    border-top: 1px solid #313244;
  }
  input[type=text] {
    flex: 1;
    background: #313244;
    border: none;
    border-radius: 8px;
    padding: 11px 14px;
    color: #cdd6f4;
    font-size: 1rem;
    outline: none;
    font-family: inherit;
  }
  input[type=text]:focus { box-shadow: 0 0 0 2px #89b4fa; }
  button {
    background: #89b4fa;
    color: #1e1e2e;
    border: none;
    border-radius: 8px;
    padding: 11px 22px;
    font-size: 0.95rem;
    font-weight: 700;
    cursor: pointer;
    transition: opacity .15s;
  }
  button:hover { opacity: 0.85; }
  .typing { color: #6c7086; font-style: italic; font-size: 0.88rem; padding: 4px 0; }
  #chat-area::-webkit-scrollbar { width: 6px; }
  #chat-area::-webkit-scrollbar-track { background: transparent; }
  #chat-area::-webkit-scrollbar-thumb { background: #45475a; border-radius: 3px; }
</style>
</head>
<body>
<h1>Python FAQ Chatbot</h1>
<div class="chat-container">
  <div id="chat-area"></div>
  <div class="input-row">
    <input type="text" id="user-input" placeholder="Ask a Python question..." autocomplete="off">
    <button onclick="sendMessage()">Send</button>
  </div>
</div>

<script>
const chatArea = document.getElementById('chat-area');
const inputEl  = document.getElementById('user-input');

function addMessage(text, role) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  const label = document.createElement('div');
  label.className = 'msg-label';
  label.textContent = role === 'user' ? 'You' : 'Bot';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  wrap.appendChild(label);
  wrap.appendChild(bubble);
  chatArea.appendChild(wrap);
  chatArea.scrollTop = chatArea.scrollHeight;
}

function showTyping() {
  const el = document.createElement('div');
  el.className = 'typing';
  el.id = 'typing';
  el.textContent = 'Bot is typing...';
  chatArea.appendChild(el);
  chatArea.scrollTop = chatArea.scrollHeight;
}

function hideTyping() {
  const el = document.getElementById('typing');
  if (el) el.remove();
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';
  addMessage(text, 'user');
  showTyping();
  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: text})
    });
    const data = await res.json();
    hideTyping();
    addMessage(data.answer, 'bot');
  } catch (e) {
    hideTyping();
    addMessage('Sorry, something went wrong.', 'bot');
  }
}

inputEl.addEventListener('keydown', e => { if (e.key === 'Enter') sendMessage(); });

// Initial bot greeting
addMessage('Hello! I\\'m your Python FAQ Chatbot. Ask me anything about Python or machine learning!', 'bot');
</script>
</body>
</html>"""


@app.route("/")
def index() -> str:
    return render_template_string(HTML)


@app.route("/ask", methods=["POST"])
def ask() -> tuple:
    data: dict = request.get_json(force=True)
    question: str = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please ask a question."}), 400
    answer: str = get_answer(question)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    import threading
    import webbrowser

    port = 5001
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    print(f"Chatbot running at http://localhost:{port}  (Ctrl+C to stop)")
    app.run(port=port, debug=False)
