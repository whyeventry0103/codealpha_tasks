"""
Language Translation Tool — Flask web app
Run from the project root (after activating .venv):
  source .venv/bin/activate
  python 'Task 1/translator.py'
Then open http://localhost:5000 in your browser
"""

import io
import tempfile

from flask import Flask, jsonify, render_template_string, request, send_file
from deep_translator import GoogleTranslator
from gtts import gTTS

app = Flask(__name__)

LANGUAGES: dict[str, str] = {
    "Auto Detect": "auto",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese (Simplified)": "zh-CN",
    "Arabic": "ar",
    "Hindi": "hi",
    "Urdu": "ur",
    "Turkish": "tr",
    "Dutch": "nl",
    "Polish": "pl",
    "Swedish": "sv",
    "Greek": "el",
    "Hebrew": "he",
}

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Language Translator</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #1e1e2e;
    color: #cdd6f4;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 32px 16px;
  }
  h1 { font-size: 2rem; font-weight: 700; margin-bottom: 28px; color: #cdd6f4; letter-spacing: -0.5px; }
  .card {
    background: #181825;
    border-radius: 16px;
    padding: 28px;
    width: 100%;
    max-width: 900px;
    box-shadow: 0 8px 32px #00000060;
  }
  .lang-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
    flex-wrap: wrap;
  }
  label { font-size: 0.9rem; color: #a6adc8; }
  select {
    background: #313244;
    color: #cdd6f4;
    border: none;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.95rem;
    cursor: pointer;
    outline: none;
  }
  select:focus { box-shadow: 0 0 0 2px #89b4fa; }
  .swap-btn {
    background: #313244;
    color: #cdd6f4;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    cursor: pointer;
    font-size: 1rem;
    transition: background .15s;
  }
  .swap-btn:hover { background: #45475a; }
  .text-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 20px;
  }
  .text-col { display: flex; flex-direction: column; gap: 6px; }
  .text-col span { font-size: 0.85rem; font-weight: 600; color: #89b4fa; }
  .text-col.target span { color: #a6e3a1; }
  textarea {
    background: #313244;
    color: #cdd6f4;
    border: none;
    border-radius: 10px;
    padding: 14px;
    font-size: 1rem;
    resize: vertical;
    min-height: 220px;
    outline: none;
    font-family: inherit;
  }
  textarea:focus { box-shadow: 0 0 0 2px #89b4fa; }
  #output {
    background: #313244;
    color: #a6e3a1;
    border-radius: 10px;
    padding: 14px;
    min-height: 220px;
    font-size: 1rem;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .btn-row { display: flex; gap: 12px; flex-wrap: wrap; }
  .btn {
    border: none;
    border-radius: 8px;
    padding: 10px 22px;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: opacity .15s, transform .1s;
  }
  .btn:hover { opacity: 0.88; transform: translateY(-1px); }
  .btn:active { transform: translateY(0); }
  .btn-primary { background: #89b4fa; color: #1e1e2e; }
  .btn-secondary { background: #313244; color: #cdd6f4; }
  .btn-danger { background: #f38ba8; color: #1e1e2e; }
  #status {
    margin-top: 14px;
    font-size: 0.88rem;
    color: #fab387;
    min-height: 20px;
  }
  #char-count { font-size: 0.78rem; color: #6c7086; text-align: right; }
  @media (max-width: 600px) {
    .text-row { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<h1>Language Translation Tool</h1>
<div class="card">
  <div class="lang-row">
    <label for="src">From:</label>
    <select id="src">
      {% for name, code in languages.items() %}
      <option value="{{ code }}" {% if name == 'Auto Detect' %}selected{% endif %}>{{ name }}</option>
      {% endfor %}
    </select>
    <button class="swap-btn" onclick="swapLangs()" title="Swap languages">&#8644;</button>
    <label for="tgt">To:</label>
    <select id="tgt">
      {% for name, code in languages.items() %}
      {% if name != 'Auto Detect' %}
      <option value="{{ code }}" {% if name == 'Spanish' %}selected{% endif %}>{{ name }}</option>
      {% endif %}
      {% endfor %}
    </select>
  </div>

  <div class="text-row">
    <div class="text-col">
      <span>Source Text</span>
      <textarea id="input" placeholder="Enter text to translate..." oninput="updateCount()"></textarea>
      <div id="char-count">0 characters</div>
    </div>
    <div class="text-col target">
      <span>Translated Text</span>
      <div id="output"></div>
    </div>
  </div>

  <div class="btn-row">
    <button class="btn btn-primary" onclick="doTranslate()">Translate</button>
    <button class="btn btn-secondary" onclick="copyResult()">Copy Result</button>
    <button class="btn btn-secondary" onclick="speakResult()">Text-to-Speech</button>
    <button class="btn btn-danger" onclick="clearAll()">Clear</button>
  </div>
  <div id="status"></div>
</div>

<script>
function setStatus(msg) { document.getElementById('status').textContent = msg; }
function updateCount() {
  const n = document.getElementById('input').value.length;
  document.getElementById('char-count').textContent = n + ' character' + (n === 1 ? '' : 's');
}

async function doTranslate() {
  const text = document.getElementById('input').value.trim();
  if (!text) { setStatus('Please enter text to translate.'); return; }
  const src = document.getElementById('src').value;
  const tgt = document.getElementById('tgt').value;
  setStatus('Translating...');
  try {
    const res = await fetch('/translate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, src, tgt})
    });
    const data = await res.json();
    if (data.error) { setStatus('Error: ' + data.error); return; }
    document.getElementById('output').textContent = data.result;
    setStatus('Translation complete.');
  } catch (e) { setStatus('Network error: ' + e); }
}

function swapLangs() {
  const src = document.getElementById('src');
  const tgt = document.getElementById('tgt');
  if (src.value === 'auto') return;
  const tmp = src.value;
  // find matching option in src for tgt value
  for (const opt of src.options) { if (opt.value === tgt.value) { src.value = tgt.value; break; } }
  tgt.value = tmp;
  // swap text content
  const inputEl = document.getElementById('input');
  const outputEl = document.getElementById('output');
  const tmp2 = inputEl.value;
  inputEl.value = outputEl.textContent;
  outputEl.textContent = tmp2;
  updateCount();
}

function copyResult() {
  const text = document.getElementById('output').textContent;
  if (!text) { setStatus('Nothing to copy.'); return; }
  navigator.clipboard.writeText(text).then(() => setStatus('Copied to clipboard.'));
}

async function speakResult() {
  const text = document.getElementById('output').textContent.trim();
  const lang = document.getElementById('tgt').value;
  if (!text) { setStatus('No translated text to speak.'); return; }
  setStatus('Generating audio...');
  try {
    const res = await fetch('/speak', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, lang})
    });
    if (!res.ok) { setStatus('TTS error: ' + (await res.json()).error); return; }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.onended = () => { setStatus('Done speaking.'); URL.revokeObjectURL(url); };
    audio.onerror = () => setStatus('Audio playback error.');
    audio.play();
    setStatus('Speaking...');
  } catch (e) { setStatus('TTS error: ' + e); }
}

function clearAll() {
  document.getElementById('input').value = '';
  document.getElementById('output').textContent = '';
  setStatus('');
  updateCount();
}

document.getElementById('input').addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') doTranslate();
});
</script>
</body>
</html>"""


@app.route("/")
def index() -> str:
    return render_template_string(HTML, languages=LANGUAGES)


@app.route("/translate", methods=["POST"])
def translate() -> tuple:
    data: dict = request.get_json(force=True)
    text: str = data.get("text", "").strip()
    src: str = data.get("src", "auto")
    tgt: str = data.get("tgt", "en")

    if not text:
        return jsonify({"error": "Empty text"}), 400
    if tgt == "auto":
        return jsonify({"error": "Please select a valid target language"}), 400

    try:
        result: str = GoogleTranslator(source=src, target=tgt).translate(text)
        return jsonify({"result": result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/speak", methods=["POST"])
def speak() -> tuple:
    data: dict = request.get_json(force=True)
    text: str = data.get("text", "").strip()
    lang: str = data.get("lang", "en")
    if lang == "auto":
        lang = "en"
    if not text:
        return jsonify({"error": "Empty text"}), 400
    try:
        tts = gTTS(text=text, lang=lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return send_file(buf, mimetype="audio/mpeg")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    import webbrowser
    import threading

    port = 5000
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    print(f"Translator running at http://localhost:{port}  (Ctrl+C to stop)")
    app.run(port=port, debug=False)
