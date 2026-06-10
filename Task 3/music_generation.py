"""
AI Music Generation with LSTM — Flask Web UI
Run: source .venv/bin/activate && python 'Task 3/music_generation.py'
Opens at http://localhost:5002
"""

import base64
import io
import os
import random
import threading

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template_string, request, send_file
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Scale / style definitions
# ---------------------------------------------------------------------------

_SCALES: dict[str, list[list[int]]] = {
    "classical": [
        [60, 62, 64, 65, 67, 69, 71, 72],   # C major
        [69, 71, 72, 74, 76, 77, 79, 81],   # A minor
        [55, 57, 59, 60, 62, 64, 66, 67],   # G major
    ],
    "jazz": [
        [60, 63, 65, 66, 67, 70, 72],        # blues
        [60, 62, 63, 65, 67, 69, 70, 72],   # dorian
    ],
    "pentatonic": [
        [60, 62, 64, 67, 69, 72],            # major pentatonic
        [60, 63, 65, 67, 70, 72],            # minor pentatonic
    ],
    "ambient": [
        [60, 64, 67, 71, 72],               # Cmaj7 arpeggio
        [60, 62, 64, 67, 69, 72],           # pentatonic sparse
    ],
}

SEQ_LEN = 16


def _generate_melody(scales: list[list[int]], length: int = 64, rest_prob: float = 0.08) -> list[int]:
    scale = random.choice(scales)
    melody: list[int] = []
    for _ in range(length):
        if random.random() < rest_prob:
            melody.append(0)
        else:
            note = random.choice(scale)
            # Occasionally step to adjacent notes for melodic flow
            if melody and random.random() < 0.3:
                last = melody[-1]
                if last > 0:
                    delta = random.choice([-2, -1, 1, 2])
                    candidate = last + delta
                    if 48 <= candidate <= 84:
                        note = candidate
            melody.append(note)
    return melody


def build_corpus(style: str, num_melodies: int = 100, melody_length: int = 64) -> list[int]:
    scales = _SCALES.get(style, _SCALES["classical"])
    rest_prob = 0.18 if style == "ambient" else 0.08
    corpus: list[int] = []
    for _ in range(num_melodies):
        corpus.extend(_generate_melody(scales, melody_length, rest_prob))
    return corpus


def prepare_sequences(notes: list[int]) -> tuple:
    unique = sorted(set(notes))
    n2i: dict[int, int] = {n: i for i, n in enumerate(unique)}
    i2n: dict[int, int] = {i: n for n, i in n2i.items()}
    n_vocab = len(unique)
    encoded = [n2i[n] for n in notes]
    X_list, y_list = [], []
    for i in range(len(encoded) - SEQ_LEN):
        X_list.append(encoded[i: i + SEQ_LEN])
        y_list.append(encoded[i + SEQ_LEN])
    X = torch.tensor(X_list, dtype=torch.float32).unsqueeze(-1) / n_vocab
    y = torch.tensor(y_list, dtype=torch.long)
    return X, y, n2i, i2n, n_vocab


class MusicLSTM(nn.Module):
    def __init__(self, n_vocab: int, hidden: int = 128) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(1, hidden, batch_first=True)
        self.drop1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(hidden, 64, batch_first=True)
        self.drop2 = nn.Dropout(0.3)
        self.fc    = nn.Linear(64, n_vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])
        return self.fc(out)


def train_model(
    model: MusicLSTM, X: torch.Tensor, y: torch.Tensor,
    epochs: int = 25, batch_size: int = 64,
    progress_cb=None,
) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)
    best_loss, patience_count = float("inf"), 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        avg = total / len(loader)
        scheduler.step(avg)
        if progress_cb:
            progress_cb(epoch, epochs, avg)
        if avg < best_loss - 1e-4:
            best_loss, patience_count = avg, 0
        else:
            patience_count += 1
            if patience_count >= 7:
                break
    model.to("cpu")


def generate_notes(
    model: MusicLSTM, seed: list[int], i2n: dict,
    n_vocab: int, length: int = 128, temperature: float = 0.8,
) -> list[int]:
    model.eval()
    current = list(seed)
    generated = list(seed)
    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor(current, dtype=torch.float32).unsqueeze(0).unsqueeze(-1) / n_vocab
            logits = model(x)[0] / temperature
            probs = torch.softmax(logits, dim=0).numpy()
            chosen = int(np.random.choice(len(probs), p=probs))
            generated.append(chosen)
            current = current[1:] + [chosen]
    return [i2n[i] for i in generated]


def notes_to_midi_b64(pitches: list[int], durations: list[float], tempo: int = 120) -> str:
    import tempfile
    from music21 import note, stream
    from music21 import tempo as m21tempo
    from music21 import midi as m21midi
    s = stream.Stream()
    s.append(m21tempo.MetronomeMark(number=tempo))
    for pitch, dur in zip(pitches, durations):
        n = note.Rest(quarterLength=dur) if pitch == 0 else note.Note(pitch, quarterLength=dur)
        s.append(n)
    mf = m21midi.translate.streamToMidiFile(s)
    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    tmp.close()
    mf.open(tmp.name, "wb")
    mf.write()
    mf.close()
    with open(tmp.name, "rb") as f:
        data = f.read()
    os.unlink(tmp.name)
    return base64.b64encode(data).decode()


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
_lock = threading.Lock()

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Music Generator</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #1e1e2e; color: #cdd6f4; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 32px 16px; }
  h1 { font-size: 2rem; font-weight: 700; margin-bottom: 6px; }
  .subtitle { font-size: 0.88rem; color: #6c7086; margin-bottom: 28px; }
  .card { background: #181825; border-radius: 16px; padding: 28px; width: 100%; max-width: 900px; box-shadow: 0 8px 32px #00000060; }
  .section-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #6c7086; margin-bottom: 14px; }
  .config-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; margin-bottom: 24px; }
  .config-item label { display: block; font-size: 0.85rem; color: #a6adc8; margin-bottom: 6px; }
  .config-item select, .config-item input[type=range] { width: 100%; }
  select { background: #313244; color: #cdd6f4; border: none; border-radius: 8px; padding: 9px 12px; font-size: 0.95rem; outline: none; cursor: pointer; }
  select:focus { box-shadow: 0 0 0 2px #89b4fa; }
  .slider-wrap { display: flex; align-items: center; gap: 10px; }
  input[type=range] { flex: 1; accent-color: #89b4fa; }
  .slider-val { font-size: 0.9rem; color: #cba6f7; min-width: 36px; text-align: right; }
  .generate-btn {
    display: block; width: 100%; padding: 14px;
    background: #89b4fa; color: #1e1e2e;
    border: none; border-radius: 10px; font-size: 1.1rem; font-weight: 700;
    cursor: pointer; transition: opacity .15s, transform .1s; margin-bottom: 24px;
  }
  .generate-btn:hover { opacity: 0.88; transform: translateY(-1px); }
  .generate-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
  #status-bar {
    background: #313244; border-radius: 8px; padding: 12px 16px;
    font-size: 0.88rem; color: #fab387; margin-bottom: 20px; display: none;
  }
  .progress-outer { background: #45475a; border-radius: 999px; height: 6px; margin-top: 8px; }
  .progress-inner { background: #89b4fa; height: 6px; border-radius: 999px; width: 0%; transition: width .3s; }
  .roll-section { margin-bottom: 20px; display: none; }
  .canvas-wrap { overflow-x: auto; border-radius: 10px; background: #11111b; padding: 8px; }
  canvas { display: block; }
  .playback-row { display: flex; gap: 12px; flex-wrap: wrap; }
  .btn { border: none; border-radius: 8px; padding: 10px 22px; font-size: 0.95rem; font-weight: 600; cursor: pointer; transition: opacity .15s; }
  .btn:hover { opacity: 0.85; }
  .btn-play { background: #a6e3a1; color: #1e1e2e; }
  .btn-stop { background: #f38ba8; color: #1e1e2e; }
  .btn-dl   { background: #cba6f7; color: #1e1e2e; }
  .btn:disabled { opacity: 0.35; cursor: not-allowed; }
  .stats-row { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 16px; }
  .stat-chip { background: #313244; border-radius: 8px; padding: 8px 14px; font-size: 0.82rem; color: #a6adc8; }
  .stat-chip strong { color: #cdd6f4; }
</style>
</head>
<body>
<h1>AI Music Generator</h1>
<p class="subtitle">LSTM trained on musical scales · Piano Roll · Web Audio playback</p>
<div class="card">

  <div class="section-title">Configuration</div>
  <div class="config-grid">
    <div class="config-item">
      <label for="style">Style</label>
      <select id="style">
        <option value="classical">Classical</option>
        <option value="jazz">Jazz / Blues</option>
        <option value="pentatonic">Pentatonic</option>
        <option value="ambient">Ambient</option>
      </select>
    </div>
    <div class="config-item">
      <label>Tempo: <span id="tempo-val">120</span> BPM</label>
      <div class="slider-wrap">
        <input type="range" id="tempo" min="60" max="180" value="120" oninput="document.getElementById('tempo-val').textContent=this.value">
      </div>
    </div>
    <div class="config-item">
      <label>Notes: <span id="notes-val">128</span></label>
      <div class="slider-wrap">
        <input type="range" id="notes" min="64" max="256" step="32" value="128" oninput="document.getElementById('notes-val').textContent=this.value">
      </div>
    </div>
    <div class="config-item">
      <label>Creativity: <span id="temp-val">0.8</span></label>
      <div class="slider-wrap">
        <input type="range" id="temperature" min="0.4" max="1.4" step="0.1" value="0.8" oninput="document.getElementById('temp-val').textContent=parseFloat(this.value).toFixed(1)">
      </div>
    </div>
  </div>

  <button class="generate-btn" id="genBtn" onclick="generate()">Generate Music</button>

  <div id="status-bar">
    <span id="status-text">Initialising…</span>
    <div class="progress-outer"><div class="progress-inner" id="progress-bar"></div></div>
  </div>

  <div class="roll-section" id="roll-section">
    <div class="section-title">Piano Roll</div>
    <div class="stats-row" id="stats-row"></div>
    <div class="canvas-wrap"><canvas id="pianoRoll" height="240"></canvas></div>
    <br>
    <div class="playback-row">
      <button class="btn btn-play" id="playBtn" onclick="playMusic()" disabled>▶ Play</button>
      <button class="btn btn-stop" id="stopBtn" onclick="stopMusic()" disabled>■ Stop</button>
      <button class="btn btn-dl"   id="dlBtn"   onclick="downloadMidi()" disabled>⬇ Download MIDI</button>
    </div>
  </div>

</div>

<script>
let _notes = [], _durations = [], _tempo = 120, _midiB64 = '';
let _audioCtx = null, _scheduledNodes = [];

async function generate() {
  const btn = document.getElementById('genBtn');
  btn.disabled = true; btn.textContent = 'Training LSTM…';
  const statusBar = document.getElementById('status-bar');
  statusBar.style.display = 'block';
  setStatus('Building training corpus…', 2);
  stopMusic();

  const style = document.getElementById('style').value;
  const tempo = parseInt(document.getElementById('tempo').value);
  const nNotes = parseInt(document.getElementById('notes').value);
  const temperature = parseFloat(document.getElementById('temperature').value);

  try {
    setStatus('Training LSTM model (this takes ~20 seconds)…', 5);
    const res = await fetch('/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({style, tempo, n_notes: nNotes, temperature})
    });
    const data = await res.json();
    if (data.error) { setStatus('Error: ' + data.error, 0); btn.disabled = false; btn.textContent = 'Generate Music'; return; }

    _notes = data.notes;
    _durations = data.durations;
    _tempo = data.tempo;
    _midiB64 = data.midi_b64;

    setStatus('Done! Drawing piano roll…', 100);
    drawPianoRoll(_notes, _durations);
    showStats(data);

    document.getElementById('roll-section').style.display = 'block';
    document.getElementById('playBtn').disabled = false;
    document.getElementById('dlBtn').disabled = false;
    btn.textContent = 'Regenerate';
  } catch(e) {
    setStatus('Error: ' + e, 0);
  } finally {
    btn.disabled = false;
  }
}

function setStatus(msg, pct) {
  document.getElementById('status-text').textContent = msg;
  if (pct !== undefined)
    document.getElementById('progress-bar').style.width = pct + '%';
}

function showStats(data) {
  const pitches = data.notes.filter(n => n > 0);
  const rests = data.notes.filter(n => n === 0).length;
  const uniqueNotes = new Set(pitches).size;
  const midiToName = n => {
    const names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
    return names[n % 12] + Math.floor(n / 12 - 1);
  };
  const lo = midiToName(Math.min(...pitches)), hi = midiToName(Math.max(...pitches));
  const row = document.getElementById('stats-row');
  row.innerHTML = [
    `<div class="stat-chip">Notes: <strong>${data.notes.length}</strong></div>`,
    `<div class="stat-chip">Unique pitches: <strong>${uniqueNotes}</strong></div>`,
    `<div class="stat-chip">Rests: <strong>${rests}</strong></div>`,
    `<div class="stat-chip">Range: <strong>${lo} – ${hi}</strong></div>`,
    `<div class="stat-chip">Tempo: <strong>${data.tempo} BPM</strong></div>`,
    `<div class="stat-chip">Style: <strong>${document.getElementById('style').value}</strong></div>`,
  ].join('');
}

function drawPianoRoll(notes, durations) {
  const canvas = document.getElementById('pianoRoll');
  const ctx = canvas.getContext('2d');
  const pitches = notes.filter(n => n > 0);
  if (!pitches.length) return;
  const minP = Math.max(36, Math.min(...pitches) - 3);
  const maxP = Math.min(96, Math.max(...pitches) + 3);
  const pitchRange = maxP - minP + 1;
  const cellH = Math.floor(240 / pitchRange);
  const PX_PER_BEAT = 12;
  const totalBeats = durations.reduce((a,b) => a+b, 0);
  const W = Math.max(860, Math.ceil(totalBeats * PX_PER_BEAT));
  canvas.width = W;

  // background rows
  for (let p = minP; p <= maxP; p++) {
    const y = (maxP - p) * cellH;
    const isBlack = [1,3,6,8,10].includes(p % 12);
    ctx.fillStyle = isBlack ? '#11111b' : '#1e1e2e';
    ctx.fillRect(0, y, W, cellH);
  }

  // beat grid
  ctx.strokeStyle = '#313244'; ctx.lineWidth = 0.5;
  for (let b = 0; b <= totalBeats; b++) {
    const x = b * PX_PER_BEAT;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 240); ctx.stroke();
  }

  // notes
  let x = 0;
  notes.forEach((pitch, i) => {
    const dur = durations[i];
    const w = dur * PX_PER_BEAT;
    if (pitch > 0 && pitch >= minP && pitch <= maxP) {
      const y = (maxP - pitch) * cellH;
      const hue = Math.round(((pitch - 36) / 60) * 220);
      ctx.fillStyle = `hsl(${hue}, 75%, 62%)`;
      ctx.beginPath();
      ctx.roundRect(x + 1, y + 1, Math.max(w - 2, 2), cellH - 2, 2);
      ctx.fill();
    }
    x += w;
  });
}

function playMusic() {
  stopMusic();
  _audioCtx = new AudioContext();
  const sPerBeat = 60 / _tempo;
  let t = _audioCtx.currentTime + 0.05;
  const masterGain = _audioCtx.createGain();
  masterGain.gain.value = 0.22;
  masterGain.connect(_audioCtx.destination);

  _notes.forEach((pitch, i) => {
    const dur = _durations[i] * sPerBeat;
    if (pitch > 0) {
      const freq = 440 * Math.pow(2, (pitch - 69) / 12);
      const osc = _audioCtx.createOscillator();
      const env = _audioCtx.createGain();
      osc.connect(env); env.connect(masterGain);
      osc.type = 'triangle';
      osc.frequency.value = freq;
      env.gain.setValueAtTime(0, t);
      env.gain.linearRampToValueAtTime(0.9, t + 0.015);
      env.gain.exponentialRampToValueAtTime(0.001, t + Math.max(dur * 0.85, 0.05));
      osc.start(t); osc.stop(t + dur);
      _scheduledNodes.push(osc);
    }
    t += dur;
  });
  document.getElementById('stopBtn').disabled = false;
  document.getElementById('playBtn').textContent = '▶ Playing…';
  setTimeout(() => { document.getElementById('playBtn').textContent = '▶ Play'; }, t * 1000 - _audioCtx.currentTime * 1000);
}

function stopMusic() {
  _scheduledNodes.forEach(n => { try { n.stop(); } catch(e) {} });
  _scheduledNodes = [];
  if (_audioCtx) { _audioCtx.close(); _audioCtx = null; }
  document.getElementById('stopBtn').disabled = true;
  document.getElementById('playBtn').textContent = '▶ Play';
}

function downloadMidi() {
  const a = document.createElement('a');
  a.href = 'data:audio/midi;base64,' + _midiB64;
  a.download = 'generated_music.mid';
  a.click();
}
</script>
</body>
</html>"""


@app.route("/")
def index() -> str:
    return render_template_string(HTML)


@app.route("/generate", methods=["POST"])
def generate_route() -> tuple:
    data: dict = request.get_json(force=True)
    style: str = data.get("style", "classical")
    tempo: int = int(data.get("tempo", 120))
    n_notes: int = int(data.get("n_notes", 128))
    temperature: float = float(data.get("temperature", 0.8))

    with _lock:
        try:
            notes_corpus = build_corpus(style, num_melodies=100, melody_length=64)
            X, y, n2i, i2n, n_vocab = prepare_sequences(notes_corpus)
            model = MusicLSTM(n_vocab)
            train_model(model, X, y, epochs=25)

            encoded = [n2i[n] for n in notes_corpus]
            start = random.randint(0, len(encoded) - SEQ_LEN - 1)
            seed = encoded[start: start + SEQ_LEN]
            generated = generate_notes(model, seed, i2n, n_vocab, length=n_notes, temperature=temperature)

            dur_cycle = [0.5, 0.5, 0.5, 1.0, 0.25, 0.25, 0.5, 1.0]
            durations = [dur_cycle[i % len(dur_cycle)] for i in range(len(generated))]

            midi_b64 = notes_to_midi_b64(generated, durations, tempo)

            return jsonify({
                "notes": generated,
                "durations": durations,
                "tempo": tempo,
                "midi_b64": midi_b64,
            })
        except Exception as exc:
            import traceback
            return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    import threading as _t
    import webbrowser
    port = 5002
    _t.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    print(f"Music Generator running at http://localhost:{port}  (Ctrl+C to stop)")
    app.run(port=port, debug=False)
