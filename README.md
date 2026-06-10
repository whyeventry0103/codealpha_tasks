# CodeAlpha AI Internship — Muhammad Ali Shah

Four AI/ML projects built during the CodeAlpha internship program.

## Quick Start

```bash
# One-time setup
bash setup.sh

# Activate environment (every session)
source .venv/bin/activate
```

---

## Task 1 — Language Translation Tool

**Tech:** Flask · deep-translator (Google Translate) · gTTS · langdetect

```bash
python "Task 1/translator.py"   # opens http://localhost:5000
```

**Features:**
- Translate between 20+ languages
- Auto-detects source language with a purple badge ("Detected: French")
- Word & character count live as you type
- Text-to-Speech via gTTS (server-side, works in all browsers)
- Swap languages button, Copy to clipboard
- Ctrl+Enter shortcut to translate

---

## Task 2 — FAQ Chatbot

**Tech:** Flask · NLTK (tokenization, lemmatization) · scikit-learn (TF-IDF + cosine similarity)

```bash
python "Task 2/chatbot.py"   # opens http://localhost:5001
```

**Features:**
- 30 Python & Machine Learning FAQ pairs
- NLP pipeline: tokenize → stopword removal → lemmatize → TF-IDF vectorize → cosine similarity match
- Colour-coded confidence badge per response (green ≥ 60%, yellow ≥ 35%, red < 35%)
- 6 clickable suggested question chips
- Typing indicator animation

---

## Task 3 — AI Music Generation

**Tech:** Flask · PyTorch (2-layer LSTM) · music21 · Web Audio API

```bash
python "Task 3/music_generation.py"   # opens http://localhost:5002
```

**Features:**
- Choose style: Classical · Jazz/Blues · Pentatonic · Ambient
- Configure tempo (60–180 BPM), note count (64–256), creativity/temperature (0.4–1.4)
- Trains LSTM in ~20 seconds, generates new music with temperature sampling
- Interactive piano roll canvas (colour-coded by pitch register)
- In-browser playback via Web Audio API (triangle oscillator with ADSR envelope)
- Download generated `.mid` file

---

## Task 4 — Object Detection & Tracking

**Tech:** YOLOv8 (ultralytics) · OpenCV · SORT tracker (Kalman filter + Hungarian algorithm)

```bash
python "Task 4 /object_detection.py"           # webcam
python "Task 4 /object_detection.py video.mp4" # video file
```

**Features:**
- Real-time YOLOv8n detection at full webcam FPS
- SORT multi-object tracker — each object gets a persistent colour-coded ID
- Live stats overlay: per-class object counts + FPS
- Press **S** to save a screenshot → `output/` folder
- Press **R** to start/stop recording → saves `.mp4` to `output/` folder
- Press **Q** to quit

---

## Dependencies

All dependencies are installed into the `.venv` virtual environment by `setup.sh`.

| Task | Key Libraries |
|------|--------------|
| 1    | flask, deep-translator, gTTS, langdetect |
| 2    | flask, nltk, scikit-learn |
| 3    | flask, torch, music21 |
| 4    | ultralytics, opencv-python, filterpy, scipy |

> **Note:** `yolov8n.pt` model weights are excluded from git (listed in `.gitignore`) and are auto-downloaded by ultralytics on first run.
