#!/usr/bin/env bash
set -e

VENV_DIR="$(dirname "$0")/.venv"

echo "Creating virtual environment at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

echo "Installing dependencies for all tasks ..."
"$VENV_DIR/bin/pip" install -q --upgrade pip

# Task 1
"$VENV_DIR/bin/pip" install -q flask deep-translator

# Task 2
"$VENV_DIR/bin/pip" install -q flask nltk scikit-learn numpy

# Task 3
"$VENV_DIR/bin/pip" install -q music21 torch numpy

# Task 4
"$VENV_DIR/bin/pip" install -q ultralytics opencv-python numpy filterpy scipy

echo ""
echo "Setup complete! Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then run any task, e.g.:"
echo "  python 'Task 1/translator.py'"
echo "  python 'Task 2/chatbot.py'"
echo "  python 'Task 3/music_generation.py'"
echo "  python 'Task 4 /object_detection.py'"
