"""
AI Music Generation with LSTM (PyTorch)
- Trains a 2-layer LSTM on synthetic note sequences (scales/patterns)
- Generates new music with temperature sampling
- Saves output as a MIDI file via music21 and auto-opens it

Run (after activating .venv):
  source .venv/bin/activate
  python 'Task 3/music_generation.py'
"""

import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# 1. Synthetic note dataset
# ---------------------------------------------------------------------------

C_MAJOR = [60, 62, 64, 65, 67, 69, 71, 72]
A_MINOR = [69, 71, 72, 74, 76, 77, 79, 81]
G_MAJOR = [55, 57, 59, 60, 62, 64, 66, 67]
BLUES   = [60, 63, 65, 66, 67, 70, 72]
PENTA   = [60, 62, 64, 67, 69, 72]
SCALES  = [C_MAJOR, A_MINOR, G_MAJOR, BLUES, PENTA]


def generate_melody(length: int = 64) -> list[int]:
    scale = random.choice(SCALES)
    melody: list[int] = []
    for _ in range(length):
        melody.append(0 if random.random() < 0.08 else random.choice(scale))
    return melody


def build_corpus(num_melodies: int = 120, melody_length: int = 64) -> list[int]:
    corpus: list[int] = []
    for _ in range(num_melodies):
        corpus.extend(generate_melody(melody_length))
    return corpus


# ---------------------------------------------------------------------------
# 2. Sequence preparation
# ---------------------------------------------------------------------------

SEQ_LEN = 16


def prepare_sequences(
    notes: list[int], seq_len: int = SEQ_LEN
) -> tuple[torch.Tensor, torch.Tensor, dict, dict, int]:
    unique = sorted(set(notes))
    note_to_int: dict[int, int] = {n: i for i, n in enumerate(unique)}
    int_to_note: dict[int, int] = {i: n for n, i in note_to_int.items()}
    n_vocab = len(unique)

    encoded = [note_to_int[n] for n in notes]
    X_list, y_list = [], []
    for i in range(len(encoded) - seq_len):
        X_list.append(encoded[i : i + seq_len])
        y_list.append(encoded[i + seq_len])

    X = torch.tensor(X_list, dtype=torch.float32).unsqueeze(-1) / n_vocab
    y = torch.tensor(y_list, dtype=torch.long)
    return X, y, note_to_int, int_to_note, n_vocab


# ---------------------------------------------------------------------------
# 3. LSTM model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------

def train(
    model: MusicLSTM, X: torch.Tensor, y: torch.Tensor,
    epochs: int = 40, batch_size: int = 64, lr: float = 1e-3,
) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"      Device: {device}")
    model.to(device)

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    best_loss = float("inf")
    patience_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        scheduler.step(avg)
        print(f"      Epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

        if avg < best_loss - 1e-4:
            best_loss = avg
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 8:
                print("      Early stopping.")
                break

    model.to("cpu")


# ---------------------------------------------------------------------------
# 5. Generation with temperature sampling
# ---------------------------------------------------------------------------

def generate_notes(
    model: MusicLSTM, seed: list[int], int_to_note: dict,
    n_vocab: int, length: int = 128, temperature: float = 0.8,
) -> list[int]:
    model.eval()
    current = list(seed)
    generated = list(seed)

    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor(current, dtype=torch.float32).unsqueeze(0).unsqueeze(-1) / n_vocab
            logits = model(x)[0]
            logits = logits / temperature
            probs = torch.softmax(logits, dim=0).numpy()
            chosen = int(np.random.choice(len(probs), p=probs))
            generated.append(chosen)
            current = current[1:] + [chosen]

    return [int_to_note[i] for i in generated]


# ---------------------------------------------------------------------------
# 6. Save as MIDI
# ---------------------------------------------------------------------------

def notes_to_midi(pitches: list[int], output_path: str, tempo: int = 120) -> str:
    from music21 import note, stream
    from music21 import tempo as m21tempo
    from music21 import midi as m21midi

    s = stream.Stream()
    s.append(m21tempo.MetronomeMark(number=tempo))
    dur_cycle = [0.5, 0.5, 0.5, 1.0, 0.25, 0.25, 0.5, 1.0]

    for idx, pitch in enumerate(pitches):
        dur = dur_cycle[idx % len(dur_cycle)]
        n = note.Rest(quarterLength=dur) if pitch == 0 else note.Note(pitch, quarterLength=dur)
        s.append(n)

    mf = m21midi.translate.streamToMidiFile(s)
    mf.open(output_path, "wb")
    mf.write()
    mf.close()
    print(f"      MIDI saved → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 55)
    print("  AI Music Generation with LSTM (PyTorch)")
    print("=" * 55)

    print("\n[1/5] Generating training corpus ...")
    notes = build_corpus(num_melodies=120, melody_length=64)
    print(f"      Notes: {len(notes)}  |  Unique pitches: {len(set(notes))}")

    print("\n[2/5] Preparing sequences ...")
    X, y, note_to_int, int_to_note, n_vocab = prepare_sequences(notes)
    print(f"      Sequences: {len(X)}  |  Vocab: {n_vocab}")

    print("\n[3/5] Building LSTM model ...")
    model = MusicLSTM(n_vocab)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters: {total_params:,}")

    print("\n[4/5] Training (up to 40 epochs) ...")
    train(model, X, y, epochs=40, batch_size=64)

    print("\n[5/5] Generating 128 notes ...")
    encoded = [note_to_int[n] for n in notes]
    start = random.randint(0, len(encoded) - SEQ_LEN - 1)
    seed = encoded[start : start + SEQ_LEN]

    generated = generate_notes(model, seed, int_to_note, n_vocab, length=128, temperature=0.8)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "generated_music.mid")
    notes_to_midi(generated, out_path)

    import platform
    import subprocess as sp
    system = platform.system()
    try:
        if system == "Darwin":
            sp.Popen(["open", out_path])
        elif system == "Windows":
            os.startfile(out_path)
        else:
            sp.Popen(["xdg-open", out_path])
    except Exception:
        print(f"      Could not auto-open. File is at: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
