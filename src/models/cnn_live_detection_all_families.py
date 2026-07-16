"""
cnn_live_detection_all_families.py - CNN Baseline
==================================================
Runs live detection on all 9 families (4 known + 5 zero-day)
and saves results to JSON. No dashboard, pure terminal output.
Author: Akhil Mudili | University of Galway
"""

import os
import json
import pickle
import torch
import torch.nn as nn

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = r"D:\ACS\Final Project\ransomware-detection-transformer"
VOCAB_PATH  = os.path.join(BASE_DIR, "data", "vocabulary.pkl")
MODELS_DIR  = os.path.join(BASE_DIR, "results", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "live_detection")
VT_BASE     = r"D:\ACS\Final Project\ransomware dataset\vt_cuckoo_reports_final"
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_SEQ_LEN  = 3000
WINDOW_SIZES = [0.25, 0.50, 0.75, 1.0]
DEVICE       = torch.device("cpu")
EMBED_DIM    = 64
NUM_FILTERS  = 128
KERNEL_SIZE  = 5
FC_DIM       = 128
DROPOUT      = 0.3

# ── load vocabulary ────────────────────────────────────────────────────────────
with open(VOCAB_PATH, "rb") as f:
    vocab_data = pickle.load(f)
api_to_id  = vocab_data["api_to_id"]
PAD_ID     = vocab_data["pad_id"]
UNK_ID     = api_to_id.get("<UNK>", 1)
VOCAB_SIZE  = vocab_data["vocab_size"]

# ── CNN model ──────────────────────────────────────────────────────────────────
class CNNBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_size, fc_dim, dropout, max_seq_len):
        super(CNNBaseline, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pool2    = nn.MaxPool1d(kernel_size=2)
        conv_out_len  = max_seq_len // 4
        self.flat_size = (num_filters * 2) * conv_out_len
        self.fc1      = nn.Linear(self.flat_size, fc_dim)
        self.dropout  = nn.Dropout(dropout)
        self.fc2      = nn.Linear(fc_dim, 2)
        self.relu     = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ── helpers ────────────────────────────────────────────────────────────────────
def load_models():
    models = {}
    for ws in WINDOW_SIZES:
        label = int(ws * 100)
        path  = os.path.join(MODELS_DIR, f"cnn_window_{label}.pt")
        model = CNNBaseline(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_filters=NUM_FILTERS,
                            kernel_size=KERNEL_SIZE, fc_dim=FC_DIM, dropout=DROPOUT, max_seq_len=MAX_SEQ_LEN).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        models[ws] = model
        print(f"  Loaded cnn_window_{label}.pt")
    return models

def extract_api_calls(json_path):
    with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
        report = json.load(f)
    calls = []
    for p in report.get("behavior", {}).get("processes", []):
        for c in p.get("calls", []):
            if c.get("api"):
                calls.append(c["api"])
    return calls

def encode_and_pad(calls):
    enc = [api_to_id.get(c, UNK_ID) for c in calls]
    if len(enc) > MAX_SEQ_LEN:
        enc = enc[:MAX_SEQ_LEN]
    else:
        enc = enc + [PAD_ID] * (MAX_SEQ_LEN - len(enc))
    return enc

def detect_sample(models, json_path, family, zeroday):
    calls = extract_api_calls(json_path)
    total = len(calls)
    if total < 10:
        return None

    results = []
    for ws in WINDOW_SIZES:
        label   = int(ws * 100)
        n_calls = max(1, int(total * ws))
        enc     = encode_and_pad(calls[:n_calls])
        tensor  = torch.tensor([enc], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            output     = models[ws](tensor)
            probs      = torch.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probs[0][prediction].item()

        verdict = "RANSOMWARE" if prediction == 1 else "BENIGN"
        correct = prediction == 1
        status  = "CORRECT" if correct else "WRONG"

        print(f"    [{label:3d}% | {n_calls:5d} calls] -> {verdict:10s} ({confidence*100:.1f}%) {status}")
        results.append({
            "window": label,
            "n_calls": n_calls,
            "verdict": verdict,
            "confidence": round(confidence * 100, 1),
            "correct": correct
        })

    return {
        "file": os.path.basename(json_path),
        "family": family,
        "zeroday": zeroday,
        "total_calls": total,
        "windows": results
    }

# ── families config ────────────────────────────────────────────────────────────
FAMILIES = [
    {"folder": r"D:\ACS\Final Project\ransomware dataset\some ransomware cuckoo analysis report\darkside",            "family": "darkside",    "zeroday": False},
    {"folder": r"D:\ACS\Final Project\ransomware dataset\some ransomware cuckoo analysis report\locky\locky",         "family": "locky",       "zeroday": False},
    {"folder": r"D:\ACS\Final Project\ransomware dataset\some ransomware cuckoo analysis report\ryuk",                "family": "ryuk",        "zeroday": False},
    {"folder": r"D:\ACS\Final Project\ransomware dataset\some ransomware cuckoo analysis report\reveton",             "family": "reveton",     "zeroday": False},
    {"folder": r"D:\ACS\Final Project\ransomware dataset\some ransomware cuckoo analysis report\wannacry\wannacry",   "family": "wannacry",    "zeroday": True},
    {"folder": r"D:\ACS\Final Project\ransomware dataset\some ransomware cuckoo analysis report\sodinokibi\sodinokibi","family": "sodinokibi", "zeroday": True},
    {"folder": os.path.join(VT_BASE, "crowti"),    "family": "crowti",    "zeroday": True},
    {"folder": os.path.join(VT_BASE, "cryptodef"), "family": "cryptodef", "zeroday": True},
    {"folder": os.path.join(VT_BASE, "ctblocker"), "family": "ctblocker", "zeroday": True},
]

SAMPLES_PER_FAMILY = 5

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading CNN models...")
    models = load_models()
    print()

    all_results = []

    for cfg in FAMILIES:
        folder  = cfg["folder"]
        family  = cfg["family"]
        zeroday = cfg["zeroday"]
        tag     = "[ZERO-DAY]" if zeroday else "[KNOWN]  "

        if not os.path.exists(folder):
            print(f"  Folder not found: {folder}")
            continue

        files = [f for f in os.listdir(folder) if f.endswith(".json")][:SAMPLES_PER_FAMILY]
        if not files:
            print(f"  No JSON files found in: {folder}")
            continue

        print(f"{'='*65}")
        print(f"  {tag} {family.upper()} ({len(files)} samples)")
        print(f"{'='*65}")

        for fname in files:
            fpath = os.path.join(folder, fname)
            print(f"\n  Sample: {fname[:50]}")
            result = detect_sample(models, fpath, family, zeroday)
            if result:
                all_results.append(result)

    # save results
    out_path = os.path.join(RESULTS_DIR, "cnn_all_families_live.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # print summary
    print(f"\n{'='*65}")
    print("CNN LIVE DETECTION - FULL SUMMARY")
    print(f"{'='*65}")

    for ws in [25, 50, 75, 100]:
        known_correct   = sum(1 for r in all_results if not r["zeroday"] for w in r["windows"] if w["window"] == ws and w["correct"])
        known_total     = sum(1 for r in all_results if not r["zeroday"])
        zeroday_correct = sum(1 for r in all_results if r["zeroday"] for w in r["windows"] if w["window"] == ws and w["correct"])
        zeroday_total   = sum(1 for r in all_results if r["zeroday"])
        total_correct   = known_correct + zeroday_correct
        total_samples   = known_total + zeroday_total

        print(f"\n  {ws}% window:")
        print(f"    Known families:    {known_correct}/{known_total}")
        print(f"    Zero-day families: {zeroday_correct}/{zeroday_total}")
        print(f"    Overall:           {total_correct}/{total_samples}")

    print(f"\nResults saved to: {out_path}")
    print(f"{'='*65}")

if __name__ == "__main__":
    main()
