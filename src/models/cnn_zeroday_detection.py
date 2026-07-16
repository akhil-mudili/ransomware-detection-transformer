import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = r"D:\ACS\Final Project\ransomware-detection-transformer"
VOCAB_PATH  = os.path.join(BASE_DIR, "data", "vocabulary.pkl")
MODELS_DIR  = os.path.join(BASE_DIR, "results", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "live_detection")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── config ─────────────────────────────────────────────────────────────────────
MAX_SEQ_LEN  = 3000
WINDOW_SIZES = [0.25, 0.50, 0.75, 1.0]
DEVICE       = torch.device("cpu")

# CNN hyperparameters (must match cnn_baseline.py exactly)
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

# ── CNN model definition (must match cnn_baseline.py exactly) ──────────────────
class CNNBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters,
                 kernel_size, fc_dim, dropout, max_seq_len):
        super(CNNBaseline, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        self.conv1 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters * 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        conv_out_len = max_seq_len // 4
        self.flat_size = (num_filters * 2) * conv_out_len
        self.fc1     = nn.Linear(self.flat_size, fc_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(fc_dim, 2)
        self.relu    = nn.ReLU()

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


# ── load all 4 CNN models ──────────────────────────────────────────────────────
def load_models():
    models = {}
    for ws in WINDOW_SIZES:
        label = int(ws * 100)
        path  = os.path.join(MODELS_DIR, f"cnn_window_{label}.pt")
        model = CNNBaseline(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            num_filters=NUM_FILTERS,
            kernel_size=KERNEL_SIZE,
            fc_dim=FC_DIM,
            dropout=DROPOUT,
            max_seq_len=MAX_SEQ_LEN
        ).to(DEVICE)
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        models[ws] = model
        print(f"  Loaded CNN model: {label}% window")
    return models


# ── extract API calls from JSON ────────────────────────────────────────────────
def extract_api_calls(json_path):
    with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
        report = json.load(f)
    calls = []
    for process in report.get("behavior", {}).get("processes", []):
        for call in process.get("calls", []):
            api = call.get("api", "")
            if api:
                calls.append(api)
    return calls


# ── encode sequence ────────────────────────────────────────────────────────────
def encode_and_pad(calls, max_len=MAX_SEQ_LEN):
    encoded = [api_to_id.get(c, UNK_ID) for c in calls]
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        encoded = encoded + [PAD_ID] * (max_len - len(encoded))
    return encoded


# ── run live detection on one sample ──────────────────────────────────────────
def live_detect(json_path, models, true_label=None, family=None):
    filename = os.path.basename(json_path)
    print(f"\n{'='*65}")
    print(f"  SAMPLE: {filename[:60]}")
    if family:
        print(f"  Family: {family} [ZERO-DAY - never seen during training]")
    if true_label is not None:
        print(f"  True label: {'RANSOMWARE' if true_label == 1 else 'BENIGN'}")
    print(f"{'='*65}")

    all_calls = extract_api_calls(json_path)
    total     = len(all_calls)
    print(f"  Total API calls extracted: {total}")

    if total < 10:
        print("  Too few API calls. Skipping.")
        return None

    results = []
    for ws in WINDOW_SIZES:
        label        = int(ws * 100)
        n_calls      = max(1, int(total * ws))
        window_calls = all_calls[:n_calls]
        encoded      = encode_and_pad(window_calls)
        tensor       = torch.tensor([encoded], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            output     = models[ws](tensor)
            probs      = torch.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probs[0][prediction].item()

        label_str = "RANSOMWARE" if prediction == 1 else "BENIGN"
        correct   = ""
        if true_label is not None:
            correct = "CORRECT" if prediction == true_label else "WRONG"

        print(
            f"  [{label:3d}% window | {n_calls:5d} calls] "
            f"-> {label_str:10s} "
            f"(confidence: {confidence*100:.1f}%) "
            f"{correct}"
        )

        results.append({
            "window":       label,
            "n_calls_used": n_calls,
            "prediction":   label_str,
            "confidence":   round(confidence * 100, 2),
            "correct":      correct
        })

    return results


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading trained CNN models...")
    models = load_models()

    zeroday_families = {
        "wannacry":     r"D:\ACS\Final Project\ransomware dataset\some ransomware cuckoo analysis report\wannacry\wannacry",
        "cryptolocker": r"D:\ACS\Final Project\ransomware dataset\582个ransomware_cuckoo analysis report\ransomware_report\cryptolocker",
    }

    samples_per_family = 5
    all_results = []

    print(f"\n{'='*65}")
    print("  CNN ZERO-DAY GENERALIZATION TEST")
    print("  Model trained on 9 families. Testing on UNSEEN families.")
    print(f"{'='*65}")

    for family, folder in zeroday_families.items():
        print(f"\n>>> Family: {family.upper()}")
        if not os.path.exists(folder):
            print(f"  Folder not found: {folder}")
            continue
        files = [f for f in os.listdir(folder) if f.endswith(".json")][:samples_per_family]
        for fname in files:
            fpath  = os.path.join(folder, fname)
            result = live_detect(fpath, models, true_label=1, family=family)
            if result:
                all_results.append({
                    "file":           fname,
                    "family":         family,
                    "true_label":     "RANSOMWARE",
                    "zero_day":       True,
                    "window_results": result
                })

    # save results
    out_path = os.path.join(RESULTS_DIR, "cnn_zeroday_detection_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*65}")
    print("CNN ZERO-DAY TEST COMPLETE")
    print(f"{'='*65}")

    print("\nSUMMARY:")
    for family in zeroday_families.keys():
        family_results = [r for r in all_results if r["family"] == family]
        print(f"\n  {family.upper()} ({len(family_results)} samples):")
        for ws in WINDOW_SIZES:
            label   = int(ws * 100)
            correct = sum(
                1 for r in family_results
                for w in r["window_results"]
                if w["window"] == label and w["correct"] == "CORRECT"
            )
            total_s = len(family_results)
            print(f"    {label}% window: {correct}/{total_s} correctly identified as RANSOMWARE")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
