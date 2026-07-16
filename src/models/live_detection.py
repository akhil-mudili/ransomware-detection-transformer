import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = r"D:\ACS\Final Project\ransomware-detection-transformer"
VOCAB_PATH  = os.path.join(BASE_DIR, "data", "vocabulary.pkl")
MODELS_DIR  = os.path.join(BASE_DIR, "src", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "live_detection")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── config ─────────────────────────────────────────────────────────────────────
MAX_SEQ_LEN  = 3000
WINDOW_SIZES = [0.25, 0.50, 0.75, 1.0]
DEVICE       = torch.device("cpu")

EMBED_DIM = 64
N_HEADS   = 4
N_LAYERS  = 2
FF_DIM    = 256
DROPOUT   = 0.1

# ── load vocabulary ────────────────────────────────────────────────────────────
with open(VOCAB_PATH, "rb") as f:
    vocab_data = pickle.load(f)
api_to_id  = vocab_data["api_to_id"]
PAD_ID     = vocab_data["pad_id"]
UNK_ID     = api_to_id.get("<UNK>", 1)
VOCAB_SIZE  = vocab_data["vocab_size"] + 1

# ── model definition (must match training) ─────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=EMBED_DIM,
            padding_idx=0
        )
        self.pos_encoding = PositionalEncoding(EMBED_DIM, MAX_SEQ_LEN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=N_HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=N_LAYERS
        )
        self.dropout = nn.Dropout(0.1)
        self.fc      = nn.Linear(EMBED_DIM, 1)

    def forward(self, x):
        pad_mask = (x == 0)
        emb = self.pos_encoding(self.embedding(x))
        enc = self.transformer_encoder(emb, src_key_padding_mask=pad_mask)
        mask_expanded = (~pad_mask).unsqueeze(-1).float()
        pooled = (enc * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return self.fc(self.dropout(pooled)).squeeze(-1)


# ── load all 4 models ──────────────────────────────────────────────────────────
def load_models():
    models = {}
    for ws in WINDOW_SIZES:
        label = int(ws * 100)
        path  = os.path.join(MODELS_DIR, f"transformer_window_{label}.pt")
        model = TransformerClassifier().to(DEVICE)
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        models[ws] = model
        print(f"  Loaded model: {label}% window")
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
            logit      = models[ws](tensor)
            prob       = torch.sigmoid(logit).item()
            prediction = 1 if prob >= 0.5 else 0
            confidence = prob if prediction == 1 else 1 - prob

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
    print("Loading trained Transformer models...")
    models = load_models()

    # Zero-day families - NEVER seen during training
    zeroday_families = {
        "wannacry":     r"D:\ACS\Final Project\ransomware dataset\some ransomware cuckoo analysis report\wannacry\wannacry",
        "cryptolocker": r"D:\ACS\Final Project\ransomware dataset\582个ransomware_cuckoo analysis report\ransomware_report\cryptolocker",
    }

    samples_per_family = 5
    all_results = []

    print(f"\n{'='*65}")
    print("  ZERO-DAY GENERALIZATION TEST")
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
                    "file":          fname,
                    "family":        family,
                    "true_label":    "RANSOMWARE",
                    "zero_day":      True,
                    "window_results": result
                })

    # save results
    out_path = os.path.join(RESULTS_DIR, "zeroday_detection_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*65}")
    print("ZERO-DAY TEST COMPLETE")
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