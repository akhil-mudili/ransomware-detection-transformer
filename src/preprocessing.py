"""
preprocessing.py
================
Ransomware Detection Project — Data Preprocessing Pipeline
Author: Akhil Mudili
Supervisor: Dr. Priyanka Verma
University of Galway

What this script does:
1. Reads all usable Cuckoo Sandbox JSON reports (ransomware + benign)
2. Extracts API call sequences from behavior -> processes -> calls -> api
3. Filters out unusable samples (fewer than 10 API calls)
4. Caps sequences at MAX_SEQ_LEN (3000 calls)
5. Builds a vocabulary of all unique API call names
6. Encodes API call names to integer IDs
7. Creates four windowed versions of each sequence (25%, 50%, 75%, 100%)
8. Saves everything to disk ready for model training
"""

import os
import json
import pickle
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION — adjust paths if needed
# ============================================================

BASE_DIR = r"D:\ACS\Final Project\ransomware dataset"
EXTRACT_DIR = os.path.join(BASE_DIR, "_extracted_temp")
OUTPUT_DIR = r"D:\ACS\Final Project\ransomware-detection-transformer\data"

# Ransomware source folders (already extracted)
RANSOMWARE_FOLDERS = [
    os.path.join(BASE_DIR, "some ransomware cuckoo analysis report"),
    os.path.join(EXTRACT_DIR, "582_ransomware_cuckoo analysis report"),
]

# Benign source folders (already extracted)
BENIGN_FOLDERS = [
    os.path.join(EXTRACT_DIR, "benign sample_cuckoo analysis report"),
    os.path.join(EXTRACT_DIR, "benign_sample_report_cuckoo analysis report_2",
                 "benign_report", "benign_report"),
]

# Preprocessing parameters
MIN_SEQ_LEN = 10        # Minimum API calls to be considered usable
MAX_SEQ_LEN = 3000      # Cap sequences at this length
WINDOW_SIZES = [0.25, 0.50, 0.75, 1.00]  # Partial observation windows

# Train/val/test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# ============================================================
# HELPERS
# ============================================================

def find_json_files(folder):
    """Recursively find all JSON files under a folder."""
    json_files = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))
    return json_files

def get_family_name(json_path, base_folder):
    """
    Infer ransomware family from subfolder name.
    Handles nested structures by taking the deepest subfolder.
    """
    rel = os.path.relpath(json_path, base_folder)
    parts = rel.split(os.sep)
    folders = parts[:-1]
    if len(folders) == 0:
        return "unknown"
    elif len(folders) == 1:
        return folders[0].lower()
    else:
        return folders[-1].lower()

def extract_api_calls(data):
    """
    Extract ordered API call sequence from Cuckoo JSON report.
    Path: behavior -> processes -> calls -> api
    Combines calls from all processes in order.
    """
    api_calls = []
    try:
        processes = data.get("behavior", {}).get("processes", [])
        for proc in processes:
            calls = proc.get("calls", [])
            for call in calls:
                api = call.get("api", None)
                if api:
                    api_calls.append(api)
    except Exception:
        pass
    return api_calls

def apply_window(sequence, window_fraction, max_len):
    """
    Apply a partial observation window to a sequence.
    Takes the first (window_fraction * len) calls, then pads/truncates to max_len.
    """
    window_len = max(1, int(len(sequence) * window_fraction))
    windowed = sequence[:window_len]
    # Truncate to max_len if needed
    windowed = windowed[:max_len]
    return windowed

def pad_sequence(sequence, max_len, pad_id):
    """Pad sequence to max_len with pad_id."""
    padded = sequence + [pad_id] * (max_len - len(sequence))
    return padded[:max_len]

# ============================================================
# STEP 1: LOAD ALL SAMPLES
# ============================================================

def load_all_samples():
    print("\n" + "="*60)
    print("STEP 1: Loading samples from JSON files")
    print("="*60)

    all_samples = []

    # Load ransomware
    print("\nLoading ransomware samples...")
    for folder in RANSOMWARE_FOLDERS:
        if not os.path.exists(folder):
            print(f"  WARNING: Folder not found: {folder}")
            continue
        json_files = find_json_files(folder)
        print(f"  Found {len(json_files)} files in: {os.path.basename(folder)}")

        for i, jf in enumerate(json_files):
            if (i+1) % 100 == 0:
                print(f"    Processed {i+1}/{len(json_files)}...")
            try:
                with open(jf, 'r', encoding='utf-8', errors='ignore') as f:
                    data = json.load(f)
                api_calls = extract_api_calls(data)
                family = get_family_name(jf, folder)
                all_samples.append({
                    "api_calls": api_calls,
                    "label": 1,           # 1 = ransomware
                    "family": family,
                    "file": os.path.basename(jf),
                })
            except Exception as e:
                continue

    # Load benign
    print("\nLoading benign samples...")
    for folder in BENIGN_FOLDERS:
        if not os.path.exists(folder):
            print(f"  WARNING: Folder not found: {folder}")
            continue
        json_files = find_json_files(folder)
        print(f"  Found {len(json_files)} files in: {os.path.basename(folder)}")

        for i, jf in enumerate(json_files):
            if (i+1) % 100 == 0:
                print(f"    Processed {i+1}/{len(json_files)}...")
            try:
                with open(jf, 'r', encoding='utf-8', errors='ignore') as f:
                    data = json.load(f)
                api_calls = extract_api_calls(data)
                all_samples.append({
                    "api_calls": api_calls,
                    "label": 0,           # 0 = benign
                    "family": "benign",
                    "file": os.path.basename(jf),
                })
            except Exception as e:
                continue

    print(f"\nTotal samples loaded: {len(all_samples)}")
    ransomware_count = sum(1 for s in all_samples if s["label"] == 1)
    benign_count = sum(1 for s in all_samples if s["label"] == 0)
    print(f"  Ransomware: {ransomware_count}")
    print(f"  Benign:     {benign_count}")

    return all_samples

# ============================================================
# STEP 2: FILTER UNUSABLE SAMPLES
# ============================================================

def filter_samples(all_samples):
    print("\n" + "="*60)
    print("STEP 2: Filtering unusable samples")
    print("="*60)

    usable = [s for s in all_samples if len(s["api_calls"]) >= MIN_SEQ_LEN]
    removed = len(all_samples) - len(usable)

    print(f"  Removed {removed} samples with fewer than {MIN_SEQ_LEN} API calls")
    print(f"  Remaining samples: {len(usable)}")

    ransomware_count = sum(1 for s in usable if s["label"] == 1)
    benign_count = sum(1 for s in usable if s["label"] == 0)
    print(f"  Ransomware: {ransomware_count}")
    print(f"  Benign:     {benign_count}")

    # Print per-family counts
    family_counts = defaultdict(int)
    for s in usable:
        family_counts[s["family"]] += 1
    print("\n  Samples per family:")
    for fam, count in sorted(family_counts.items(), key=lambda x: -x[1]):
        print(f"    {fam:<30} {count}")

    return usable

# ============================================================
# STEP 3: BUILD VOCABULARY
# ============================================================

def build_vocabulary(samples):
    print("\n" + "="*60)
    print("STEP 3: Building API call vocabulary")
    print("="*60)

    # Collect all unique API calls
    all_apis = set()
    for s in samples:
        all_apis.update(s["api_calls"])

    # Build vocab: special tokens first, then sorted API names
    vocab = [PAD_TOKEN, UNK_TOKEN] + sorted(all_apis)
    api_to_id = {api: idx for idx, api in enumerate(vocab)}
    id_to_api = {idx: api for api, idx in api_to_id.items()}

    print(f"  Vocabulary size: {len(vocab)} (including PAD and UNK tokens)")
    print(f"  PAD token ID: {api_to_id[PAD_TOKEN]}")
    print(f"  UNK token ID: {api_to_id[UNK_TOKEN]}")

    return vocab, api_to_id, id_to_api

# ============================================================
# STEP 4: ENCODE AND WINDOW SEQUENCES
# ============================================================

def encode_and_window(samples, api_to_id):
    print("\n" + "="*60)
    print("STEP 4: Encoding sequences and applying windows")
    print("="*60)

    pad_id = api_to_id[PAD_TOKEN]
    unk_id = api_to_id[UNK_TOKEN]

    processed = []
    seq_lengths = []

    for s in samples:
        # Encode API names to integers
        full_seq = [api_to_id.get(api, unk_id) for api in s["api_calls"]]

        # Cap at MAX_SEQ_LEN
        full_seq = full_seq[:MAX_SEQ_LEN]
        seq_lengths.append(len(full_seq))

        # Create windowed versions
        windows = {}
        for w in WINDOW_SIZES:
            windowed = apply_window(full_seq, w, MAX_SEQ_LEN)
            padded = pad_sequence(windowed, MAX_SEQ_LEN, pad_id)
            windows[w] = padded

        processed.append({
            "label": s["label"],
            "family": s["family"],
            "file": s["file"],
            "seq_len": len(full_seq),
            "windows": windows,
        })

    # Report sequence length stats after capping
    seq_lengths = np.array(seq_lengths)
    print(f"  Sequence lengths after capping at {MAX_SEQ_LEN}:")
    print(f"    Min:    {seq_lengths.min()}")
    print(f"    Max:    {seq_lengths.max()}")
    print(f"    Mean:   {seq_lengths.mean():.1f}")
    print(f"    Median: {np.median(seq_lengths):.1f}")
    capped = np.sum(seq_lengths == MAX_SEQ_LEN)
    print(f"    Samples capped at {MAX_SEQ_LEN}: {capped}")

    return processed

# ============================================================
# STEP 5: TRAIN/VAL/TEST SPLIT
# ============================================================

def split_dataset(processed):
    print("\n" + "="*60)
    print("STEP 5: Splitting into train/val/test sets")
    print("="*60)

    labels = [s["label"] for s in processed]

    # First split: train vs temp (val + test)
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        processed, labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=labels,
        random_state=RANDOM_SEED
    )

    # Second split: val vs test
    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp_labels,
        random_state=RANDOM_SEED
    )

    def split_summary(data, name):
        ransomware = sum(1 for s in data if s["label"] == 1)
        benign = sum(1 for s in data if s["label"] == 0)
        print(f"  {name}: {len(data)} samples "
              f"(ransomware: {ransomware}, benign: {benign})")

    split_summary(train_data, "Train")
    split_summary(val_data,   "Val  ")
    split_summary(test_data,  "Test ")

    return train_data, val_data, test_data

# ============================================================
# STEP 6: COMPUTE CLASS WEIGHTS
# ============================================================

def compute_class_weights(train_data):
    print("\n" + "="*60)
    print("STEP 6: Computing class weights for weighted loss")
    print("="*60)

    n_total = len(train_data)
    n_ransomware = sum(1 for s in train_data if s["label"] == 1)
    n_benign = sum(1 for s in train_data if s["label"] == 0)

    # Standard sklearn-style class weight: n_total / (n_classes * n_class_i)
    weight_ransomware = n_total / (2 * n_ransomware)
    weight_benign = n_total / (2 * n_benign)

    class_weights = {0: weight_benign, 1: weight_ransomware}

    print(f"  Ransomware samples in train: {n_ransomware} -> weight: {weight_ransomware:.4f}")
    print(f"  Benign samples in train:     {n_benign} -> weight: {weight_benign:.4f}")

    return class_weights

# ============================================================
# STEP 7: SAVE EVERYTHING
# ============================================================

def save_outputs(train_data, val_data, test_data,
                 vocab, api_to_id, id_to_api, class_weights):
    print("\n" + "="*60)
    print("STEP 7: Saving processed data")
    print("="*60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save splits
    for name, data in [("train", train_data),
                        ("val",   val_data),
                        ("test",  test_data)]:
        path = os.path.join(OUTPUT_DIR, f"{name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  Saved {name}.pkl ({len(data)} samples)")

    # Save vocabulary
    vocab_data = {
        "vocab": vocab,
        "api_to_id": api_to_id,
        "id_to_api": id_to_api,
        "vocab_size": len(vocab),
        "pad_id": api_to_id[PAD_TOKEN],
        "unk_id": api_to_id[UNK_TOKEN],
    }
    vocab_path = os.path.join(OUTPUT_DIR, "vocabulary.pkl")
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_data, f)
    print(f"  Saved vocabulary.pkl (size: {len(vocab)})")

    # Save class weights
    weights_path = os.path.join(OUTPUT_DIR, "class_weights.pkl")
    with open(weights_path, 'wb') as f:
        pickle.dump(class_weights, f)
    print(f"  Saved class_weights.pkl")

    # Save config/metadata
    config = {
        "min_seq_len": MIN_SEQ_LEN,
        "max_seq_len": MAX_SEQ_LEN,
        "window_sizes": WINDOW_SIZES,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "random_seed": RANDOM_SEED,
        "vocab_size": len(vocab),
        "n_train": len(train_data),
        "n_val": len(val_data),
        "n_test": len(test_data),
        "class_weights": class_weights,
    }
    config_path = os.path.join(OUTPUT_DIR, "config.pkl")
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"  Saved config.pkl")

    print(f"\n  All outputs saved to: {OUTPUT_DIR}")

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("RANSOMWARE DETECTION — PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Load
    all_samples = load_all_samples()

    # Step 2: Filter
    usable_samples = filter_samples(all_samples)

    # Step 3: Vocabulary
    vocab, api_to_id, id_to_api = build_vocabulary(usable_samples)

    # Step 4: Encode and window
    processed = encode_and_window(usable_samples, api_to_id)

    # Step 5: Split
    train_data, val_data, test_data = split_dataset(processed)

    # Step 6: Class weights
    class_weights = compute_class_weights(train_data)

    # Step 7: Save
    save_outputs(train_data, val_data, test_data,
                 vocab, api_to_id, id_to_api, class_weights)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print("\nNext step: run the CNN baseline model training.")

if __name__ == "__main__":
    main()
