import os
import json
import pickle
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

# change these paths if the dataset folder moves
BASE_DIR = r"D:\ACS\Final Project\ransomware dataset"
EXTRACT_DIR = os.path.join(BASE_DIR, "_extracted_temp")
OUTPUT_DIR = r"D:\ACS\Final Project\ransomware-detection-transformer\data"

RANSOMWARE_FOLDERS = [
    os.path.join(BASE_DIR, "some ransomware cuckoo analysis report"),
    os.path.join(EXTRACT_DIR, "582_ransomware_cuckoo analysis report"),
]

BENIGN_FOLDERS = [
    os.path.join(EXTRACT_DIR, "benign sample_cuckoo analysis report"),
    os.path.join(EXTRACT_DIR, "benign_sample_report_cuckoo analysis report_2",
                 "benign_report", "benign_report"),
]

# supplementary samples pulled from the API_traces_malware_detection dataset
# (VMI hypervisor tracer, not cuckoo), built by extract_api_traces_samples.py.
# this replaces the earlier malbehavd addition, better sequence lengths and
# also brings in real ransomware samples across 7 named families, not just
# benign. benign side gets capped below so this doesn't just swap one
# imbalance for the opposite one
EXTRA_SAMPLES_JSON = os.path.join(
    r"D:\ACS\Final Project\ransomware-detection-transformer\src\models",
    "api_traces_samples.json"
)
EXTRA_BENIGN_CAP = 1300  # roughly matches how much ransomware the new data
                          # adds, keeps the final ratio close to even instead
                          # of skewing hard toward benign like last time

MIN_SEQ_LEN = 10        # drop samples with less api calls than this, basically useless
MAX_SEQ_LEN = 3000       # cut off anything longer than this
WINDOW_SIZES = [0.25, 0.50, 0.75, 1.00]

# these two families need to stay completely out of train/val/test, they're
# used later as "unseen" families for the live detection test, so if they
# leak into training here the whole known-vs-unseen comparison is broken
EXCLUDED_FAMILIES = ["wannacry", "sodinokibi"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def find_json_files(folder):
    # just walks every subfolder and grabs the json files
    json_files = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))
    return json_files


def get_family_name(json_path, base_folder):
    # family name = whatever subfolder the file is sitting in
    # some folders are nested so just take the last one
    rel = os.path.relpath(json_path, base_folder)
    parts = rel.split(os.sep)
    folders = parts[:-1]
    if len(folders) == 0:
        return "unknown"
    if len(folders) == 1:
        return folders[0].lower()
    return folders[-1].lower()


def extract_api_calls(data):
    # cuckoo json structure is behavior -> processes -> calls -> api
    api_calls = []
    try:
        processes = data.get("behavior", {}).get("processes", [])
        for proc in processes:
            for call in proc.get("calls", []):
                api = call.get("api", None)
                if api:
                    api_calls.append(api)
    except Exception:
        pass
    return api_calls


def apply_window(sequence, window_fraction, max_len):
    # only keep the first X% of the sequence, this is the partial observation part
    window_len = max(1, int(len(sequence) * window_fraction))
    windowed = sequence[:window_len]
    windowed = windowed[:max_len]
    return windowed


def pad_sequence(sequence, max_len, pad_id):
    # pad up to max_len so every sample is the same size for the model
    padded = sequence + [pad_id] * (max_len - len(sequence))
    return padded[:max_len]


def remove_duplicate_files(all_samples):
    # found out the hard way that some samples show up in more than one source
    # folder, probably because the dataset was put together from a couple of
    # different releases that reused some of the same malware samples.
    # if I don't catch this here, the same file can end up in both train and
    # test after the split, which is data leakage, model gets an easy answer
    # on "unseen" samples it actually already saw during training
    print("\nchecking for duplicate files across source folders")

    seen_files = set()
    deduped = []
    dupes_removed = 0

    for s in all_samples:
        if s["file"] in seen_files:
            dupes_removed += 1
            continue
        seen_files.add(s["file"])
        deduped.append(s)

    print(f"  removed {dupes_removed} duplicate files (same filename seen more than once)")
    print(f"  {len(deduped)} samples left")

    return deduped


def load_all_samples():
    print("\nloading samples from json files")

    all_samples = []

    print("ransomware samples first...")
    for folder in RANSOMWARE_FOLDERS:
        if not os.path.exists(folder):
            print(f"  couldn't find {folder}, skipping")
            continue

        json_files = find_json_files(folder)
        print(f"  {len(json_files)} files in {os.path.basename(folder)}")

        count = 0
        for jf in json_files:
            count += 1
            if count % 100 == 0:
                print(f"    {count}/{len(json_files)} done")
            try:
                f = open(jf, 'r', encoding='utf-8', errors='ignore')
                data = json.load(f)
                f.close()
            except Exception:
                continue

            api_calls = extract_api_calls(data)
            family = get_family_name(jf, folder)

            if family in EXCLUDED_FAMILIES:
                continue  # keep these out entirely, they're reserved for the unseen-family live detection test

            all_samples.append({
                "api_calls": api_calls,
                "label": 1,  # 1 = ransomware
                "family": family,
                "file": os.path.basename(jf),
            })

    print("now benign samples...")
    for folder in BENIGN_FOLDERS:
        if not os.path.exists(folder):
            print(f"  couldn't find {folder}, skipping")
            continue

        json_files = find_json_files(folder)
        print(f"  {len(json_files)} files in {os.path.basename(folder)}")

        count = 0
        for jf in json_files:
            count += 1
            if count % 100 == 0:
                print(f"    {count}/{len(json_files)} done")
            try:
                f = open(jf, 'r', encoding='utf-8', errors='ignore')
                data = json.load(f)
                f.close()
            except Exception:
                continue

            api_calls = extract_api_calls(data)
            all_samples.append({
                "api_calls": api_calls,
                "label": 0,  # 0 = benign
                "family": "benign",
                "file": os.path.basename(jf),
            })

    print("also pulling in the supplementary samples from API_traces_malware_detection...")
    if os.path.exists(EXTRA_SAMPLES_JSON):
        f = open(EXTRA_SAMPLES_JSON, 'r')
        extra_samples = json.load(f)
        f.close()

        # wannacry needs to stay out here too, same reason as everywhere
        # else in this script, it's reserved for the unseen-family test
        extra_samples = [s for s in extra_samples if s["family"] not in EXCLUDED_FAMILIES]

        extra_ransomware = [s for s in extra_samples if s["label"] == 1]
        extra_benign = [s for s in extra_samples if s["label"] == 0]

        # cap the benign side so this doesn't just create the opposite
        # imbalance, random sample with a fixed seed so it's reproducible
        if len(extra_benign) > EXTRA_BENIGN_CAP:
            rng = np.random.RandomState(RANDOM_SEED)
            keep_idx = rng.choice(len(extra_benign), size=EXTRA_BENIGN_CAP, replace=False)
            extra_benign = [extra_benign[i] for i in keep_idx]

        all_samples.extend(extra_ransomware)
        all_samples.extend(extra_benign)
        print(f"  added {len(extra_ransomware)} extra ransomware samples "
              f"(wannacry excluded)")
        print(f"  added {len(extra_benign)} extra benign samples "
              f"(capped at {EXTRA_BENIGN_CAP})")
    else:
        print(f"  couldn't find {EXTRA_SAMPLES_JSON}, skipping")

    print(f"total loaded: {len(all_samples)}")
    print(f"  (wannacry and sodinokibi kept out on purpose, they're the unseen families for later)")
    ransomware_count = 0
    benign_count = 0
    for s in all_samples:
        if s["label"] == 1:
            ransomware_count += 1
        else:
            benign_count += 1
    print(f"  ransomware: {ransomware_count}, benign: {benign_count}")

    return all_samples


def filter_samples(all_samples):
    # drop the ones with too few api calls, they're not really useful for the model
    print("\nfiltering out short samples")

    usable = []
    for s in all_samples:
        if len(s["api_calls"]) >= MIN_SEQ_LEN:
            usable.append(s)

    removed = len(all_samples) - len(usable)
    print(f"  removed {removed} samples under {MIN_SEQ_LEN} api calls")
    print(f"  {len(usable)} left")

    ransomware_count = 0
    benign_count = 0
    for s in usable:
        if s["label"] == 1:
            ransomware_count += 1
        else:
            benign_count += 1
    print(f"  ransomware: {ransomware_count}, benign: {benign_count}")

    family_counts = defaultdict(int)
    for s in usable:
        family_counts[s["family"]] += 1
    print("  per family:")
    for fam in sorted(family_counts, key=lambda x: -family_counts[x]):
        print(f"    {fam}: {family_counts[fam]}")

    return usable


def build_vocabulary(samples):
    # collect every unique api call name across the whole dataset
    print("\nbuilding vocab")

    all_apis = set()
    for s in samples:
        for api in s["api_calls"]:
            all_apis.add(api)

    vocab = [PAD_TOKEN, UNK_TOKEN] + sorted(all_apis)

    api_to_id = {}
    for idx in range(len(vocab)):
        api_to_id[vocab[idx]] = idx

    id_to_api = {}
    for api in api_to_id:
        id_to_api[api_to_id[api]] = api

    print(f"  vocab size: {len(vocab)} (PAD + UNK included)")
    print(f"  PAD id: {api_to_id[PAD_TOKEN]}, UNK id: {api_to_id[UNK_TOKEN]}")

    return vocab, api_to_id, id_to_api


def encode_and_window(samples, api_to_id):
    # turn the api names into numbers and cut out the 4 windows for each sample
    print("\nencoding + windowing")

    pad_id = api_to_id[PAD_TOKEN]
    unk_id = api_to_id[UNK_TOKEN]

    processed = []
    seq_lengths = []

    for s in samples:
        full_seq = []
        for api in s["api_calls"]:
            if api in api_to_id:
                full_seq.append(api_to_id[api])
            else:
                full_seq.append(unk_id)

        full_seq = full_seq[:MAX_SEQ_LEN]
        seq_lengths.append(len(full_seq))

        windows = {}
        for w in WINDOW_SIZES:
            windowed = apply_window(full_seq, w, MAX_SEQ_LEN)
            windows[w] = pad_sequence(windowed, MAX_SEQ_LEN, pad_id)

        processed.append({
            "label": s["label"],
            "family": s["family"],
            "file": s["file"],
            "seq_len": len(full_seq),
            "windows": windows,
        })

    seq_lengths = np.array(seq_lengths)
    print(f"  seq lengths after capping at {MAX_SEQ_LEN}:")
    print(f"    min {seq_lengths.min()}, max {seq_lengths.max()}, mean {seq_lengths.mean():.1f}, median {np.median(seq_lengths):.1f}")
    capped = np.sum(seq_lengths == MAX_SEQ_LEN)
    print(f"    {capped} samples got capped")

    return processed


def split_dataset(processed):
    # standard 70/15/15, stratified so the ransomware/benign ratio stays the same in each split
    print("\nsplitting train/val/test")

    labels = [s["label"] for s in processed]

    train_data, temp_data, train_labels, temp_labels = train_test_split(
        processed, labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=labels,
        random_state=RANDOM_SEED
    )

    val_share = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels,
        test_size=(1 - val_share),
        stratify=temp_labels,
        random_state=RANDOM_SEED
    )

    def print_split(data, name):
        ransomware = sum(1 for s in data if s["label"] == 1)
        benign = sum(1 for s in data if s["label"] == 0)
        print(f"  {name}: {len(data)} (ransomware {ransomware}, benign {benign})")

    print_split(train_data, "train")
    print_split(val_data, "val")
    print_split(test_data, "test")

    return train_data, val_data, test_data


def compute_class_weights(train_data):
    # dataset is imbalanced (more ransomware than benign) so weight the loss instead of undersampling
    print("\nworking out class weights")

    n_total = len(train_data)
    n_ransomware = sum(1 for s in train_data if s["label"] == 1)
    n_benign = sum(1 for s in train_data if s["label"] == 0)

    weight_ransomware = n_total / (2 * n_ransomware)
    weight_benign = n_total / (2 * n_benign)

    class_weights = {0: weight_benign, 1: weight_ransomware}

    print(f"  ransomware weight {weight_ransomware:.4f} ({n_ransomware} samples)")
    print(f"  benign weight {weight_benign:.4f} ({n_benign} samples)")

    return class_weights


def save_outputs(train_data, val_data, test_data, vocab, api_to_id, id_to_api, class_weights):
    print("\nsaving everything")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    splits = [("train", train_data), ("val", val_data), ("test", test_data)]
    for name, data in splits:
        path = os.path.join(OUTPUT_DIR, f"{name}.pkl")
        f = open(path, 'wb')
        pickle.dump(data, f)
        f.close()
        print(f"  {name}.pkl saved, {len(data)} samples")

    vocab_data = {
        "vocab": vocab,
        "api_to_id": api_to_id,
        "id_to_api": id_to_api,
        "vocab_size": len(vocab),
        "pad_id": api_to_id[PAD_TOKEN],
        "unk_id": api_to_id[UNK_TOKEN],
    }
    f = open(os.path.join(OUTPUT_DIR, "vocabulary.pkl"), 'wb')
    pickle.dump(vocab_data, f)
    f.close()
    print(f"  vocabulary.pkl saved, size {len(vocab)}")

    f = open(os.path.join(OUTPUT_DIR, "class_weights.pkl"), 'wb')
    pickle.dump(class_weights, f)
    f.close()
    print("  class_weights.pkl saved")

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
    f = open(os.path.join(OUTPUT_DIR, "config.pkl"), 'wb')
    pickle.dump(config, f)
    f.close()
    print("  config.pkl saved")

    print(f"\neverything saved to {OUTPUT_DIR}")


def main():
    print("running preprocessing")

    all_samples = load_all_samples()
    all_samples = remove_duplicate_files(all_samples)
    usable_samples = filter_samples(all_samples)
    vocab, api_to_id, id_to_api = build_vocabulary(usable_samples)
    processed = encode_and_window(usable_samples, api_to_id)
    train_data, val_data, test_data = split_dataset(processed)
    class_weights = compute_class_weights(train_data)
    save_outputs(train_data, val_data, test_data, vocab, api_to_id, id_to_api, class_weights)

    print("\ndone, next step is running cnn_baseline.py")


if __name__ == "__main__":
    main()
