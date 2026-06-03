import os
import json
import zipfile
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = r"D:\ACS\Final Project\ransomware dataset"

RANSOMWARE_SOURCES = [
    "some ransomware cuckoo analysis report",
    "582个ransomware_cuckoo analysis report.zip",
]

BENIGN_SOURCES = [
    "benign sample_cuckoo analysis report.zip",
    "benign_sample_report_cuckoo analysis report_2.zip",
]

EXTRACT_DIR = os.path.join(BASE_DIR, "_extracted_temp")

# ============================================================
# HELPERS
# ============================================================

def extract_zip(zip_path, extract_to):
    print(f"  Extracting: {os.path.basename(zip_path)} ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    print(f"  Done.")

def find_json_files(folder):
    json_files = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))
    return json_files

def get_family_name(json_path, base_folder):
    """
    Infer family from folder structure.
    Handles both:
      base_folder/family/sample.json         -> family
      base_folder/wrapper/family/sample.json -> family (goes deepest meaningful folder)
    """
    rel = os.path.relpath(json_path, base_folder)
    parts = rel.split(os.sep)
    # parts[-1] is the filename, parts[:-1] are folders
    folders = parts[:-1]
    if len(folders) == 0:
        return "unknown"
    elif len(folders) == 1:
        return folders[0].lower()
    else:
        # nested: take the deepest folder as family name
        return folders[-1].lower()

def extract_api_calls(data):
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

def analyse_files(json_files, base_folder, label):
    samples = []
    all_apis = set()
    api_frequency = defaultdict(int)
    skipped = 0

    for i, jf in enumerate(json_files):
        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{len(json_files)} files...")

        try:
            with open(jf, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
        except Exception:
            skipped += 1
            continue

        api_calls = extract_api_calls(data)
        family = get_family_name(jf, base_folder) if label == "ransomware" else "benign"
        all_apis.update(api_calls)
        for api in api_calls:
            api_frequency[api] += 1

        samples.append({
            "file": os.path.basename(jf),
            "family": family,
            "label": label,
            "num_calls": len(api_calls),
        })

    return samples, all_apis, api_frequency, skipped

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("RANSOMWARE DATASET ANALYSIS")
    print("=" * 60)

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    all_samples = []
    all_apis = set()
    all_api_frequency = defaultdict(int)

    # ---- RANSOMWARE ----
    print("\n[1] Processing RANSOMWARE sources...")

    for source in RANSOMWARE_SOURCES:
        full_path = os.path.join(BASE_DIR, source)

        if source.endswith(".zip"):
            extract_to = os.path.join(EXTRACT_DIR, source.replace(".zip", "").replace("个", "_"))
            if not os.path.exists(extract_to):
                extract_zip(full_path, extract_to)
            search_folder = extract_to
        else:
            search_folder = full_path

        if not os.path.exists(search_folder):
            print(f"  WARNING: Could not find {search_folder}, skipping.")
            continue

        json_files = find_json_files(search_folder)
        print(f"  Found {len(json_files)} JSON files in: {source}")

        samples, apis, api_freq, skipped = analyse_files(
            json_files, search_folder, "ransomware"
        )
        all_samples.extend(samples)
        all_apis.update(apis)
        for k, v in api_freq.items():
            all_api_frequency[k] += v

        if skipped > 0:
            print(f"  Skipped (unreadable): {skipped}")

    # ---- BENIGN ----
    print("\n[2] Processing BENIGN sources...")

    for source in BENIGN_SOURCES:
        full_path = os.path.join(BASE_DIR, source)

        if source.endswith(".zip"):
            extract_to = os.path.join(EXTRACT_DIR, source.replace(".zip", ""))
            if not os.path.exists(extract_to):
                extract_zip(full_path, extract_to)
            search_folder = extract_to
        else:
            search_folder = full_path

        if not os.path.exists(search_folder):
            print(f"  WARNING: Could not find {search_folder}, skipping.")
            continue

        json_files = find_json_files(search_folder)
        print(f"  Found {len(json_files)} JSON files in: {source}")

        samples, apis, api_freq, skipped = analyse_files(
            json_files, search_folder, "benign"
        )
        all_samples.extend(samples)
        all_apis.update(apis)
        for k, v in api_freq.items():
            all_api_frequency[k] += v

        if skipped > 0:
            print(f"  Skipped (unreadable): {skipped}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    ransomware_samples = [s for s in all_samples if s["label"] == "ransomware"]
    benign_samples     = [s for s in all_samples if s["label"] == "benign"]

    # Usable samples (more than 10 calls)
    usable_ransomware = [s for s in ransomware_samples if s["num_calls"] >= 10]
    usable_benign     = [s for s in benign_samples     if s["num_calls"] >= 10]

    print(f"\nTotal samples:              {len(all_samples)}")
    print(f"  Ransomware (total):       {len(ransomware_samples)}")
    print(f"  Ransomware (usable >=10): {len(usable_ransomware)}")
    print(f"  Benign (total):           {len(benign_samples)}")
    print(f"  Benign (usable >=10):     {len(usable_benign)}")
    print(f"\nUnique API calls (vocabulary size): {len(all_apis)}")

    # Per-family counts (usable only)
    family_counts_total  = defaultdict(int)
    family_counts_usable = defaultdict(int)
    for s in ransomware_samples:
        family_counts_total[s["family"]] += 1
    for s in usable_ransomware:
        family_counts_usable[s["family"]] += 1

    print("\n--- Ransomware samples per family (total | usable) ---")
    for fam in sorted(family_counts_total.keys()):
        total  = family_counts_total[fam]
        usable = family_counts_usable.get(fam, 0)
        print(f"  {fam:<30} total: {total:<6} usable: {usable}")

    # Sequence length stats
    def stats(samples_list, label):
        lengths = [s["num_calls"] for s in samples_list]
        if not lengths:
            print(f"\n--- {label}: no samples ---")
            return
        sorted_l = sorted(lengths)
        print(f"\n--- API call sequence length: {label} ---")
        print(f"  Count:   {len(lengths)}")
        print(f"  Min:     {min(lengths)}")
        print(f"  Max:     {max(lengths)}")
        print(f"  Average: {sum(lengths)/len(lengths):.1f}")
        print(f"  Median:  {sorted_l[len(sorted_l)//2]}")
        print(f"  Samples with 0 calls:    {sum(1 for l in lengths if l == 0)}")
        print(f"  Samples with <10 calls:  {sum(1 for l in lengths if l < 10)}")
        print(f"  Samples with >=10 calls: {sum(1 for l in lengths if l >= 10)}")

    stats(ransomware_samples, "RANSOMWARE (all)")
    stats(usable_ransomware,  "RANSOMWARE (usable >=10 calls)")
    stats(benign_samples,     "BENIGN (all)")
    stats(usable_benign,      "BENIGN (usable >=10 calls)")

    # Top 20 most common APIs
    print("\n--- Top 20 most common API calls across all samples ---")
    sorted_apis = sorted(all_api_frequency.items(), key=lambda x: -x[1])
    for api, count in sorted_apis[:20]:
        print(f"  {api:<45} {count}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
