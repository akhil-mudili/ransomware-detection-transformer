import os
import json
import zipfile
from collections import defaultdict

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


def extract_zip(zip_path, extract_to):
    print(f"  extracting {os.path.basename(zip_path)}...")
    z = zipfile.ZipFile(zip_path, 'r')
    z.extractall(extract_to)
    z.close()
    print("  done")


def find_json_files(folder):
    json_files = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))
    return json_files


def get_family_name(json_path, base_folder):
    # family = subfolder name, if nested just use the deepest one
    rel = os.path.relpath(json_path, base_folder)
    parts = rel.split(os.sep)
    folders = parts[:-1]
    if len(folders) == 0:
        return "unknown"
    if len(folders) == 1:
        return folders[0].lower()
    return folders[-1].lower()


def extract_api_calls(data):
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


def analyse_files(json_files, base_folder, label):
    samples = []
    all_apis = set()
    api_frequency = defaultdict(int)
    skipped = 0

    count = 0
    for jf in json_files:
        count += 1
        if count % 50 == 0:
            print(f"    {count}/{len(json_files)} processed...")

        try:
            f = open(jf, 'r', encoding='utf-8', errors='ignore')
            data = json.load(f)
            f.close()
        except Exception:
            skipped += 1
            continue

        api_calls = extract_api_calls(data)
        if label == "ransomware":
            family = get_family_name(jf, base_folder)
        else:
            family = "benign"

        for api in api_calls:
            all_apis.add(api)
            api_frequency[api] += 1

        samples.append({
            "file": os.path.basename(jf),
            "family": family,
            "label": label,
            "num_calls": len(api_calls),
        })

    return samples, all_apis, api_frequency, skipped


def main():
    print("RANSOMWARE DATASET ANALYSIS")

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    all_samples = []
    all_apis = set()
    all_api_frequency = defaultdict(int)

    print("\nransomware sources...")
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
            print(f"  couldn't find {search_folder}, skipping")
            continue

        json_files = find_json_files(search_folder)
        print(f"  {len(json_files)} json files in {source}")

        samples, apis, api_freq, skipped = analyse_files(json_files, search_folder, "ransomware")
        all_samples.extend(samples)
        all_apis.update(apis)
        for k in api_freq:
            all_api_frequency[k] += api_freq[k]

        if skipped > 0:
            print(f"  skipped {skipped} (couldn't read)")

    print("\nbenign sources...")
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
            print(f"  couldn't find {search_folder}, skipping")
            continue

        json_files = find_json_files(search_folder)
        print(f"  {len(json_files)} json files in {source}")

        samples, apis, api_freq, skipped = analyse_files(json_files, search_folder, "benign")
        all_samples.extend(samples)
        all_apis.update(apis)
        for k in api_freq:
            all_api_frequency[k] += api_freq[k]

        if skipped > 0:
            print(f"  skipped {skipped} (couldn't read)")

    print("\nRESULTS")

    ransomware_samples = [s for s in all_samples if s["label"] == "ransomware"]
    benign_samples = [s for s in all_samples if s["label"] == "benign"]

    usable_ransomware = [s for s in ransomware_samples if s["num_calls"] >= 10]
    usable_benign = [s for s in benign_samples if s["num_calls"] >= 10]

    print(f"\ntotal samples: {len(all_samples)}")
    print(f"  ransomware total:  {len(ransomware_samples)}")
    print(f"  ransomware usable: {len(usable_ransomware)}")
    print(f"  benign total:      {len(benign_samples)}")
    print(f"  benign usable:     {len(usable_benign)}")
    print(f"\nvocab size (unique api calls): {len(all_apis)}")

    family_counts_total = defaultdict(int)
    family_counts_usable = defaultdict(int)
    for s in ransomware_samples:
        family_counts_total[s["family"]] += 1
    for s in usable_ransomware:
        family_counts_usable[s["family"]] += 1

    print("\nransomware per family (total | usable):")
    for fam in sorted(family_counts_total.keys()):
        total = family_counts_total[fam]
        usable = family_counts_usable.get(fam, 0)
        print(f"  {fam}: total {total}, usable {usable}")

    def stats(samples_list, label):
        lengths = [s["num_calls"] for s in samples_list]
        if not lengths:
            print(f"\n{label}: no samples")
            return
        sorted_l = sorted(lengths)
        print(f"\napi call sequence length - {label}")
        print(f"  count:   {len(lengths)}")
        print(f"  min:     {min(lengths)}")
        print(f"  max:     {max(lengths)}")
        print(f"  average: {sum(lengths)/len(lengths):.1f}")
        print(f"  median:  {sorted_l[len(sorted_l)//2]}")
        print(f"  with 0 calls:    {sum(1 for l in lengths if l == 0)}")
        print(f"  with <10 calls:  {sum(1 for l in lengths if l < 10)}")
        print(f"  with >=10 calls: {sum(1 for l in lengths if l >= 10)}")

    stats(ransomware_samples, "ransomware (all)")
    stats(usable_ransomware, "ransomware (usable)")
    stats(benign_samples, "benign (all)")
    stats(usable_benign, "benign (usable)")

    print("\ntop 20 api calls overall:")
    sorted_apis = sorted(all_api_frequency.items(), key=lambda x: -x[1])
    for api, count in sorted_apis[:20]:
        print(f"  {api}: {count}")

    print("\nanalysis done")


if __name__ == "__main__":
    main()
