import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             classification_report, confusion_matrix)

DATA_DIR = r"D:\ACS\Final Project\ransomware-detection-transformer\data"
RESULTS_DIR = r"D:\ACS\Final Project\ransomware-detection-transformer\results"
MODEL_DIR = r"D:\ACS\Final Project\ransomware-detection-transformer\results\models"

# random forest settings, nothing fancy, just defaults that seemed reasonable
N_ESTIMATORS = 200
MAX_DEPTH = None  # let it grow, RF handles overfitting fine on its own with enough trees
MIN_SAMPLES_LEAF = 2
RANDOM_SEED = 42

WINDOW_SIZES = [0.25, 0.50, 0.75, 1.00]
WINDOW_NAMES = {0.25: "25%", 0.50: "50%", 0.75: "75%", 1.00: "100%"}

np.random.seed(RANDOM_SEED)


def sequence_to_freq_vector(seq, vocab_size, pad_id):
    # this is the whole point of the RF baseline, no order info at all
    # just count how many times each api call shows up and turn that into
    # a fraction of the total calls, so a longer sequence doesn't just look
    # "busier" than a short one for no reason
    seq = np.array(seq)
    real_calls = seq[seq != pad_id]  # drop the padding, only count real calls

    counts = np.zeros(vocab_size, dtype=np.float64)
    if len(real_calls) > 0:
        ids, freqs = np.unique(real_calls, return_counts=True)
        counts[ids] = freqs
        counts = counts / len(real_calls)  # now it's proportions not raw counts

    return counts


def build_feature_matrix(data, window_size, vocab_size, pad_id):
    # same idea as the dataset classes in the other two scripts, just building
    # a plain numpy array instead of a torch Dataset since sklearn wants that
    X = np.zeros((len(data), vocab_size), dtype=np.float64)
    y = np.zeros(len(data), dtype=np.int64)

    for i, sample in enumerate(data):
        seq = sample["windows"][window_size]
        X[i] = sequence_to_freq_vector(seq, vocab_size, pad_id)
        y[i] = sample["label"]

    return X, y


def train_and_evaluate_window(window_size, train_data, val_data, test_data,
                               vocab_data, class_weights):
    window_name = WINDOW_NAMES[window_size]
    print(f"\ntraining Random Forest, window {window_name}")

    vocab_size = vocab_data["vocab_size"]
    pad_id = vocab_data["pad_id"]

    X_train, y_train = build_feature_matrix(train_data, window_size, vocab_size, pad_id)
    X_val, y_val = build_feature_matrix(val_data, window_size, vocab_size, pad_id)
    X_test, y_test = build_feature_matrix(test_data, window_size, vocab_size, pad_id)

    print(f"  train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

    # sklearn wants a plain dict for class_weight, same numbers we already
    # computed in preprocessing, just reusing them here
    sk_class_weight = {0: class_weights[0], 1: class_weights[1]}

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight=sk_class_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1  # use all cpu cores, this trains fast anyway
    )

    model.fit(X_train, y_train)

    # no epochs or early stopping here since RF doesn't train iteratively
    # like the neural nets do, so the val set isn't really doing anything
    # except letting me sanity check the numbers look reasonable
    val_preds = model.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
    print(f"  val f1 (macro): {val_f1:.4f}")

    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)
    test_prec = precision_score(y_test, test_preds, average='macro', zero_division=0)
    test_rec = recall_score(y_test, test_preds, average='macro', zero_division=0)

    print(f"\n  test results, window {window_name}")
    print(f"  accuracy:  {test_acc:.4f}")
    print(f"  f1 (macro):{test_f1:.4f}")
    print(f"  precision: {test_prec:.4f}")
    print(f"  recall:    {test_rec:.4f}")
    print("\n  classification report:")
    print(classification_report(y_test, test_preds, target_names=["Benign", "Ransomware"], zero_division=0))

    # saving the model too even though it's not strictly needed for the paper,
    # figured better to have it just in case
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"rf_window_{int(window_size*100)}.pkl")
    f = open(model_path, 'wb')
    pickle.dump(model, f)
    f.close()

    return {
        "window": window_size,
        "window_name": window_name,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_preds": test_preds.tolist(),
        "test_labels": y_test.tolist(),
        "confusion_matrix": confusion_matrix(y_test, test_preds).tolist(),
    }


def main():
    print("RANDOM FOREST BASELINE - RANSOMWARE DETECTION")
    print("(bag of counts features, no sequence order, normalized by length)")

    print("\nloading processed data...")
    f = open(os.path.join(DATA_DIR, "train.pkl"), 'rb')
    train_data = pickle.load(f)
    f.close()

    f = open(os.path.join(DATA_DIR, "val.pkl"), 'rb')
    val_data = pickle.load(f)
    f.close()

    f = open(os.path.join(DATA_DIR, "test.pkl"), 'rb')
    test_data = pickle.load(f)
    f.close()

    f = open(os.path.join(DATA_DIR, "vocabulary.pkl"), 'rb')
    vocab_data = pickle.load(f)
    f.close()

    f = open(os.path.join(DATA_DIR, "class_weights.pkl"), 'rb')
    class_weights = pickle.load(f)
    f.close()

    print(f"  train {len(train_data)} | val {len(val_data)} | test {len(test_data)}")
    print(f"  vocab size: {vocab_data['vocab_size']}")
    print(f"  class weights: {class_weights}")

    all_results = []
    for window_size in WINDOW_SIZES:
        result = train_and_evaluate_window(window_size, train_data, val_data, test_data, vocab_data, class_weights)
        all_results.append(result)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "rf_results.pkl")
    f = open(results_path, 'wb')
    pickle.dump(all_results, f)
    f.close()

    print("\nRANDOM FOREST - FINAL SUMMARY")
    print(f"{'window':<10} {'accuracy':>10} {'f1':>10} {'precision':>10} {'recall':>10}")
    for r in all_results:
        print(f"{r['window_name']:<10} {r['test_accuracy']:>10.4f} {r['test_f1']:>10.4f} "
              f"{r['test_precision']:>10.4f} {r['test_recall']:>10.4f}")

    print(f"\nresults saved to {results_path}")
    print("next step: run lstm_model.py")


if __name__ == "__main__":
    main()

