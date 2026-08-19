import os
import pickle
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# same seed as the transformer script, keeps everything reproducible and
# comparable across models
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BASE_DIR = r"D:\ACS\Final Project\ransomware-detection-transformer"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "results", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

f = open(os.path.join(DATA_DIR, "config.pkl"), "rb")
config = pickle.load(f)
f.close()

VOCAB_SIZE = config["vocab_size"] + 1  # +1 for the padding index
MAX_SEQ_LEN = config["max_seq_len"]
WINDOW_SIZES = config["window_sizes"]
CLASS_WEIGHTS = config["class_weights"]

# model settings, embedding dim kept at 64 to match CNN and Transformer so
# the three models are comparable and the only real difference is architecture
EMBED_DIM = 64
HIDDEN_DIM = 64
N_LSTM_LAYERS = 1
FC_DROPOUT = 0.1
LR = 0.0005
MAX_EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 16
LR_PATIENCE = 3
LR_FACTOR = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device: {DEVICE}")
print(f"vocab size: {VOCAB_SIZE}, max seq len: {MAX_SEQ_LEN}")
print(f"window sizes: {WINDOW_SIZES}")
print()


class APISequenceDataset(Dataset):
    def __init__(self, samples, window_size):
        self.sequences = []
        self.labels = []
        for s in samples:
            seq = s["windows"][window_size]
            self.sequences.append(torch.tensor(seq, dtype=torch.long))
            self.labels.append(s["label"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def make_loader(samples, window_size, shuffle=True):
    ds = APISequenceDataset(samples, window_size)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=EMBED_DIM,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=N_LSTM_LAYERS,
            batch_first=True,
            bidirectional=False  # keeping this a plain forward LSTM, not
                                  # bidirectional, to keep the comparison
                                  # with the transformer clean, a bidirectional
                                  # version would need its own justification
        )

        self.dropout = nn.Dropout(FC_DROPOUT)
        self.fc = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x, lengths):
        emb = self.embedding(x)

        # pack the padded sequence so the LSTM skips over the padding instead
        # of wasting compute (and learning garbage) on it. lengths has to be
        # on cpu for this call, torch is picky about that
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, (h_n, _) = self.lstm(packed)

        # h_n is the final hidden state, shape (num_layers, batch, hidden_dim)
        # since it's a single layer, non-bidirectional lstm, h_n[-1] is just
        # the last layer's final hidden state, this naturally only reflects
        # the real (non-padded) part of the sequence because we packed it
        final_hidden = h_n[-1]

        out = self.dropout(final_hidden)
        out = self.fc(out).squeeze(-1)
        return out


def get_lengths(x, pad_id=0):
    # figures out how many real (non-pad) tokens are in each sequence in the
    # batch, needed for pack_padded_sequence. clamped to at least 1 since a
    # fully empty sequence would break the packing
    lengths = (x != pad_id).sum(dim=1)
    lengths = torch.clamp(lengths, min=1)
    return lengths


def get_loss_fn():
    # same reasoning as the transformer script, pos_weight = w1/w0 mirrors
    # how CrossEntropyLoss(weight=[w0, w1]) treats the two classes in the CNN
    weights = torch.tensor([CLASS_WEIGHTS[0], CLASS_WEIGHTS[1]], dtype=torch.float).to(DEVICE)
    return nn.BCEWithLogitsLoss(pos_weight=weights[1] / weights[0])


def evaluate(model, loader, loss_fn, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for seqs, labels in loader:
            seqs = seqs.to(DEVICE)
            labels = labels.to(DEVICE).float()
            lengths = get_lengths(seqs)

            logits = model(seqs, lengths)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.long().cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, f1, prec, rec, cm, all_labels, all_probs


def find_best_threshold(labels, probs):
    labels = np.array(labels)
    probs = np.array(probs)

    best_threshold = 0.5
    best_f1 = -1.0

    for t in np.arange(0.1, 0.9, 0.02):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return round(float(best_threshold), 2), best_f1


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0

    for seqs, labels in loader:
        seqs = seqs.to(DEVICE)
        labels = labels.to(DEVICE).float()
        lengths = get_lengths(seqs)

        optimizer.zero_grad()
        logits = model(seqs, lengths)
        loss = loss_fn(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def train_window(train_data, val_data, test_data, window_size):
    label = int(window_size * 100)
    print(f"\nWINDOW: {label}%")

    train_loader = make_loader(train_data, window_size, shuffle=True)
    val_loader = make_loader(val_data, window_size, shuffle=False)
    test_loader = make_loader(test_data, window_size, shuffle=False)

    model = LSTMClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE
    )
    loss_fn = get_loss_fn()

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_no_imp = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_acc, val_f1, val_prec, val_rec, _, _, _ = evaluate(model, val_loader, loss_fn)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | lr={current_lr:.6f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_imp = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= PATIENCE:
                print(f"  stopping early at epoch {epoch} (best was epoch {best_epoch})")
                break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    _, _, _, _, _, _, val_labels, val_probs = evaluate(model, val_loader, loss_fn)
    best_threshold, val_f1_at_threshold = find_best_threshold(val_labels, val_probs)
    print(f"  best threshold from val set: {best_threshold} (val f1 at that threshold: {val_f1_at_threshold:.4f})")

    _, test_acc, test_f1, test_prec, test_rec, test_cm, _, _ = evaluate(
        model, test_loader, loss_fn, threshold=best_threshold
    )

    print(f"\n  test results ({label}% window):")
    print(f"    accuracy:  {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"    f1 macro:  {test_f1:.4f}  ({test_f1*100:.2f}%)")
    print(f"    precision: {test_prec:.4f}")
    print(f"    recall:    {test_rec:.4f}")
    print(f"    confusion matrix:\n{test_cm}")

    model_path = os.path.join(MODELS_DIR, f"lstm_window_{label}.pt")
    torch.save(best_state, model_path)
    print(f"  model saved to {model_path}")

    return {
        "window": label,
        "best_epoch": best_epoch,
        "decision_threshold": best_threshold,
        "accuracy": round(test_acc * 100, 2),
        "f1_macro": round(test_f1 * 100, 2),
        "precision": round(test_prec * 100, 2),
        "recall": round(test_rec * 100, 2),
        "confusion_matrix": test_cm.tolist()
    }


def main():
    print("loading data...")
    f = open(os.path.join(DATA_DIR, "train.pkl"), "rb")
    train_data = pickle.load(f)
    f.close()

    f = open(os.path.join(DATA_DIR, "val.pkl"), "rb")
    val_data = pickle.load(f)
    f.close()

    f = open(os.path.join(DATA_DIR, "test.pkl"), "rb")
    test_data = pickle.load(f)
    f.close()

    print(f"train {len(train_data)} | val {len(val_data)} | test {len(test_data)}")

    all_results = []
    for ws in WINDOW_SIZES:
        result = train_window(train_data, val_data, test_data, ws)
        all_results.append(result)

    results_path = os.path.join(RESULTS_DIR, "lstm_results.json")
    f = open(results_path, "w")
    json.dump(all_results, f, indent=2)
    f.close()
    print(f"\nall results saved to {results_path}")

    print("\nLSTM RESULTS SUMMARY")
    print(f"{'window':<10} {'accuracy':>10} {'f1 macro':>10} {'precision':>10} {'recall':>10}")
    for r in all_results:
        print(f"{str(r['window'])+'%':<10} {str(r['accuracy'])+'%':>10} {str(r['f1_macro'])+'%':>10} "
              f"{str(r['precision'])+'%':>10} {str(r['recall'])+'%':>10}")


if __name__ == "__main__":
    main()
