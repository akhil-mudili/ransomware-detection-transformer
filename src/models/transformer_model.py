import os
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

BASE_DIR = r"D:\ACS\Final Project\ransomware-detection-transformer"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "src", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

f = open(os.path.join(DATA_DIR, "config.pkl"), "rb")
config = pickle.load(f)
f.close()

VOCAB_SIZE = config["vocab_size"] + 1  # +1 for the padding index
MAX_SEQ_LEN = config["max_seq_len"]     # 3000
WINDOW_SIZES = config["window_sizes"]   # [0.25, 0.5, 0.75, 1.0]
CLASS_WEIGHTS = config["class_weights"]  # {0: 2.45, 1: 0.63}

# model settings
EMBED_DIM = 64
N_HEADS = 4
N_LAYERS = 2
FF_DIM = 256
DROPOUT = 0.1
FC_DROPOUT = 0.1
LR = 0.0005
MAX_EPOCHS = 50
PATIENCE = 7
BATCH_SIZE = 32
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


class PositionalEncoding(nn.Module):
    # standard sinusoidal positional encoding, same as the original transformer paper
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

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
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)

        self.dropout = nn.Dropout(FC_DROPOUT)
        self.fc = nn.Linear(EMBED_DIM, 1)

    def forward(self, x):
        pad_mask = (x == 0)  # true wherever there's padding

        emb = self.embedding(x)
        emb = self.pos_encoding(emb)

        enc = self.transformer_encoder(emb, src_key_padding_mask=pad_mask)

        # masked average pooling, ignore the padded positions
        mask_expanded = (~pad_mask).unsqueeze(-1).float()
        pooled = (enc * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        out = self.dropout(pooled)
        out = self.fc(out).squeeze(-1)
        return out


def get_loss_fn():
    weights = torch.tensor([CLASS_WEIGHTS[0], CLASS_WEIGHTS[1]], dtype=torch.float).to(DEVICE)
    return nn.BCEWithLogitsLoss(pos_weight=weights[1] / weights[0])


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for seqs, labels in loader:
            seqs = seqs.to(DEVICE)
            labels = labels.to(DEVICE).float()

            logits = model(seqs)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.long().cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, f1, prec, rec, cm


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0

    for seqs, labels in loader:
        seqs = seqs.to(DEVICE)
        labels = labels.to(DEVICE).float()

        optimizer.zero_grad()
        logits = model(seqs)
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

    model = TransformerClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = get_loss_fn()

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_no_imp = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_acc, val_f1, val_prec, val_rec, _ = evaluate(model, val_loader, loss_fn)

        print(f"  epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

        # keep the best model so far, stop if it hasn't improved in PATIENCE epochs
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
    _, test_acc, test_f1, test_prec, test_rec, test_cm = evaluate(model, test_loader, loss_fn)

    print(f"\n  test results ({label}% window):")
    print(f"    accuracy:  {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"    f1 macro:  {test_f1:.4f}  ({test_f1*100:.2f}%)")
    print(f"    precision: {test_prec:.4f}")
    print(f"    recall:    {test_rec:.4f}")
    print(f"    confusion matrix:\n{test_cm}")

    model_path = os.path.join(MODELS_DIR, f"transformer_window_{label}.pt")
    torch.save(best_state, model_path)
    print(f"  model saved to {model_path}")

    return {
        "window": label,
        "best_epoch": best_epoch,
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

    results_path = os.path.join(RESULTS_DIR, "transformer_results.json")
    f = open(results_path, "w")
    json.dump(all_results, f, indent=2)
    f.close()
    print(f"\nall results saved to {results_path}")

    print("\nTRANSFORMER RESULTS SUMMARY")
    print(f"{'window':<10} {'accuracy':>10} {'f1 macro':>10} {'precision':>10} {'recall':>10}")
    for r in all_results:
        print(f"{str(r['window'])+'%':<10} {str(r['accuracy'])+'%':>10} {str(r['f1_macro'])+'%':>10} "
              f"{str(r['precision'])+'%':>10} {str(r['recall'])+'%':>10}")

    print("\nCNN baseline for comparison:")
    print(f"{'window':<10} {'accuracy':>10} {'f1 macro':>10}")
    cnn = [
        ("25%", "98.14%", "97.11%"),
        ("50%", "92.55%", "89.47%"),
        ("75%", "95.03%", "92.54%"),
        ("100%", "90.68%", "87.20%"),
    ]
    for w, acc, f1 in cnn:
        print(f"{w:<10} {acc:>10} {f1:>10}")


if __name__ == "__main__":
    main()
