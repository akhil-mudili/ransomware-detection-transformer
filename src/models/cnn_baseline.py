import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             classification_report, confusion_matrix)

DATA_DIR = r"D:\ACS\Final Project\ransomware-detection-transformer\data"
RESULTS_DIR = r"D:\ACS\Final Project\ransomware-detection-transformer\results"
MODEL_DIR = r"D:\ACS\Final Project\ransomware-detection-transformer\results\models"

# model settings
EMBED_DIM = 64
NUM_FILTERS = 128
KERNEL_SIZE = 5
DROPOUT = 0.3
FC_DIM = 128

# training settings
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
PATIENCE = 7  # stop early if val f1 doesn't improve for this many epochs

WINDOW_SIZES = [0.25, 0.50, 0.75, 1.00]
WINDOW_NAMES = {0.25: "25%", 0.50: "50%", 0.75: "75%", 1.00: "100%"}

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {DEVICE}")


class RansomwareDataset(Dataset):
    def __init__(self, data, window_size):
        self.samples = data
        self.window_size = window_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sequence = torch.tensor(sample["windows"][self.window_size], dtype=torch.long)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return sequence, label


class CNNBaseline(nn.Module):
    # basic conv1d classifier, loosely based on RGV2 but adapted for
    # api call sequences instead of file i/o logs
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

        # two maxpools with kernel 2 means the sequence length gets divided by 4
        conv_out_len = max_seq_len // 4
        self.flat_size = (num_filters * 2) * conv_out_len

        self.fc1 = nn.Linear(self.flat_size, fc_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_dim, 2)  # 2 classes: benign or ransomware

        self.relu = nn.ReLU()

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


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, f1


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, acc, f1, prec, rec, all_preds, all_labels


def train_and_evaluate_window(window_size, train_data, val_data, test_data,
                               vocab_data, class_weights):
    window_name = WINDOW_NAMES[window_size]
    print(f"\ntraining CNN, window {window_name}")

    vocab_size = vocab_data["vocab_size"]
    max_seq_len = 3000

    train_ds = RansomwareDataset(train_data, window_size)
    val_ds = RansomwareDataset(val_data, window_size)
    test_ds = RansomwareDataset(test_data, window_size)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CNNBaseline(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
        fc_dim=FC_DIM,
        dropout=DROPOUT,
        max_seq_len=max_seq_len
    ).to(DEVICE)

    weights = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_f1 = 0
    patience_count = 0
    best_model_state = None
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    for epoch in range(NUM_EPOCHS):
        train_loss, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, val_f1, val_prec, val_rec, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(f"  epoch {epoch+1}/{NUM_EPOCHS} | train loss {train_loss:.4f} f1 {train_f1:.4f} | "
              f"val loss {val_loss:.4f} f1 {val_f1:.4f}")

        # early stopping: keep the best model, give up after PATIENCE epochs of no improvement
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_count = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  stopping early at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_f1, test_prec, test_rec, test_preds, test_labels = evaluate(
        model, test_loader, criterion, DEVICE
    )

    print(f"\n  test results, window {window_name}")
    print(f"  accuracy:  {test_acc:.4f}")
    print(f"  f1 (macro):{test_f1:.4f}")
    print(f"  precision: {test_prec:.4f}")
    print(f"  recall:    {test_rec:.4f}")
    print("\n  classification report:")
    print(classification_report(test_labels, test_preds, target_names=["Benign", "Ransomware"], zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"cnn_window_{int(window_size*100)}.pt")
    torch.save(best_model_state, model_path)

    return {
        "window": window_size,
        "window_name": window_name,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_preds": test_preds,
        "test_labels": test_labels,
        "history": history,
        "confusion_matrix": confusion_matrix(test_labels, test_preds).tolist(),
    }


def main():
    print("CNN BASELINE - RANSOMWARE DETECTION")

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
    results_path = os.path.join(RESULTS_DIR, "cnn_results.pkl")
    f = open(results_path, 'wb')
    pickle.dump(all_results, f)
    f.close()

    print("\nCNN BASELINE - FINAL SUMMARY")
    print(f"{'window':<10} {'accuracy':>10} {'f1':>10} {'precision':>10} {'recall':>10}")
    for r in all_results:
        print(f"{r['window_name']:<10} {r['test_accuracy']:>10.4f} {r['test_f1']:>10.4f} "
              f"{r['test_precision']:>10.4f} {r['test_recall']:>10.4f}")

    print(f"\nresults saved to {results_path}")
    print("next step: run transformer_model.py")


if __name__ == "__main__":
    main()
