# src/train.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import DATA_DIR, FIGURES_DIR
from src.Step4_DatasetSplitting.dataset import make_dataloaders, SleepDataset, CLASS_NAMES
from models import get_model
from trainer import train

CHECKPOINT_DIR = Path(r"../../checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")

# Raw epochs (for CNN and CNN-LSTM)
X_train = np.load(DATA_DIR / "X_train_bal.npy")   # balanced
X_val   = np.load(DATA_DIR / "X_val.npy")
X_test  = np.load(DATA_DIR / "X_test.npy")

# Feature vectors (for MLP baseline)
F_train = np.load(DATA_DIR / "F_train_bal.npy")   # balanced
F_val   = np.load(DATA_DIR / "F_val.npy")
F_test  = np.load(DATA_DIR / "F_test.npy")

# Labels
y_train = np.load(DATA_DIR / "y_train_bal.npy")
y_val   = np.load(DATA_DIR / "y_val.npy")
y_test  = np.load(DATA_DIR / "y_test.npy")

# Class weights (computed on original unbalanced train set)
class_weights = np.load(DATA_DIR / "class_weights.npy")

print(f"  X_train : {X_train.shape}  (balanced)")
print(f"  X_val   : {X_val.shape}")
print(f"  X_test  : {X_test.shape}")
print(f"  Class weights: {class_weights}")


# ── DataLoaders ───────────────────────────────────────────────────────────────
from torch.utils.data import DataLoader

def make_loader(X, y, batch_size, shuffle):
    ds = SleepDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())

BATCH_SIZE = 128

# CNN loaders (raw signal)
cnn_train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
cnn_val_loader   = make_loader(X_val,   y_val,   BATCH_SIZE * 2, shuffle=False)

# MLP loaders (features)
mlp_train_loader = make_loader(F_train, y_train, BATCH_SIZE, shuffle=True)
mlp_val_loader   = make_loader(F_val,   y_val,   BATCH_SIZE * 2, shuffle=False)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

all_histories = {}

# ── Model A: Baseline MLP ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  TRAINING: Baseline MLP")
print("=" * 55)

mlp = get_model("mlp", n_features=29, n_classes=3, dropout=0.3)
print(f"Parameters: {sum(p.numel() for p in mlp.parameters()):,}")

history_mlp = train(
    model          = mlp,
    train_loader   = mlp_train_loader,
    val_loader     = mlp_val_loader,
    class_weights  = class_weights,
    checkpoint_dir = CHECKPOINT_DIR,
    model_name     = "mlp",
    lr             = 1e-3,
    n_epochs       = 80,
    patience       = 10,
)
all_histories["mlp"] = history_mlp
torch.save(mlp.state_dict(), CHECKPOINT_DIR / "mlp_final.pt")


# ── Model B: 1D CNN ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  TRAINING: 1D CNN")
print("=" * 55)

cnn = get_model("cnn", n_channels=2, n_classes=3, dropout=0.3)
print(f"Parameters: {sum(p.numel() for p in cnn.parameters()):,}")

history_cnn = train(
    model          = cnn,
    train_loader   = cnn_train_loader,
    val_loader     = cnn_val_loader,
    class_weights  = class_weights,
    checkpoint_dir = CHECKPOINT_DIR,
    model_name     = "cnn",
    lr             = 1e-3,
    n_epochs       = 100,
    patience       = 12,
)
all_histories["cnn"] = history_cnn
torch.save(cnn.state_dict(), CHECKPOINT_DIR / "cnn_final.pt")


# ── Model C: CNN + LSTM ───────────────────────────────────────────────────────
# Note: CNN-LSTM needs sequences of epochs, not single epochs.
# We build a sequence dataset where each sample is SEQ_LEN consecutive epochs.

print("\n" + "=" * 55)
print("  TRAINING: CNN + LSTM")
print("=" * 55)

SEQ_LEN = 21   # 21 epochs × 4s = 84s of context

from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    """
    Wraps epoch arrays into overlapping sequences of length SEQ_LEN.
    The label is for the center epoch of each sequence.
    Only creates sequences within the same subject to avoid
    mixing epochs from different recordings.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 subjects: np.ndarray, seq_len: int = 21):
        self.sequences = []
        self.labels    = []
        half           = seq_len // 2

        for sub_id in np.unique(subjects):
            idx = np.where(subjects == sub_id)[0]
            X_sub = X[idx]
            y_sub = y[idx]

            for i in range(half, len(X_sub) - half):
                seq = X_sub[i - half : i + half + 1]   # (seq_len, C, T)
                self.sequences.append(seq)
                self.labels.append(y_sub[i])

        self.sequences = np.stack(self.sequences).astype(np.float32)
        self.labels    = np.array(self.labels,    dtype=np.int64)

    def __len__(self):  return len(self.labels)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.sequences[idx]),
                torch.tensor(self.labels[idx]))

# Load subject IDs for sequence building
subs_train = np.load(DATA_DIR / "subjects_all.npy")
# Align subject IDs with the balanced training set
# (we stored original subject IDs — re-map to balanced set)
# For simplicity we use the unbalanced raw set for CNN-LSTM
X_train_orig = np.load(DATA_DIR / "X_train.npy")
y_train_orig = np.load(DATA_DIR / "y_train.npy")
train_subs   = np.load(DATA_DIR / "subjects_all.npy")

# Filter to training subjects only
with open(DATA_DIR / "splits_metadata.json") as f:
    split_meta   = json.load(f)
train_sub_ids = split_meta["train_subjects"]
val_sub_ids   = split_meta["val_subjects"]

all_subs = np.load(DATA_DIR / "subjects_all.npy")
all_X    = np.load(DATA_DIR / "X_all.npy")
all_y    = np.load(DATA_DIR / "y_all.npy")

tr_mask = np.isin(all_subs, train_sub_ids)
va_mask = np.isin(all_subs, val_sub_ids)

print("Building sequence datasets (may take a minute)...")
seq_train_ds = SequenceDataset(all_X[tr_mask], all_y[tr_mask],
                               all_subs[tr_mask], seq_len=SEQ_LEN)
seq_val_ds   = SequenceDataset(all_X[va_mask], all_y[va_mask],
                               all_subs[va_mask], seq_len=SEQ_LEN)

print(f"  Train sequences: {len(seq_train_ds):,}")
print(f"  Val   sequences: {len(seq_val_ds):,}")

seq_train_loader = DataLoader(seq_train_ds, batch_size=64,
                              shuffle=True,  num_workers=0)
seq_val_loader   = DataLoader(seq_val_ds,   batch_size=64,
                              shuffle=False, num_workers=0)

cnn_lstm = get_model("cnn_lstm", n_channels=2, n_classes=3,
                     seq_len=SEQ_LEN, lstm_hidden=128, lstm_layers=2)
print(f"Parameters: {sum(p.numel() for p in cnn_lstm.parameters()):,}")

history_lstm = train(
    model          = cnn_lstm,
    train_loader   = seq_train_loader,
    val_loader     = seq_val_loader,
    class_weights  = class_weights,
    checkpoint_dir = CHECKPOINT_DIR,
    model_name     = "cnn_lstm",
    lr             = 5e-4,      # lower LR for LSTM stability
    n_epochs       = 80,
    patience       = 12,
    weight_decay   = 1e-4,
)
all_histories["cnn_lstm"] = history_lstm
torch.save(cnn_lstm.state_dict(), CHECKPOINT_DIR / "cnn_lstm_final.pt")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════

model_colors = {"mlp": "#888780", "cnn": "#378ADD", "cnn_lstm": "#1D9E75"}
model_labels = {"mlp": "Baseline MLP", "cnn": "1D CNN",
                "cnn_lstm": "CNN + LSTM"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, history in all_histories.items():
    epochs = range(1, len(history["train_loss"]) + 1)
    color  = model_colors[name]
    label  = model_labels[name]

    axes[0].plot(epochs, history["val_loss"],
                 color=color, linewidth=2, label=label)
    axes[0].plot(epochs, history["train_loss"],
                 color=color, linewidth=1, linestyle="--", alpha=0.5)

    axes[1].plot(epochs, history["val_acc"],
                 color=color, linewidth=2, label=label)
    axes[1].plot(epochs, history["train_acc"],
                 color=color, linewidth=1, linestyle="--", alpha=0.5)

for ax, title, ylabel in zip(
    axes,
    ["Loss curves (solid=val, dashed=train)",
     "Accuracy curves (solid=val, dashed=train)"],
    ["Cross-entropy loss", "Accuracy"]
):
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)

plt.suptitle("Training curves — all models", fontsize=12)
plt.tight_layout()
out = FIGURES_DIR / "16_training_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out}")

# Save histories
with open(DATA_DIR / "training_histories.json", "w") as f:
    json.dump(all_histories, f, indent=2)

print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
