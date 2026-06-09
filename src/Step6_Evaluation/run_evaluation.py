# src/run_evaluation.py
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))  # project root
sys.path.insert(0, str(_HERE))                # local imports (evaluate.py)

import numpy as np
import torch
import json
from torch.utils.data import DataLoader

from src.config import DATA_DIR, FIGURES_DIR, CHECKPOINT_DIR
from src.Step4_DatasetSplitting.dataset import SleepDataset
from src.Step5_Model.models import get_model
from evaluate import (
    get_predictions, compute_metrics, print_metrics,
    plot_confusion_matrix, plot_roc_curves,
    plot_per_subject_accuracy, plot_predicted_hypnogram,
    plot_model_comparison, plot_rem_analysis,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on: {DEVICE}\n")

# ── Load test data ─────────────────────────────────────────────────────────────
X_test  = np.load(DATA_DIR / "X_test.npy")
F_test  = np.load(DATA_DIR / "F_test.npy")
y_test  = np.load(DATA_DIR / "y_test.npy")

print(f"Test set: {len(y_test):,} epochs")
print(f"  Wake : {(y_test==0).sum():,}")
print(f"  NREM : {(y_test==1).sum():,}")
print(f"  REM  : {(y_test==2).sum():,}\n")

# Raw epoch loader (CNN / CNN-LSTM)
raw_loader = DataLoader(
    SleepDataset(X_test, y_test),
    batch_size=256, shuffle=False, num_workers=0
)
# Feature loader (MLP)
feat_loader = DataLoader(
    SleepDataset(F_test, y_test),
    batch_size=256, shuffle=False, num_workers=0
)

# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATE EACH MODEL
# ══════════════════════════════════════════════════════════════════════════════

all_metrics = {}

# ── MLP ───────────────────────────────────────────────────────────────────────
print("Loading MLP...")
n_features = F_test.shape[1]   # auto-detects 29 or 33 from data
mlp = get_model("mlp", n_features=n_features, n_classes=3, dropout=0.3).to(DEVICE)
mlp.load_state_dict(
    torch.load(CHECKPOINT_DIR / "mlp_best.pt",
               map_location=DEVICE, weights_only=True)
)

y_true, y_pred, y_proba = get_predictions(mlp, feat_loader, DEVICE)
metrics_mlp = compute_metrics(y_true, y_pred, y_proba)
print_metrics(metrics_mlp, "Baseline MLP")
all_metrics["mlp"] = metrics_mlp

plot_confusion_matrix(y_true, y_pred, "MLP")
plot_roc_curves(y_true, y_proba, "MLP")
plot_per_subject_accuracy(mlp, DEVICE, "MLP", use_features=True)
plot_predicted_hypnogram(mlp, DEVICE, "MLP", use_features=True)


# ── CNN ───────────────────────────────────────────────────────────────────────
print("\nLoading CNN...")
cnn = get_model("cnn", n_channels=2, n_classes=3, dropout=0.3).to(DEVICE)
cnn.load_state_dict(
    torch.load(CHECKPOINT_DIR / "cnn_best.pt",
               map_location=DEVICE, weights_only=True)
)

y_true, y_pred, y_proba = get_predictions(cnn, raw_loader, DEVICE)
metrics_cnn = compute_metrics(y_true, y_pred, y_proba)
print_metrics(metrics_cnn, "1D CNN")
all_metrics["cnn"] = metrics_cnn

plot_confusion_matrix(y_true, y_pred, "CNN")
plot_roc_curves(y_true, y_proba, "CNN")
plot_per_subject_accuracy(cnn, DEVICE, "CNN")
plot_predicted_hypnogram(cnn, DEVICE, "CNN")


# ── CNN + LSTM ────────────────────────────────────────────────────────────────
print("\nLoading CNN + LSTM...")

SEQ_LEN  = 31
all_X    = np.load(DATA_DIR / "X_all.npy")
all_y    = np.load(DATA_DIR / "y_all.npy")
all_subs = np.load(DATA_DIR / "subjects_all.npy")

with open(DATA_DIR / "splits_metadata.json") as f:
    split_meta = json.load(f)
test_sub_ids = split_meta["test_subjects"]

te_mask = np.isin(all_subs, test_sub_ids)

# Rebuild sequence dataset for test subjects
from src.Step4_DatasetSplitting.dataset import SequenceDataset
seq_test_ds = SequenceDataset(
    all_X[te_mask], all_y[te_mask],
    all_subs[te_mask], seq_len=SEQ_LEN
)
seq_test_loader = DataLoader(
    seq_test_ds, batch_size=64, shuffle=False, num_workers=0
)

cnn_lstm = get_model(
    "cnn_lstm", n_channels=2, n_classes=3,
    seq_len=SEQ_LEN, lstm_hidden=256, lstm_layers=2
).to(DEVICE)
cnn_lstm.load_state_dict(
    torch.load(CHECKPOINT_DIR / "cnn_lstm_best.pt",
               map_location=DEVICE, weights_only=True)
)

y_true, y_pred, y_proba = get_predictions(cnn_lstm, seq_test_loader, DEVICE)
metrics_lstm = compute_metrics(y_true, y_pred, y_proba)
print_metrics(metrics_lstm, "CNN + LSTM")
all_metrics["cnn_lstm"] = metrics_lstm

plot_confusion_matrix(y_true, y_pred, "CNN_LSTM")
plot_roc_curves(y_true, y_proba, "CNN_LSTM")
plot_per_subject_accuracy(cnn_lstm, DEVICE, "CNN_LSTM", use_sequences=True, seq_len=SEQ_LEN)
plot_predicted_hypnogram(cnn_lstm, DEVICE, "CNN_LSTM", use_sequences=True, seq_len=SEQ_LEN)


# ── Transformer ───────────────────────────────────────────────────────────────
print("\nLoading Transformer...")
transformer = get_model(
    "transformer", n_channels=2, n_classes=3,
    seq_len=SEQ_LEN, d_model=256, nhead=8,
    num_layers=4, dim_feedforward=512, dropout=0.1,
).to(DEVICE)
transformer.load_state_dict(
    torch.load(CHECKPOINT_DIR / "transformer_best.pt",
               map_location=DEVICE, weights_only=True)
)

y_true, y_pred, y_proba = get_predictions(transformer, seq_test_loader, DEVICE)
metrics_transformer = compute_metrics(y_true, y_pred, y_proba)
print_metrics(metrics_transformer, "Transformer")
all_metrics["transformer"] = metrics_transformer

plot_confusion_matrix(y_true, y_pred, "Transformer")
plot_roc_curves(y_true, y_proba, "Transformer")
plot_per_subject_accuracy(transformer, DEVICE, "Transformer",
                          use_sequences=True, seq_len=SEQ_LEN)
plot_predicted_hypnogram(transformer, DEVICE, "Transformer",
                         use_sequences=True, seq_len=SEQ_LEN)


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARISON PLOTS
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating comparison plots...")
plot_model_comparison(all_metrics)
plot_rem_analysis(all_metrics)


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

with open(DATA_DIR / "evaluation_results.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
print(f"\nSaved -> {DATA_DIR / 'evaluation_results.json'}")


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"  FINAL RESULTS SUMMARY")
print(f"{'='*65}")
print(f"  {'Model':<18} {'Acc':>7} {'Bal Acc':>9} "
      f"{'F1':>7} {'Kappa':>8} {'AUC':>7}")
print(f"  {'-'*60}")

model_labels = {
    "mlp":         "Baseline MLP",
    "cnn":         "1D CNN",
    "cnn_lstm":    "CNN + LSTM",
    "transformer": "Transformer",
}
for name, m in all_metrics.items():
    print(
        f"  {model_labels[name]:<18} "
        f"{m['accuracy']:>7.4f} "
        f"{m['balanced_accuracy']:>9.4f} "
        f"{m['macro_f1']:>7.4f} "
        f"{m['cohen_kappa']:>8.4f} "
        f"{m['auc_roc']:>7.4f}"
    )

print(f"\n  REM F1 (hardest class):")
for name, m in all_metrics.items():
    rem_f1 = m["per_class"]["REM"]["f1"]
    rem_rec = m["per_class"]["REM"]["recall"]
    print(f"  {model_labels[name]:<18} F1={rem_f1:.4f}  "
          f"Recall={rem_rec:.4f}")

print(f"\nAll figures saved to: {FIGURES_DIR}")
print("\nProject complete.")