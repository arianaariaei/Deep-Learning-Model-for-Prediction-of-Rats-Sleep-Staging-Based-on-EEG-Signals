# src/evaluate.py
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))  # project root
sys.path.insert(0, str(_HERE))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, cohen_kappa_score, roc_auc_score,
    balanced_accuracy_score
)
from src.config import DATA_DIR, FIGURES_DIR

CLASS_NAMES  = ["Wake", "NREM", "REM"]
CLASS_COLORS = ["#378ADD", "#1D9E75", "#EF9F27"]


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_predictions(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model on all batches in loader.
    Returns:
        y_true  : (N,)    ground truth labels
        y_pred  : (N,)    predicted labels
        y_proba : (N, 3)  softmax probabilities
    """
    model.eval()
    all_true, all_pred, all_proba = [], [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        logits  = model(X_batch)
        proba   = torch.softmax(logits, dim=1).cpu().numpy()
        pred    = proba.argmax(axis=1)

        all_true.append(y_batch.numpy())
        all_pred.append(pred)
        all_proba.append(proba)

    return (np.concatenate(all_true),
            np.concatenate(all_pred),
            np.concatenate(all_proba))


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """
    Compute a comprehensive set of evaluation metrics.

    Metrics explained:
      Accuracy         — overall correct / total (misleading when imbalanced)
      Balanced accuracy — mean recall across classes (better for imbalance)
      Macro F1         — F1 averaged equally across classes (punishes poor REM)
      Cohen's Kappa    — agreement above chance (standard in sleep scoring papers)
      AUC-ROC          — area under ROC curve, one-vs-rest per class
      Per-class F1     — F1 for each of Wake / NREM / REM individually
      Per-class recall — how many of each true class were caught
      Per-class precision — of predicted class X, how many were correct
    """
    accuracy  = float((y_true == y_pred).mean())
    bal_acc   = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1  = float(f1_score(y_true, y_pred, average="macro"))
    kappa     = float(cohen_kappa_score(y_true, y_pred))

    # AUC-ROC (one-vs-rest, needs probability scores)
    try:
        auc = float(roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro"
        ))
    except ValueError:
        auc = float("nan")

    # Per-class metrics
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for cls in CLASS_NAMES:
        per_class[cls] = {
            "precision": report[cls]["precision"],
            "recall":    report[cls]["recall"],
            "f1":        report[cls]["f1-score"],
            "support":   int(report[cls]["support"]),
        }

    return {
        "accuracy":          accuracy,
        "balanced_accuracy": bal_acc,
        "macro_f1":          macro_f1,
        "cohen_kappa":       kappa,
        "auc_roc":           auc,
        "per_class":         per_class,
    }


def print_metrics(metrics: dict, model_name: str):
    print(f"\n{'='*55}")
    print(f"  RESULTS: {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1          : {metrics['macro_f1']:.4f}")
    print(f"  Cohen's Kappa     : {metrics['cohen_kappa']:.4f}")
    print(f"  AUC-ROC (macro)   : {metrics['auc_roc']:.4f}")
    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':<8} {'Precision':<12} {'Recall':<12} "
          f"{'F1':<12} {'Support'}")
    print(f"  {'-'*52}")
    for cls in CLASS_NAMES:
        m = metrics["per_class"][cls]
        print(f"  {cls:<8} {m['precision']:<12.4f} {m['recall']:<12.4f} "
              f"{m['f1']:<12.4f} {m['support']:,}")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_confusion_matrix(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    model_name: str,
    normalize:  bool = True,
):
    """
    Plot confusion matrix.
    Normalized version shows recall per class (row sums to 1).
    Raw version shows epoch counts.
    Both are saved.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, norm, suffix in zip(
        axes,
        [True, False],
        ["normalized", "counts"]
    ):
        cm = confusion_matrix(y_true, y_pred)
        if norm:
            cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt, vmax  = ".2f", 1.0
            title      = f"{model_name} — Confusion matrix (row-normalized)"
        else:
            cm_display = cm
            fmt, vmax  = "d", None
            title      = f"{model_name} — Confusion matrix (counts)"

        im = ax.imshow(cm_display, interpolation="nearest",
                       cmap="Blues", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046)

        ticks = range(len(CLASS_NAMES))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(CLASS_NAMES, fontsize=11)
        ax.set_yticklabels(CLASS_NAMES, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True",      fontsize=11)
        ax.set_title(title,        fontsize=11)

        # Annotate cells
        thresh = cm_display.max() / 2.0
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                val  = cm_display[i, j]
                text = f"{val:{fmt}}" if not norm else f"{val:.2f}\n({cm[i,j]:,})"
                ax.text(j, i, text,
                        ha="center", va="center", fontsize=9,
                        color="white" if val > thresh else "black")

    plt.tight_layout()
    out = FIGURES_DIR / f"17_confusion_matrix_{model_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


def plot_roc_curves(
    y_true:     np.ndarray,
    y_proba:    np.ndarray,
    model_name: str,
):
    """Plot one-vs-rest ROC curve per class."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(8, 6))

    for cls_idx, (cls_name, color) in enumerate(
        zip(CLASS_NAMES, CLASS_COLORS)
    ):
        fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], y_proba[:, cls_idx])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{cls_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — ROC curves (one-vs-rest)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = FIGURES_DIR / f"18_roc_curves_{model_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


def plot_per_subject_accuracy(
    model:         nn.Module,
    device:        torch.device,
    model_name:    str,
    use_sequences: bool = False,
    use_features:  bool = False,
    seq_len:       int  = 31,
):
    """
    Evaluate the model separately on each test subject.
    Shows how well the model generalizes to unseen mice.
    """
    X_all  = np.load(DATA_DIR / ("F_all.npy" if use_features else "X_all.npy"))
    y_all  = np.load(DATA_DIR / "y_all.npy")
    subs   = np.load(DATA_DIR / "subjects_all.npy")

    import json
    with open(DATA_DIR / "splits_metadata.json") as f:
        split_meta = json.load(f)
    test_subs = split_meta["test_subjects"]

    from src.Step4_DatasetSplitting.dataset import SleepDataset, SequenceDataset

    results = []
    for sub_id in sorted(test_subs):
        mask  = subs == sub_id
        X_sub = X_all[mask]
        y_sub = y_all[mask]

        if len(y_sub) == 0:
            continue

        if use_sequences:
            ds = SequenceDataset(X_sub, y_sub, subs[mask], seq_len=seq_len)
            if len(ds) == 0:
                continue
        else:
            ds = SleepDataset(X_sub, y_sub)
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        y_true, y_pred, y_proba = get_predictions(model, loader, device)

        acc      = float((y_true == y_pred).mean())
        macro_f1 = float(f1_score(y_true, y_pred,
                                  average="macro", zero_division=0))
        rem_f1   = float(f1_score(y_true, y_pred,
                                  labels=[2], average="macro",
                                  zero_division=0))
        results.append({
            "subject": sub_id,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "rem_f1":   rem_f1,
            "n_epochs": len(y_sub),
        })

    subjects = [r["subject"] for r in results]
    x        = np.arange(len(subjects))
    width    = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(subjects)), 5))
    ax.bar(x - width, [r["accuracy"] for r in results],
           width, label="Accuracy", color="#378ADD", alpha=0.85)
    ax.bar(x,         [r["macro_f1"] for r in results],
           width, label="Macro F1", color="#1D9E75", alpha=0.85)
    ax.bar(x + width, [r["rem_f1"] for r in results],
           width, label="REM F1",   color="#EF9F27", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"sub-{s:03d}" for s in subjects],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{model_name} — Per-subject performance on test set")
    ax.legend()
    ax.axhline(0.8, color="gray", linestyle="--",
               linewidth=0.8, alpha=0.5, label="0.8 reference")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out = FIGURES_DIR / f"19_per_subject_{model_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")
    return results


def plot_predicted_hypnogram(
    model:         nn.Module,
    device:        torch.device,
    model_name:    str,
    sub_id:        int  = None,
    use_features:  bool = False,
    use_sequences: bool = False,
    seq_len:       int  = 31,
):
    """
    For one test subject, plot the true hypnogram vs the model's predicted
    hypnogram side by side. This is the most intuitive visualization of
    where the model gets confused.
    """
    X_all = np.load(DATA_DIR / ("F_all.npy" if use_features else "X_all.npy"))
    y_all = np.load(DATA_DIR / "y_all.npy")
    subs  = np.load(DATA_DIR / "subjects_all.npy")

    import json
    with open(DATA_DIR / "splits_metadata.json") as f:
        split_meta  = json.load(f)
    test_subs = split_meta["test_subjects"]

    # Pick the first test subject if none specified
    if sub_id is None:
        sub_id = sorted(test_subs)[0]
    elif sub_id not in test_subs:
        print(f"  Warning: sub-{sub_id:03d} is not in test set. "
              f"Using sub-{sorted(test_subs)[0]:03d} instead.")
        sub_id = sorted(test_subs)[0]

    mask  = subs == sub_id
    X_sub = X_all[mask]
    y_sub = y_all[mask]

    from src.Step4_DatasetSplitting.dataset import SleepDataset, SequenceDataset
    if use_sequences:
        sub_ids = subs[mask]
        ds = SequenceDataset(X_sub, y_sub, sub_ids, seq_len=seq_len)
    else:
        ds = SleepDataset(X_sub, y_sub)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    y_true, y_pred, _ = get_predictions(model, loader, device)

    # Convert class indices back to stage y-positions
    stage_y   = {0: 3, 2: 2, 1: 1}   # Wake=top, REM=mid, NREM=bottom
    epoch_sec = 4
    times     = np.arange(len(y_true)) * epoch_sec / 3600   # hours

    colors_true = [CLASS_COLORS[c] for c in y_true]
    colors_pred = [CLASS_COLORS[c] for c in y_pred]

    fig, axes = plt.subplots(2, 1, figsize=(18, 5), sharex=True)

    for ax, labels, colors, title in zip(
        axes,
        [y_true, y_pred],
        [colors_true, colors_pred],
        [f"sub-{sub_id:03d} — True hypnogram",
         f"sub-{sub_id:03d} — Predicted ({model_name})"],
    ):
        for i, (cls, color) in enumerate(zip(labels, colors)):
            y_pos = stage_y[cls]
            ax.broken_barh(
                [(times[i], epoch_sec / 3600)],
                (y_pos - 0.4, 0.8),
                facecolors=color, edgecolors="none"
            )
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["NREM", "REM", "Wake"])
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.25, linestyle="--")

    # Highlight errors in predicted panel
    error_mask = y_true != y_pred
    for i in np.where(error_mask)[0]:
        axes[1].axvspan(times[i], times[i] + epoch_sec / 3600,
                        alpha=0.25, color="#E24B4A", linewidth=0)

    axes[1].set_xlabel("Time (hours)")
    acc = (y_true == y_pred).mean()
    fig.suptitle(
        f"Hypnogram comparison — accuracy: {acc:.3f}  "
        f"(red = error)", fontsize=12
    )
    plt.tight_layout()
    out = FIGURES_DIR / f"20_hypnogram_{model_name}_sub{sub_id:03d}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


def plot_model_comparison(all_metrics: dict):
    """
    Bar chart comparing all three models on the key metrics.
    """
    metrics_to_plot = [
        ("accuracy",          "Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("macro_f1",          "Macro F1"),
        ("cohen_kappa",       "Cohen's Kappa"),
    ]

    model_names  = list(all_metrics.keys())
    model_colors = {"mlp": "#888780", "cnn": "#378ADD",
                    "cnn_lstm": "#1D9E75", "transformer": "#9B59B6"}
    model_labels = {
        "mlp":         "Baseline MLP",
        "cnn":         "1D CNN",
        "cnn_lstm":    "CNN + LSTM",
        "transformer": "Transformer",
    }

    n_models = len(model_names)
    fig, axes = plt.subplots(1, 4, figsize=(4 * max(n_models, 3) + 4, 5))

    for ax, (metric_key, metric_label) in zip(axes, metrics_to_plot):
        values = [all_metrics[m][metric_key] for m in model_names]
        colors = [model_colors[m] for m in model_names]
        labels = [model_labels[m] for m in model_names]

        bars = ax.bar(labels, values, color=colors,
                      edgecolor="white", linewidth=0.4)
        ax.set_title(metric_label, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Model comparison on test set", fontsize=12)
    plt.tight_layout()
    out = FIGURES_DIR / "21_model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


def plot_rem_analysis(all_results: dict):
    """
    Focused plot on REM performance — the hardest class.
    Shows precision, recall and F1 for REM across all models.
    """
    model_labels = {
        "mlp":         "Baseline MLP",
        "cnn":         "1D CNN",
        "cnn_lstm":    "CNN + LSTM",
        "transformer": "Transformer",
    }
    model_colors = {"mlp": "#888780", "cnn": "#378ADD",
                    "cnn_lstm": "#1D9E75", "transformer": "#9B59B6"}

    n_models = len(all_results)
    fig, ax = plt.subplots(figsize=(max(10, n_models * 3), 5))
    x       = np.arange(3)   # precision, recall, F1
    width   = 0.8 / max(n_models, 1)
    xlabels = ["Precision", "Recall", "F1"]

    for i, (model_name, metrics) in enumerate(all_results.items()):
        rem = metrics["per_class"]["REM"]
        vals = [rem["precision"], rem["recall"], rem["f1"]]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width,
                      label=model_labels[model_name],
                      color=model_colors[model_name],
                      edgecolor="white", linewidth=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("REM sleep classification — detailed analysis")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(0.7, color="gray", linestyle="--",
               linewidth=0.8, alpha=0.6, label="0.7 target")
    plt.tight_layout()
    out = FIGURES_DIR / "22_rem_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")