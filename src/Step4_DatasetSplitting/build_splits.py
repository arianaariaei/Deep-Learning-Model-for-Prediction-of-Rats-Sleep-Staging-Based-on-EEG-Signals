# src/build_splits.py
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))  # project root
sys.path.insert(0, str(_HERE))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from src.config import DATA_DIR, FIGURES_DIR

from dataset import (
    subject_split,
    apply_subject_split,
    compute_class_weights,
    CLASS_NAMES,
)

# ── Load preprocessed data ────────────────────────────────────────────────────
print("Loading data...")
X    = np.load(DATA_DIR / "X_all.npy")
y    = np.load(DATA_DIR / "y_all.npy")
F    = np.load(DATA_DIR / "F_all.npy")
subs = np.load(DATA_DIR / "subjects_all.npy")

print(f"  X shape : {X.shape}   (raw epochs)")
print(f"  F shape : {F.shape}   (feature vectors)")
print(f"  Subjects: {np.unique(subs).tolist()}\n")

# ── Split subjects ────────────────────────────────────────────────────────────
train_subs, val_subs, test_subs = subject_split(subs)

print(f"Subject split:")
print(f"  Train : {len(train_subs)} subjects — {sorted(train_subs.tolist())}")
print(f"  Val   : {len(val_subs)}  subjects — {sorted(val_subs.tolist())}")
print(f"  Test  : {len(test_subs)}  subjects — {sorted(test_subs.tolist())}\n")

# ── Apply split to raw epochs ─────────────────────────────────────────────────
splits_raw = apply_subject_split(X, y, subs, train_subs, val_subs, test_subs)

# ── Apply split to feature matrix ─────────────────────────────────────────────
splits_feat = apply_subject_split(F, y, subs, train_subs, val_subs, test_subs)

# ── Print distribution before balancing ───────────────────────────────────────
print("=" * 55)
print("  EPOCH COUNTS PER SPLIT  (before oversampling)")
print("=" * 55)
for split_name in ["train", "val", "test"]:
    y_sp = splits_raw[split_name]["y"]
    total = len(y_sp)
    print(f"\n  {split_name.upper()} — {total:,} epochs")
    for cls, name in CLASS_NAMES.items():
        n   = int((y_sp == cls).sum())
        pct = n / total * 100
        print(f"    {name:<6}: {n:>7,}  ({pct:5.1f}%)")

# ── Class weights ──────────────────────────────────────────────────────────────
y_train     = splits_raw["train"]["y"]
class_weights = compute_class_weights(y_train)
print(f"\nClass weights (for loss function):")
for cls, name in CLASS_NAMES.items():
    print(f"  {name:<6}: {class_weights[cls]:.4f}")

# ── Save everything ───────────────────────────────────────────────────────────
print("\nSaving splits...")

# Raw epoch splits
np.save(DATA_DIR / "X_train.npy",    splits_raw["train"]["X"])
np.save(DATA_DIR / "X_val.npy",      splits_raw["val"]["X"])
np.save(DATA_DIR / "X_test.npy",     splits_raw["test"]["X"])

# Feature matrix splits
np.save(DATA_DIR / "F_train.npy",    splits_feat["train"]["X"])
np.save(DATA_DIR / "F_val.npy",      splits_feat["val"]["X"])
np.save(DATA_DIR / "F_test.npy",     splits_feat["test"]["X"])

# Labels
np.save(DATA_DIR / "y_train.npy",    splits_raw["train"]["y"])
np.save(DATA_DIR / "y_val.npy",      splits_raw["val"]["y"])
np.save(DATA_DIR / "y_test.npy",     splits_raw["test"]["y"])

# Class weights (needed in Step 6)
np.save(DATA_DIR / "class_weights.npy", class_weights)

# Metadata
meta = {
    "train_subjects":  sorted(train_subs.tolist()),
    "val_subjects":    sorted(val_subs.tolist()),
    "test_subjects":   sorted(test_subs.tolist()),
    "class_weights":   class_weights.tolist(),
    "counts": {
        split: {
            CLASS_NAMES[c]: int((splits_raw[split]["y"] == c).sum())
            for c in [0, 1, 2]
        }
        for split in ["train", "val", "test"]
    },
}
with open(DATA_DIR / "splits_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nFiles saved to {DATA_DIR}:")
for fname in sorted(DATA_DIR.glob("*.npy")):
    size_mb = fname.stat().st_size / 1e6
    print(f"  {fname.name:<25} {size_mb:6.1f} MB")

# ── Plot: class distribution across splits ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
colors = ["#378ADD", "#1D9E75", "#EF9F27"]

datasets = [
    ("Train", splits_raw["train"]["y"]),
    ("Val",   splits_raw["val"]["y"]),
    ("Test",  splits_raw["test"]["y"]),
]

for ax, (title, y_sp) in zip(axes, datasets):
    counts = [int((y_sp == c).sum()) for c in [0, 1, 2]]
    bars   = ax.bar(list(CLASS_NAMES.values()), counts,
                    color=colors, edgecolor="white", linewidth=0.4)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Epochs")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                f"{val:,}", ha="center", va="bottom", fontsize=8)

plt.suptitle("Class distribution across splits", fontsize=12)
plt.tight_layout()
out = FIGURES_DIR / "14_split_distribution.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved -> {out}")

# ── Plot: subject assignment visualization ────────────────────────────────────
all_subs    = np.unique(subs)
split_label = np.zeros(len(all_subs), dtype=int)   # 0=train,1=val,2=test
for i, s in enumerate(all_subs):
    if s in val_subs:
        split_label[i] = 1
    elif s in test_subs:
        split_label[i] = 2

split_colors = ["#378ADD", "#EF9F27", "#E24B4A"]
split_names  = ["Train", "Val", "Test"]

fig, ax = plt.subplots(figsize=(16, 2.5))
for i, (sub_id, label) in enumerate(zip(all_subs, split_label)):
    ax.bar(i, 1, color=split_colors[label],
           edgecolor="white", linewidth=0.3)
    if len(all_subs) <= 50:
        ax.text(i, 0.5, str(sub_id),
                ha="center", va="center", fontsize=6, color="white")

from matplotlib.patches import Patch
legend = [Patch(color=split_colors[i], label=split_names[i]) for i in range(3)]
ax.legend(handles=legend, loc="upper right", fontsize=9)
ax.set_xlim(-0.5, len(all_subs) - 0.5)
ax.set_yticks([])
ax.set_xlabel("Subject index")
ax.set_title("Subject assignment to train / val / test")
plt.tight_layout()
out = FIGURES_DIR / "15_subject_assignment.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved -> {out}")
