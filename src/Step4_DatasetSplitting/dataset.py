# src/dataset.py
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))  # project root
sys.path.insert(0, str(_HERE))

import numpy as np
from collections import Counter
from src.config import DATA_DIR

CLASS_NAMES = {0: "Wake", 1: "NREM", 2: "REM"}

# ══════════════════════════════════════════════════════════════════════════════
#  SPLIT STRATEGY
# ══════════════════════════════════════════════════════════════════════════════
#
#  We split by SUBJECT, not by epoch.
#
#  Why? Because epochs from the same subject are highly correlated —
#  if sub-001 epoch 500 is in train and epoch 501 is in val, the model
#  sees nearly identical signals in both sets and validation accuracy
#  becomes artificially inflated. This is called "data leakage".
#
#  Subject-level split = realistic estimate of how the model performs
#  on a mouse it has never seen before.
#
#  Split: 70% train / 15% val / 15% test  (by number of subjects)
# ══════════════════════════════════════════════════════════════════════════════

def subject_split(
    subject_ids: np.ndarray,
    train_frac:  float = 0.70,
    val_frac:    float = 0.15,
    seed:        int   = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split unique subject IDs into train / val / test groups.
    Returns three arrays of subject IDs.
    """
    rng      = np.random.default_rng(seed)
    unique   = np.unique(subject_ids)
    n        = len(unique)
    shuffled = rng.permutation(unique)

    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_subs = shuffled[:n_train]
    val_subs   = shuffled[n_train : n_train + n_val]
    test_subs  = shuffled[n_train + n_val :]

    return train_subs, val_subs, test_subs


def apply_subject_split(
    X:           np.ndarray,
    y:           np.ndarray,
    subject_ids: np.ndarray,
    train_subs:  np.ndarray,
    val_subs:    np.ndarray,
    test_subs:   np.ndarray,
) -> dict:
    """
    Use the subject ID arrays to create epoch-level index masks,
    then slice X and y accordingly.

    Returns a dict with keys: train, val, test
    Each value is a dict with keys: X, y, subjects
    """
    def mask(subs):
        return np.isin(subject_ids, subs)

    splits = {}
    for name, subs in [("train", train_subs),
                        ("val",   val_subs),
                        ("test",  test_subs)]:
        m = mask(subs)
        splits[name] = {
            "X":        X[m],
            "y":        y[m],
            "subjects": subject_ids[m],
        }

    return splits


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS BALANCING  (training set only)
# ══════════════════════════════════════════════════════════════════════════════
#
#  The dataset is imbalanced: Wake ~51%, NREM ~42%, REM ~7%.
#  Without correction the model learns to ignore REM almost entirely,
#  achieving high overall accuracy but terrible REM recall.
#
#  We use TWO complementary strategies:
#
#  1. OVERSAMPLING (SMOTE-style random oversampling on raw epochs):
#     Duplicate minority class epochs randomly until all classes
#     have equal count. Applied to the training set only — never
#     val or test (that would give a false picture of real performance).
#
#  2. CLASS WEIGHTS (for the loss function):
#     Tell the model to penalize REM mistakes more heavily.
#     Used during training in Step 6 as an alternative or complement
#     to oversampling.
# ══════════════════════════════════════════════════════════════════════════════

def random_oversample(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly duplicate minority class samples until all classes
    have the same count as the majority class.

    Only call this on the TRAINING set.
    Input/output shape: (N, ...) — works for both raw epochs and features.
    """
    rng        = np.random.default_rng(seed)
    counts     = Counter(y.tolist())
    max_count  = max(counts.values())

    X_list = [X]
    y_list = [y]

    for cls, count in counts.items():
        if count == max_count:
            continue
        n_needed = max_count - count
        idx      = np.where(y == cls)[0]
        chosen   = rng.choice(idx, size=n_needed, replace=True)
        X_list.append(X[chosen])
        y_list.append(y[chosen])

    X_bal = np.concatenate(X_list, axis=0)
    y_bal = np.concatenate(y_list, axis=0)

    # Shuffle so classes aren't blocked together
    perm  = rng.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute per-class weights inversely proportional to class frequency.
    Used as the `weight` argument in nn.CrossEntropyLoss in Step 5.

    Formula: w_c = N / (n_classes × count_c)
    This ensures the loss contribution is equal across classes
    regardless of how imbalanced the counts are.

    Returns array of shape (3,) — one weight per class.
    """
    counts  = np.bincount(y, minlength=3).astype(float)
    n_total = len(y)
    n_cls   = 3
    weights = n_total / (n_cls * counts)
    # Normalize so weights average to 1.0
    weights = weights / weights.mean()
    return weights.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  PYTORCH DATASET WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

import torch
from torch.utils.data import Dataset, DataLoader

class SleepDataset(Dataset):
    """
    PyTorch Dataset wrapping numpy arrays.
    Works for both raw epochs X (N, C, T) and feature vectors F (N, 29).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_dataloaders(
    splits:        dict,
    batch_size:    int  = 128,
    oversample:    bool = True,
    num_workers:   int  = 0,      # set to 4 if on Linux; keep 0 on Windows
    seed:          int  = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """
    Build train / val / test DataLoaders from the splits dict.

    Args:
        splits      : output of apply_subject_split()
        batch_size  : epochs per batch
        oversample  : if True, balance training set via random oversampling
        num_workers : parallel data loading workers

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    X_tr, y_tr = splits["train"]["X"], splits["train"]["y"]
    X_va, y_va = splits["val"]["X"],   splits["val"]["y"]
    X_te, y_te = splits["test"]["X"],  splits["test"]["y"]

    # Compute class weights BEFORE oversampling (on real distribution)
    class_weights = compute_class_weights(y_tr)

    # Oversample training set only
    if oversample:
        X_tr, y_tr = random_oversample(X_tr, y_tr, seed=seed)

    train_loader = DataLoader(
        SleepDataset(X_tr, y_tr),
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        SleepDataset(X_va, y_va),
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        SleepDataset(X_te, y_te),
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, class_weights


class SequenceDataset(Dataset):
    """
    Wraps epoch arrays into overlapping sequences of length seq_len.
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
            idx   = np.where(subjects == sub_id)[0]
            X_sub = X[idx]
            y_sub = y[idx]

            for i in range(half, len(X_sub) - half):
                seq = X_sub[i - half : i + half + 1]
                self.sequences.append(seq)
                self.labels.append(y_sub[i])

        self.sequences = np.stack(self.sequences).astype(np.float32)
        self.labels    = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.sequences[idx]),
                torch.tensor(self.labels[idx]))