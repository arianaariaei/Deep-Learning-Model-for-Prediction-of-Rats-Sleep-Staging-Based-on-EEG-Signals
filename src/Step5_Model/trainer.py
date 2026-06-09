# src/trainer.py
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))  # project root
sys.path.insert(0, str(_HERE))

import torch
import torch.nn as nn
import numpy as np
import json
import time
from torch.utils.data import DataLoader
from src.config import DATA_DIR


def augment_batch(X: torch.Tensor) -> torch.Tensor:
    """
    Random augmentation for raw EEG batches.
    Works for CNN (B, C, T) and CNN+LSTM/Transformer (B, S, C, T).

    Gaussian noise + amplitude jitter + random time shift.
    Applied only during training, not validation.
    """
    orig_shape = X.shape
    if X.dim() == 3:
        flat = X.clone()
    elif X.dim() == 4:
        B, S, C, T = X.shape
        flat = X.clone().view(B * S, C, T)
    else:
        return X

    n = flat.shape[0]
    flat = flat + torch.randn_like(flat) * 0.02
    scale = torch.empty(n, 1, 1, device=flat.device).uniform_(0.8, 1.2)
    flat = flat * scale
    shifts = torch.randint(-16, 17, (n,)).tolist()
    for i, s in enumerate(shifts):
        if s != 0:
            flat[i] = torch.roll(flat[i], s, dims=-1)
    return flat.view(orig_shape)


class EarlyStopping:
    """
    Stop training when validation loss stops improving.
    Saves the best model weights automatically.
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 checkpoint_path: Path = None):
        self.patience         = patience
        self.min_delta        = min_delta
        self.checkpoint_path  = checkpoint_path
        self.best_loss        = np.inf
        self.counter          = 0
        self.best_state       = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
            if self.checkpoint_path:
                torch.save(self.best_state, self.checkpoint_path)
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        """Load the best weights back into the model."""
        if self.best_state:
            model.load_state_dict(self.best_state)


def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    scaler:    torch.cuda.amp.GradScaler = None,
    augment:   bool = False,
) -> tuple[float, float]:
    """
    One full pass over the training set.
    Returns (avg_loss, accuracy).
    Uses mixed precision if scaler is provided (faster on GPU).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        if augment:
            X_batch = augment_batch(X_batch)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    """Evaluate on val or test set. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        logits  = model(X_batch)
        loss    = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


def train(
    model:          nn.Module,
    train_loader:   DataLoader,
    val_loader:     DataLoader,
    class_weights:  np.ndarray,
    checkpoint_dir: Path,
    model_name:     str,
    lr:             float = 1e-3,
    n_epochs:       int   = 100,
    patience:       int   = 10,
    weight_decay:   float = 1e-4,
    augment:        bool  = False,
) -> dict:
    """
    Full training loop with:
      - Weighted cross-entropy loss   (handles class imbalance)
      - Adam optimizer
      - ReduceLROnPlateau scheduler   (halves LR when val loss plateaus)
      - Early stopping                (stops when val loss stops improving)
      - Mixed precision               (if CUDA available — ~2× faster)
      - Gradient clipping             (prevents exploding gradients in LSTM)

    Returns history dict with loss/accuracy curves.
    """
    device = torch.device("cuda")
    print(f"\nTraining on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = model.to(device)

    # Weighted loss — penalizes REM errors more heavily
    weights   = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    checkpoint_path = checkpoint_dir / f"{model_name}_best.pt"
    early_stop      = EarlyStopping(patience=patience,
                                    checkpoint_path=checkpoint_path)

    # Mixed precision scaler (only meaningful on CUDA)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "lr":         [],
    }

    print(f"\n{'Epoch':<7} {'Train Loss':<12} {'Train Acc':<12} "
          f"{'Val Loss':<12} {'Val Acc':<12} {'LR':<10} {'Time'}")
    print("-" * 75)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            augment=augment,
        )
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(va_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["lr"].append(current_lr)

        print(f"{epoch:<7} {tr_loss:<12.4f} {tr_acc:<12.4f} "
              f"{va_loss:<12.4f} {va_acc:<12.4f} {current_lr:<10.2e} "
              f"{elapsed:.1f}s")

        if early_stop.step(va_loss, model):
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(best val loss: {early_stop.best_loss:.4f})")
            break

    # Restore best weights
    early_stop.restore_best(model)
    print(f"\nBest weights restored from: {checkpoint_path}")

    return history