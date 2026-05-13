# src/models.py
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL A — Baseline MLP (trained on handcrafted features)
#  Input : (batch, 29)   — feature vectors from Step 4
#  Output: (batch, 3)    — logits for Wake / NREM / REM
# ══════════════════════════════════════════════════════════════════════════════

class BaselineMLP(nn.Module):
    """
    Simple 3-layer MLP on the 29 handcrafted features.
    Use this as a baseline — if the CNN doesn't beat this,
    something is wrong with the deep model.
    """
    def __init__(self, n_features: int = 29, n_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL B — 1D CNN  (trained on raw EEG epochs)
#  Input : (batch, 2, 512)  — 2 channels (EEG1, EMG), 512 time samples
#  Output: (batch, 3)
#
#  Architecture:
#    Three convolutional blocks with increasing filters and decreasing
#    kernel size — wide kernels first to capture slow waves (delta),
#    narrow kernels later to capture fast oscillations (spindles, gamma).
#    Global average pooling collapses the time dimension so the
#    classifier head is independent of input length.
# ══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Conv1d → BatchNorm → ReLU → MaxPool → Dropout"""
    def __init__(self, in_ch, out_ch, kernel, pool=2, dropout=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(pool),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class SleepCNN(nn.Module):
    """
    1D Convolutional Neural Network for sleep stage classification.

    Input shape : (batch, 2, 512)
    Output shape: (batch, 3)

    The network learns its own features directly from the raw signal.
    It should outperform the MLP baseline because it can discover
    patterns (like spindle shape, delta wave morphology) that the
    handcrafted features miss.
    """
    def __init__(self, n_channels: int = 2, n_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()

        # ── Feature extractor ─────────────────────────────────────────────────
        # Block 1: large kernel (50 ≈ 0.4s at 128Hz) catches slow waves
        # Block 2: medium kernel (25) catches spindles / alpha bursts
        # Block 3: small kernel (10) catches fast gamma activity
        self.features = nn.Sequential(
            ConvBlock(n_channels, 32,  kernel=50, pool=4, dropout=0.25),
            ConvBlock(32,         64,  kernel=25, pool=2, dropout=0.25),
            ConvBlock(64,         128, kernel=10, pool=2, dropout=0.25),
            ConvBlock(128,        256, kernel=5,  pool=2, dropout=0.25),
        )

        # Global average pooling — averages across remaining time steps
        # so output is always (batch, 256) regardless of input length
        self.gap = nn.AdaptiveAvgPool1d(1)

        # ── Classifier head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)    # (batch, 256, T')
        x = self.gap(x)         # (batch, 256, 1)
        x = self.classifier(x)  # (batch, 3)
        return x


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL C — CNN + LSTM  (captures temporal context between epochs)
#  Input : (batch, seq_len, 2, 512)  — a sequence of consecutive epochs
#  Output: (batch, 3)
#
#  Sleep stages don't occur independently — REM always follows NREM,
#  Wake transitions to NREM gradually. The LSTM captures this context.
#  This is the strongest model but also the most complex to train.
# ══════════════════════════════════════════════════════════════════════════════

class SleepCNNLSTM(nn.Module):
    """
    CNN feature extractor + LSTM sequence model.

    Each epoch in a sequence is passed through the CNN independently
    to get a 256-dim embedding. The sequence of embeddings is then
    passed through an LSTM. The final hidden state is classified.

    Input shape : (batch, seq_len, n_channels, n_samples)
    Output shape: (batch, n_classes)  — label for the LAST epoch in seq
    """
    def __init__(self, n_channels: int = 2, n_classes: int = 3,
                 seq_len: int = 21, lstm_hidden: int = 128,
                 lstm_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.seq_len = seq_len

        # Shared CNN encoder (same weights applied to every epoch in sequence)
        self.cnn = nn.Sequential(
            ConvBlock(n_channels, 32,  kernel=50, pool=4, dropout=0.2),
            ConvBlock(32,         64,  kernel=25, pool=2, dropout=0.2),
            ConvBlock(64,         128, kernel=10, pool=2, dropout=0.2),
            ConvBlock(128,        256, kernel=5,  pool=2, dropout=0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),           # → (batch, 256)
        )

        # LSTM reads the sequence of CNN embeddings
        self.lstm = nn.LSTM(
            input_size  = 256,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
            bidirectional = True,
        )

        # Classifier on the final LSTM output
        lstm_out_size = lstm_hidden * 2   # ×2 because bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, C, T)
        batch, seq_len, C, T = x.shape

        # Encode each epoch independently with CNN
        x = x.view(batch * seq_len, C, T)   # (batch*seq, C, T)
        x = self.cnn(x)                      # (batch*seq, 256)
        x = x.view(batch, seq_len, 256)      # (batch, seq, 256)

        # LSTM over the sequence
        out, _ = self.lstm(x)                # (batch, seq, lstm_out)

        # Take only the output at the last time step (center epoch)
        x = out[:, seq_len // 2, :]         # (batch, lstm_out)
        return self.classifier(x)            # (batch, n_classes)


def get_model(name: str, **kwargs) -> nn.Module:
    """Factory function — returns model by name."""
    models = {
        "mlp":      BaselineMLP,
        "cnn":      SleepCNN,
        "cnn_lstm": SleepCNNLSTM,
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(models)}")
    return models[name](**kwargs)