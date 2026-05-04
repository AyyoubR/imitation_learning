"""Behavioral cloning model: image -> (steer, throttle, brake).

Two architectures are provided:

* ``PilotNet`` — the classic NVIDIA end-to-end CNN (5 conv + 3 FC). Works well on
  small inputs (e.g. 200x88 or 66x200) and is a strong baseline for BC.
* ``DeepCNN`` — a slightly deeper ResNet-inspired stack for higher-resolution inputs.

Both feed a shared trunk into three linear heads whose outputs are squashed to
the physical control ranges when ``activation="bounded"``:
  * steer    -> tanh     -> [-1, 1]
  * throttle -> sigmoid  -> [0, 1]
  * brake    -> sigmoid  -> [0, 1]
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Output squashing
# ---------------------------------------------------------------------------

def bounded_controls(raw: torch.Tensor) -> torch.Tensor:
    """Map raw 3-vector -> (tanh, sigmoid, sigmoid)."""
    steer = torch.tanh(raw[..., 0:1])
    throttle = torch.sigmoid(raw[..., 1:2])
    brake = torch.sigmoid(raw[..., 2:3])
    return torch.cat([steer, throttle, brake], dim=-1)


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------

class _PilotNet(nn.Module):
    """NVIDIA PilotNet — 5 conv layers, designed for 200x66 or 200x88 inputs."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class _ResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.skip = (nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                                   nn.BatchNorm2d(out_c))
                     if (stride != 1 or in_c != out_c) else nn.Identity())

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class _DeepCNN(nn.Module):
    """Small ResNet-like backbone for 200x88+ inputs."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(_ResBlock(32, 64, stride=2), _ResBlock(64, 64))
        self.layer2 = nn.Sequential(_ResBlock(64, 128, stride=2), _ResBlock(128, 128))
        self.layer3 = nn.Sequential(_ResBlock(128, 256, stride=2), _ResBlock(256, 256))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class _DeepCNN_LSTM_steering_only(nn.Module):
    """DeepCNN encoder + LSTM temporal head for steering-only regression.

    Input shape : ``(B, T, C, H, W)`` — a short window of frames ordered
                  oldest→newest. Default ``T=4`` (current + 3 past frames).
    Output shape: ``(B, 1)`` steering. When ``activation="bounded"`` the
                  output is squashed to ``[-1, 1]`` with tanh.

    The DeepCNN encoder is shared across timesteps (time is folded into the
    batch dimension for a single conv pass), followed by global average
    pooling to a per-frame 256-d vector. The sequence of per-frame vectors
    is fed to an LSTM and the final hidden state is mapped to a single
    steering value through a small MLP head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        seq_len: int = 4,
        feat_dim: int = 256,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        activation: Literal["bounded", "linear"] = "bounded",
        input_hw: tuple[int, int] = (88, 200),
    ):
        super().__init__()
        # Attributes the training/checkpointing code reads off the model.
        self.arch = "deepcnn_lstm"
        self.activation = activation
        self.input_hw = input_hw
        self.seq_len = seq_len
        self.feat_dim = feat_dim

        # Per-frame CNN encoder — weights are shared across timesteps by
        # running the encoder once on (B*T, C, H, W).
        self.encoder = _DeepCNN(in_channels=in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"Expected input of shape (B, T, C, H, W); got {tuple(x.shape)}"
            )
        b, t, c, h, w = x.shape
        if t != self.seq_len:
            raise ValueError(
                f"Expected T={self.seq_len} frames (current + {self.seq_len - 1} past); got T={t}"
            )

        feats = self.encoder(x.reshape(b * t, c, h, w))   # (B*T, 256, h', w')
        feats = self.pool(feats).flatten(1)               # (B*T, 256)
        feats = feats.view(b, t, self.feat_dim)           # (B, T, 256)

        lstm_out, _ = self.lstm(feats)                    # (B, T, hidden)
        last = lstm_out[:, -1, :]                         # (B, hidden)

        raw = self.head(last)                             # (B, 1)
        if self.activation == "bounded":
            return torch.tanh(raw)
        return raw

    @torch.no_grad()
    def predict_controls(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        if self.activation != "bounded":
            out = out.clamp(-1.0, 1.0)
        return out
    
# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class BCModel(nn.Module):
    """Image encoder + regression heads for (steer, throttle, brake)."""

    def __init__(
        self,
        arch: Literal["pilotnet", "deepcnn"] = "pilotnet",
        dropout: float = 0.2,
        activation: Literal["bounded", "linear"] = "bounded",
        input_hw: tuple[int, int] = (88, 200),
    ):
        super().__init__()
        self.arch = arch
        self.activation = activation
        self.input_hw = input_hw

        if arch == "pilotnet":
            self.backbone = _PilotNet()
        elif arch == "deepcnn":
            self.backbone = _DeepCNN()
        else:
            raise ValueError(f"Unknown arch: {arch}")

        # Figure out the flattened feature size with a dry forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_hw[0], input_hw[1])
            feat = self.backbone(dummy)
            feat_channels = feat.shape[1]
            feat_flat = feat.flatten(1).shape[1]

        if arch == "pilotnet":
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(feat_flat, 100), nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(100, 50), nn.ELU(),
                nn.Linear(50, 10), nn.ELU(),
                nn.Linear(10, 3),
            )
        else:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(feat_channels, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, 3),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        raw = self.head(feat)
        if self.activation == "bounded":
            return bounded_controls(raw)
        # "linear" — return raw values; caller is expected to clamp for inference
        return raw

    @torch.no_grad()
    def predict_controls(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model and always clamp to physical control ranges."""
        out = self.forward(x)
        if self.activation != "bounded":
            out = torch.stack([
                out[..., 0].clamp(-1.0, 1.0),
                out[..., 1].clamp(0.0, 1.0),
                out[..., 2].clamp(0.0, 1.0),
            ], dim=-1)
        return out


class BCModel_SteeringOnly(nn.Module):
    """Image encoder + regression head for steering only."""

    def __init__(
        self,
        arch: Literal["pilotnet", "deepcnn"] = "pilotnet",
        dropout: float = 0.2,
        activation: Literal["bounded", "linear"] = "bounded",
        input_hw: tuple[int, int] = (88, 200),
    ):
        super().__init__()
        self.arch = arch
        self.activation = activation
        self.input_hw = input_hw

        if arch == "pilotnet":
            self.backbone = _PilotNet()
        elif arch == "deepcnn":
            self.backbone = _DeepCNN()
        else:
            raise ValueError(f"Unknown arch: {arch}")

        # Figure out the flattened feature size with a dry forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_hw[0], input_hw[1])
            feat = self.backbone(dummy)
            feat_channels = feat.shape[1]
            feat_flat = feat.flatten(1).shape[1]

        if arch == "pilotnet":
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(feat_flat, 100), nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(100, 50), nn.ELU(),
                nn.Linear(50, 10), nn.ELU(),
                nn.Linear(10, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(feat_channels, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        raw = self.head(feat)
        if self.activation == "bounded":
            return bounded_controls(raw)
        # "linear" — return raw values; caller is expected to clamp for inference
        return raw

    @torch.no_grad()
    def predict_controls(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model and always clamp to physical control ranges."""
        out = self.forward(x)
        if self.activation != "bounded":
            out = torch.stack([
                out[..., 0].clamp(-1.0, 1.0)
            ], dim=-1)
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg) -> BCModel:
    image_cfg = cfg.data.image
    input_hw = (int(image_cfg.resize_height), int(image_cfg.resize_width))
    model_cfg = cfg.model
    return BCModel(
        arch=str(model_cfg.get("arch", "pilotnet")),
        dropout=float(model_cfg.get("dropout", 0.2)),
        activation=str(model_cfg.get("activation", "bounded")),
        input_hw=input_hw,
    )

def build_model_steering_only(cfg):
    image_cfg = cfg.data.image
    input_hw = (int(image_cfg.resize_height), int(image_cfg.resize_width))
    model_cfg = cfg.model
    arch = str(model_cfg.get("arch", "pilotnet"))
    if arch == "deepcnn_lstm":
        return _DeepCNN_LSTM_steering_only(
            seq_len=int(model_cfg.get("seq_len", 4)),
            lstm_hidden=int(model_cfg.get("lstm_hidden", 256)),
            lstm_layers=int(model_cfg.get("lstm_layers", 1)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            activation=str(model_cfg.get("activation", "bounded")),
            input_hw=input_hw,
        )
    return BCModel_SteeringOnly(
        arch=arch,
        dropout=float(model_cfg.get("dropout", 0.2)),
        activation=str(model_cfg.get("activation", "bounded")),
        input_hw=input_hw,
    )