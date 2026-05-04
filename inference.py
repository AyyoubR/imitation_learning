"""Standalone inference helper for CARLA integration.

Usage pattern inside a CARLA control loop:

    from inference import BCController
    ctrl = BCController("runs/bc_pilotnet_v1/checkpoints/best.pt")
    ...
    image_rgb = array_from_carla_camera  # uint8 HxWx3
    steer, throttle, brake = ctrl.act(image_rgb)
    vehicle.apply_control(carla.VehicleControl(steer=steer, throttle=throttle, brake=brake))

This file only uses torch + PIL + numpy, so it can be dropped into any CARLA client.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from models.model import (
    BCModel,
    BCModel_SteeringOnly,
    _DeepCNN_LSTM_steering_only,
    build_model,
    build_model_steering_only,
)
from utils.transforms import Preprocess


class BCController:
    def __init__(self, checkpoint_path: str | Path, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        state = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        cfg = state["config"]
        image_cfg = cfg["data"]["image"]

        self.preprocess = Preprocess(
            crop_top=int(image_cfg.get("crop_top", 0)),
            crop_bottom=int(image_cfg.get("crop_bottom", 0)),
            resize_width=int(image_cfg["resize_width"]),
            resize_height=int(image_cfg["resize_height"]),
            normalize=str(image_cfg.get("normalize", "minmax") or "none"),
        )

        input_hw = (int(image_cfg["resize_height"]), int(image_cfg["resize_width"]))
        arch = str(state.get("arch", cfg["model"].get("arch", "pilotnet")))
        activation = str(state.get("activation", cfg["model"].get("activation", "bounded")))
        dropout = float(cfg["model"].get("dropout", 0.0))
        self.arch = arch

        if arch == "deepcnn_lstm":
            # Steering-only temporal model. No throttle/brake head — the CARLA
            # script supplies those from its own speed controller.
            self.multi_task = False
            self.seq_len = int(cfg["model"].get("seq_len", 4))
            self.model = _DeepCNN_LSTM_steering_only(
                seq_len=self.seq_len,
                lstm_hidden=int(cfg["model"].get("lstm_hidden", 256)),
                lstm_layers=int(cfg["model"].get("lstm_layers", 1)),
                dropout=dropout,
                activation=activation,
                input_hw=input_hw,
            ).to(self.device)
            # FIFO holding the most recent `seq_len` preprocessed frames on GPU.
            self._frame_buf: deque = deque(maxlen=self.seq_len)
        else:
            self.seq_len = 1
            self.multi_task = bool(cfg["model"].get("multi_task_steering_throttle_brake", False))
            if self.multi_task:
                self.model = BCModel(
                    arch=arch,
                    dropout=dropout,
                    activation=activation,
                    input_hw=input_hw,
                ).to(self.device)
            else:
                self.model = BCModel_SteeringOnly(
                    arch=arch,
                    dropout=dropout,
                    activation=activation,
                    input_hw=input_hw,
                ).to(self.device)

        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

    def reset(self) -> None:
        """Clear the temporal frame buffer. Call between episodes so the LSTM
        doesn't carry frames from the previous run into the next one."""
        if hasattr(self, "_frame_buf"):
            self._frame_buf.clear()

    @torch.no_grad()
    def act(self, rgb_image: np.ndarray) -> Tuple[float, float, float]:
        """Take an HxWx3 uint8 RGB array and return (steer, throttle, brake).

        steer ∈ [-1, 1], throttle ∈ [0, 1], brake ∈ [0, 1].

        For the temporal ``deepcnn_lstm`` arch, this maintains an internal
        FIFO of the last ``seq_len`` frames and pads with the first buffered
        frame until the buffer fills — matching the training-time padding in
        ``CarlaBCSequenceDataset``. Call ``reset()`` between episodes.

        Throttle and brake are mutually exclusive on return: whichever the
        model predicts larger is kept, the other is zeroed. BC training
        labels from CARLA's autopilot often contain simultaneous small-brake +
        big-throttle frames; at steady-state driving the throttle wins, but
        from a cold spawn (engine RPM=0) even a tiny brake torque exceeds the
        throttle's wheel torque and the car sits motionless.
        """
        if rgb_image.dtype != np.uint8:
            raise TypeError("Expected uint8 RGB image (HxWx3)")
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB, got {rgb_image.shape}")

        img = Image.fromarray(rgb_image, mode="RGB")
        tensor = self.preprocess(img).to(self.device)  # (C, H, W)

        if self.arch == "deepcnn_lstm":
            self._frame_buf.append(tensor)
            # Pad by repeating the OLDEST buffered frame — mirrors
            # CarlaBCSequenceDataset where padding repeats frame 0 of the episode.
            first = self._frame_buf[0]
            needed = self.seq_len - len(self._frame_buf)
            frames = ([first] * needed) + list(self._frame_buf)
            x = torch.stack(frames, dim=0).unsqueeze(0)   # (1, T, C, H, W)
        else:
            x = tensor.unsqueeze(0)                       # (1, C, H, W)

        pred = self.model.predict_controls(x).squeeze(0).cpu().numpy()
        if self.multi_task:
            steer = float(np.clip(pred[0], -1.0, 1.0))
            throttle = float(np.clip(pred[1], 0.0, 1.0))
            brake = float(np.clip(pred[2], 0.0, 1.0))
            if throttle >= brake:
                brake = 0.0
            else:
                throttle = 0.0
        else:
            steer = float(np.clip(pred[0], -1.0, 1.0))
            throttle = 0.0
            brake = 0.0
        return steer, throttle, brake

    @torch.no_grad()
    def act_with_raw(self, rgb_image: np.ndarray) -> Tuple[float, float, float, float, float]:
        """Same as :meth:`act` but also returns the pre-gate throttle/brake.

        Returns ``(steer, throttle, brake, throttle_raw, brake_raw)``. The
        raw values are the clipped model outputs before mutual-exclusion
        gating — useful for diagnosing deadlocks (if both are consistently
        non-zero, the training labels had the same dual-output artifact).

        Only valid for the multi-task model (steer + throttle + brake heads).
        """
        if not self.multi_task:
            raise RuntimeError(
                "act_with_raw requires a multi-task model; this checkpoint "
                f"has arch={self.arch!r} (steering-only)."
            )
        if rgb_image.dtype != np.uint8:
            raise TypeError("Expected uint8 RGB image (HxWx3)")
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB, got {rgb_image.shape}")

        img = Image.fromarray(rgb_image, mode="RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        pred = self.model.predict_controls(tensor).squeeze(0).cpu().numpy()

        steer = float(np.clip(pred[0], -1.0, 1.0))
        throttle_raw = float(np.clip(pred[1], 0.0, 1.0))
        brake_raw = float(np.clip(pred[2], 0.0, 1.0))
        if throttle_raw >= brake_raw:
            throttle, brake = throttle_raw, 0.0
        else:
            throttle, brake = 0.0, brake_raw
        return steer, throttle, brake, throttle_raw, brake_raw


def _demo(checkpoint: str, image_path: str) -> None:
    """Quick smoke test: load a checkpoint + run inference on a sample image."""
    ctrl = BCController(checkpoint)
    img = np.array(Image.open(image_path).convert("RGB"))
    steer, throttle, brake = ctrl.act(img)
    print(f"steer={steer:+.4f} throttle={throttle:.4f} brake={brake:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    args = p.parse_args()
    _demo(args.checkpoint, args.image)
