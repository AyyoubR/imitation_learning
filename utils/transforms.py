"""Image preprocessing + training augmentations (torch/PIL only, no torchvision)."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from PIL import Image


# ImageNet stats — sensible defaults even though we don't use a pretrained backbone,
# because they center/normalize natural images well.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Preprocess:
    """Deterministic preprocessing: crop → resize → to-tensor → normalize.

    Designed to be called inside Dataset.__getitem__ (CPU), so workers can parallelize.
    """
    crop_top: int
    crop_bottom: int
    resize_width: int
    resize_height: int
    normalize: str  # "imagenet" | "minmax"

    def __call__(self, img: Image.Image) -> torch.Tensor:
        w, h = img.size
        top = max(0, self.crop_top)
        bottom = max(top + 1, h - self.crop_bottom)
        img = img.crop((0, top, w, bottom))
        img = img.resize((self.resize_width, self.resize_height), Image.BILINEAR)

        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
        if self.normalize == "imagenet":
            mean = np.array(IMAGENET_MEAN, dtype=np.float32)
            std = np.array(IMAGENET_STD, dtype=np.float32)
            arr = (arr - mean) / std
        elif self.normalize == "minmax":
            arr = arr * 2.0 - 1.0  # [-1, 1]
        elif self.normalize in (None, "none"):
            pass
        else:
            raise ValueError(f"Unknown normalize mode: {self.normalize!r}")

        # HWC -> CHW
        tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        return tensor


def build_preprocess(image_cfg) -> Preprocess:
    return Preprocess(
        crop_top=int(image_cfg.get("crop_top", 0)),
        crop_bottom=int(image_cfg.get("crop_bottom", 0)),
        resize_width=int(image_cfg.resize_width),
        resize_height=int(image_cfg.resize_height),
        normalize=str(image_cfg.get("normalize", "minmax") or "none"),
    )


# ---------------------------------------------------------------------------
# Augmentations are applied to the *PIL image* before preprocessing so that
# preprocess (deterministic) can remain a single object on both train & val.
# ---------------------------------------------------------------------------

class Augment:
    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        hue_shift: float = 0.0,
        gaussian_noise_std: float = 0.0,
        rotation_deg: float = 0.0,
        horizontal_flip_prob: float = 0.0,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.hue_shift = hue_shift
        self.noise_std = gaussian_noise_std
        self.rotation_deg = rotation_deg
        self.hflip_p = horizontal_flip_prob

    def __call__(self, img: Image.Image, steer: float) -> tuple[Image.Image, float]:
        # 1) color jitter via numpy (keeps things dependency-light).
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC [0,1]
        if self.brightness > 0:
            arr = arr + random.uniform(-self.brightness, self.brightness)
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            arr = (arr - 0.5) * factor + 0.5
        if self.hue_shift > 0:
            # cheap approximation: small channel-wise biases
            shifts = np.random.uniform(-self.hue_shift, self.hue_shift, size=(1, 1, 3)).astype(np.float32)
            arr = arr + shifts
        arr = np.clip(arr, 0.0, 1.0)

        # 2) Gaussian noise
        if self.noise_std > 0:
            arr = arr + np.random.normal(0.0, self.noise_std, size=arr.shape).astype(np.float32)
            arr = np.clip(arr, 0.0, 1.0)

        img = Image.fromarray((arr * 255.0).astype(np.uint8))

        # 3) Small rotation — simulates camera tilt, doesn't change steering sign.
        if self.rotation_deg > 0:
            angle = random.uniform(-self.rotation_deg, self.rotation_deg)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

        # 4) Horizontal flip — flips the sign of steering! Guard with prob.
        if self.hflip_p > 0 and random.random() < self.hflip_p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            steer = -steer

        return img, steer


def build_augment(aug_cfg) -> Augment | None:
    if not aug_cfg or not aug_cfg.get("enabled", False):
        return None
    return Augment(
        brightness=float(aug_cfg.get("brightness", 0.0)),
        contrast=float(aug_cfg.get("contrast", 0.0)),
        hue_shift=float(aug_cfg.get("hue_shift", 0.0)),
        gaussian_noise_std=float(aug_cfg.get("gaussian_noise_std", 0.0)),
        rotation_deg=float(aug_cfg.get("rotation_deg", 0.0)),
        horizontal_flip_prob=float(aug_cfg.get("horizontal_flip_prob", 0.0)),
    )


def denormalize_for_display(tensor: torch.Tensor, normalize: str) -> np.ndarray:
    """Invert the normalization so an image tensor can be shown / logged."""
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    if normalize == "imagenet":
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        arr = arr * std + mean
    elif normalize == "minmax":
        arr = (arr + 1.0) / 2.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr
