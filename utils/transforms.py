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

    def _sample_params(self) -> dict:
        """Draw one set of augmentation params — reused across a sequence."""
        return {
            "bright": (random.uniform(-self.brightness, self.brightness)
                       if self.brightness > 0 else 0.0),
            "contrast": (1.0 + random.uniform(-self.contrast, self.contrast)
                         if self.contrast > 0 else 1.0),
            "hue": (np.random.uniform(-self.hue_shift, self.hue_shift, size=(1, 1, 3)).astype(np.float32)
                    if self.hue_shift > 0 else None),
            "rotation": (random.uniform(-self.rotation_deg, self.rotation_deg)
                         if self.rotation_deg > 0 else 0.0),
            "hflip": (self.hflip_p > 0 and random.random() < self.hflip_p),
        }

    def _apply_to_image(self, img: Image.Image, p: dict) -> Image.Image:
        """Apply a pre-sampled param dict to one PIL image. Noise is sampled fresh
        per call so sensor noise remains iid across frames in a sequence."""
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC [0,1]
        if self.brightness > 0:
            arr = arr + p["bright"]
        if self.contrast > 0:
            arr = (arr - 0.5) * p["contrast"] + 0.5
        if self.hue_shift > 0:
            arr = arr + p["hue"]
        arr = np.clip(arr, 0.0, 1.0)

        if self.noise_std > 0:
            arr = arr + np.random.normal(0.0, self.noise_std, size=arr.shape).astype(np.float32)
            arr = np.clip(arr, 0.0, 1.0)

        img = Image.fromarray((arr * 255.0).astype(np.uint8))

        if self.rotation_deg > 0 and abs(p["rotation"]) > 1e-6:
            img = img.rotate(p["rotation"], resample=Image.BILINEAR, expand=False)

        if p["hflip"]:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def __call__(self, img: Image.Image, steer: float) -> tuple[Image.Image, float]:
        p = self._sample_params()
        img = self._apply_to_image(img, p)
        if p["hflip"]:
            steer = -steer
        return img, steer

    def apply_sequence(
        self, imgs: list[Image.Image], steer: float
    ) -> tuple[list[Image.Image], float]:
        """Apply the SAME jitter/rotation/flip params to every frame in a sequence.

        Keeps temporal coherence across the T-frame window so the LSTM doesn't
        see, e.g., a flip that toggles mid-sequence. Noise is per-frame iid.
        """
        p = self._sample_params()
        out = [self._apply_to_image(im, p) for im in imgs]
        if p["hflip"]:
            steer = -steer
        return out, steer


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
