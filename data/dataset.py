"""Dataset loader for CARLA imitation-learning data.

Layout auto-detection
---------------------
The loader does not assume a fixed schema. It walks `root` looking for directories
that contain both:

  * a ``labels/`` directory (with ``*.parquet``, ``*.csv``, or ``*.json``), OR
    a single label file at episode root (``labels.parquet`` etc.)
  * at least one image directory (``images/<camera>/*.jpg`` is the canonical CARLA
    layout; a flat ``images/*.jpg`` directory is also supported).

Label rows are expected to expose — at minimum — ``steer``, ``throttle``, ``brake``
(or ``control_*`` equivalents). Image paths are read from a column matching
``img_<camera>`` or any ``image*`` column, and are resolved relative to the
episode directory.

Optional fields (``speed_kmh``, ``reward_total``, ``bucket``) are picked up when
present and passed through in the sample dict for downstream use.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from utils.sampler import steering_balanced_weights
from utils.transforms import Augment, Preprocess, build_augment, build_preprocess


log = logging.getLogger("bc.data")


# ---------------------------------------------------------------------------
# Column name resolution (schema-tolerant)
# ---------------------------------------------------------------------------

_STEER_ALIASES = ("steer", "steering", "control_steer", "steering_angle")
_THROTTLE_ALIASES = ("throttle", "control_throttle")
_BRAKE_ALIASES = ("brake", "control_brake")
_SPEED_ALIASES = ("speed_kmh", "speed", "speed_mps")
_REWARD_ALIASES = ("reward_total", "reward", "total_reward")


def _first_match(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    cols = {c.lower(): c for c in columns}
    for a in aliases:
        if a in cols:
            return cols[a]
    return None


def _resolve_image_column(columns: list[str], camera: str | None) -> str:
    """Find the column that holds the image path."""
    if camera:
        target = f"img_{camera}".lower()
        for c in columns:
            if c.lower() == target:
                return c
    # fall back: any column starting with img_ / image / path ending in common image keys
    candidates = [c for c in columns if c.lower().startswith(("img_", "image"))]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        log.warning("multiple image columns found %s; using %s", candidates, candidates[0])
        return candidates[0]
    raise KeyError(
        f"No image column found. Columns={columns}. "
        f"Expected one starting with 'img_' or matching camera={camera!r}."
    )


# ---------------------------------------------------------------------------
# Discovery + loading
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRef:
    episode_id: str
    root: Path                        # directory containing labels/ + images/
    label_files: list[Path] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def discover_episodes(
    root: str | Path,
    label_formats: list[str] | tuple[str, ...] = ("parquet", "csv", "json"),
) -> list[EpisodeRef]:
    """Walk `root` and return all detected episodes."""
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    episodes: list[EpisodeRef] = []
    seen: set[Path] = set()

    # Look for any dir that has a labels/ subdir OR a labels.* file, and an images/ subdir.
    # Use rglob on labels files then climb one level.
    globs = []
    for fmt in label_formats:
        globs.append(f"**/labels/*.{fmt}")
        globs.append(f"**/labels.{fmt}")

    label_paths: list[Path] = []
    for pattern in globs:
        label_paths.extend(root.glob(pattern))

    # Group by episode root (parent of labels/ or parent of a top-level labels.* file).
    by_root: dict[Path, list[Path]] = {}
    for lp in label_paths:
        ep_root = lp.parent.parent if lp.parent.name == "labels" else lp.parent
        by_root.setdefault(ep_root, []).append(lp)

    for ep_root, files in sorted(by_root.items()):
        if ep_root in seen:
            continue
        seen.add(ep_root)

        metadata = {}
        meta_path = ep_root / "metadata.json"
        if meta_path.exists():
            try:
                with meta_path.open("r") as f:
                    metadata = json.load(f)
            except Exception as e:
                log.warning("failed to read metadata for %s: %s", ep_root, e)

        ep_id = metadata.get("episode_id") or ep_root.name
        episodes.append(EpisodeRef(episode_id=str(ep_id), root=ep_root,
                                   label_files=sorted(files), metadata=metadata))

    if not episodes:
        raise RuntimeError(
            f"No episodes found under {root}. "
            f"Expected labels/*.{{{','.join(label_formats)}}} or labels.<fmt>."
        )

    log.info("discovered %d episode(s) under %s", len(episodes), root)
    return episodes


def _load_one_label_file(path: Path) -> pd.DataFrame:
    """Read a single parquet/csv/json labels file into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        # support both list-of-records and columnar formats
        try:
            return pd.read_json(path)
        except ValueError:
            with path.open("r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict):
                return pd.DataFrame.from_dict(data)
            raise
    raise ValueError(f"Unsupported label format: {path}")


def load_episode_labels(ep: EpisodeRef, camera: str | None = None) -> pd.DataFrame:
    """Load + concat all label files for an episode and normalize column names."""
    frames = []
    for f in ep.label_files:
        try:
            frames.append(_load_one_label_file(f))
        except Exception as e:
            log.warning("skipping unreadable label file %s: %s", f, e)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    cols = list(df.columns)
    steer = _first_match(cols, _STEER_ALIASES)
    thr = _first_match(cols, _THROTTLE_ALIASES)
    brk = _first_match(cols, _BRAKE_ALIASES)
    if not all([steer, thr, brk]):
        raise KeyError(
            f"[{ep.episode_id}] missing required control columns. "
            f"steer={steer} throttle={thr} brake={brk}; available={cols}"
        )

    img_col = _resolve_image_column(cols, camera)

    # Build a normalized frame with a stable set of names.
    out = pd.DataFrame({
        "episode_id": ep.episode_id,
        "image_path": df[img_col].astype(str).values,
        "steer": df[steer].astype(np.float32).values,
        "throttle": df[thr].astype(np.float32).values,
        "brake": df[brk].astype(np.float32).values,
    })

    # Optional passthrough fields
    if (c := _first_match(cols, _SPEED_ALIASES)) is not None:
        out["speed_kmh"] = df[c].astype(np.float32).values
    if (c := _first_match(cols, _REWARD_ALIASES)) is not None:
        out["reward_total"] = df[c].astype(np.float32).values
    if "bucket" in df.columns:
        out["bucket"] = df["bucket"].astype(str).values
    if "frame_idx" in df.columns:
        out["frame_idx"] = df["frame_idx"].astype(np.int64).values

    # Drop rows where controls are missing / NaN
    before = len(out)
    out = out.dropna(subset=["steer", "throttle", "brake"]).reset_index(drop=True)
    if len(out) != before:
        log.warning("[%s] dropped %d rows with NaN controls", ep.episode_id, before - len(out))
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CarlaBCDataset(Dataset):
    """Torch Dataset over (image, steer/throttle/brake) tuples."""

    def __init__(
        self,
        samples: pd.DataFrame,
        episode_roots: dict[str, Path],
        preprocess: Preprocess,
        augment: Augment | None = None,
        return_meta: bool = False,
    ):
        if "image_path" not in samples.columns:
            raise ValueError("samples dataframe must contain an 'image_path' column")

        self.df = samples.reset_index(drop=True)
        self.episode_roots = {str(k): Path(v) for k, v in episode_roots.items()}
        self.preprocess = preprocess
        self.augment = augment
        self.return_meta = return_meta

        # Precompute absolute image paths (fast; avoids string ops in __getitem__).
        rel_paths = self.df["image_path"].values
        ep_ids = self.df["episode_id"].values
        abs_paths = []
        for ep_id, rel in zip(ep_ids, rel_paths):
            root = self.episode_roots[str(ep_id)]
            p = Path(rel)
            abs_paths.append(str(p if p.is_absolute() else root / p))
        self._abs_paths: list[str] = abs_paths

        self._steer = self.df["steer"].to_numpy(dtype=np.float32)
        self._throttle = self.df["throttle"].to_numpy(dtype=np.float32)
        self._brake = self.df["brake"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def steer_values(self) -> np.ndarray:
        return self._steer

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path = self._abs_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Return a neighbor instead of crashing a worker on a corrupt file.
            log.warning("failed to open %s (%s) — using neighbor", img_path, e)
            return self.__getitem__((idx + 1) % len(self))

        steer = float(self._steer[idx])
        throttle = float(self._throttle[idx])
        brake = float(self._brake[idx])

        if self.augment is not None:
            img, steer = self.augment(img, steer)

        tensor = self.preprocess(img)
        targets = torch.tensor([steer, throttle, brake], dtype=torch.float32)

        out = {"image": tensor, "targets": targets}
        if self.return_meta:
            out["image_path"] = img_path
            out["episode_id"] = str(self.df["episode_id"].iloc[idx])
        return out


# ---------------------------------------------------------------------------
# Train/val split + dataloaders
# ---------------------------------------------------------------------------

def _split_episodes(
    episode_ids: list[str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    ids = list(episode_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_val = max(1, int(round(val_fraction * n))) if val_fraction > 0 else 0
    n_test = int(round(test_fraction * n))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"split leaves no training episodes: n={n} val={n_val} test={n_test}"
        )
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def _apply_filters(df: pd.DataFrame, filt) -> pd.DataFrame:
    if filt is None:
        return df
    out = df
    min_speed = filt.get("min_speed_kmh") if hasattr(filt, "get") else getattr(filt, "min_speed_kmh", None)
    buckets = filt.get("buckets") if hasattr(filt, "get") else getattr(filt, "buckets", None)

    if min_speed is not None and "speed_kmh" in out.columns:
        before = len(out)
        out = out[out["speed_kmh"] >= float(min_speed)].reset_index(drop=True)
        log.info("filter min_speed_kmh>=%s: %d -> %d rows", min_speed, before, len(out))
    if buckets and "bucket" in out.columns:
        before = len(out)
        out = out[out["bucket"].isin(list(buckets))].reset_index(drop=True)
        log.info("filter buckets=%s: %d -> %d rows", list(buckets), before, len(out))
    return out


def build_dataloaders(cfg) -> dict:
    """Top-level helper: discover data, split, build Datasets + DataLoaders.

    Returns a dict with keys: train_loader, val_loader, test_loader (maybe None),
    train_dataset, val_dataset, num_train, num_val, num_test.
    """
    data_cfg = cfg.data
    root = Path(data_cfg.root).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()

    episodes = discover_episodes(root, label_formats=list(data_cfg.get("label_formats",
                                                                       ["parquet", "csv", "json"])))
    ep_roots = {ep.episode_id: ep.root for ep in episodes}

    # Load all labels once; cheaper than reloading per split.
    per_ep_frames: dict[str, pd.DataFrame] = {}
    for ep in episodes:
        df = load_episode_labels(ep, camera=data_cfg.get("camera"))
        if len(df) == 0:
            log.warning("episode %s has no usable rows", ep.episode_id)
            continue
        df = _apply_filters(df, data_cfg.get("filter"))
        if len(df) == 0:
            log.warning("episode %s empty after filtering", ep.episode_id)
            continue
        per_ep_frames[ep.episode_id] = df

    if not per_ep_frames:
        raise RuntimeError("No data left after loading/filtering all episodes.")

    ep_ids = list(per_ep_frames.keys())
    split_cfg = data_cfg.split
    seed = int(cfg.experiment.seed)

    if split_cfg.strategy == "episode":
        train_ids, val_ids, test_ids = _split_episodes(
            ep_ids, float(split_cfg.val_fraction), float(split_cfg.get("test_fraction", 0.0)), seed
        )
        def _concat(ids): return pd.concat([per_ep_frames[i] for i in ids], ignore_index=True) if ids else pd.DataFrame()
        train_df = _concat(train_ids)
        val_df = _concat(val_ids)
        test_df = _concat(test_ids)
        log.info("episode split: train=%d val=%d test=%d episodes", len(train_ids), len(val_ids), len(test_ids))
    elif split_cfg.strategy == "random":
        full = pd.concat(list(per_ep_frames.values()), ignore_index=True)
        full = full.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(full)
        n_val = int(round(n * float(split_cfg.val_fraction)))
        n_test = int(round(n * float(split_cfg.get("test_fraction", 0.0))))
        n_train = n - n_val - n_test
        train_df = full.iloc[:n_train].reset_index(drop=True)
        val_df = full.iloc[n_train:n_train + n_val].reset_index(drop=True)
        test_df = full.iloc[n_train + n_val:].reset_index(drop=True)
    else:
        raise ValueError(f"Unknown split strategy: {split_cfg.strategy}")

    preprocess = build_preprocess(data_cfg.image)
    augment = build_augment(data_cfg.get("augment"))

    train_ds = CarlaBCDataset(train_df, ep_roots, preprocess, augment=augment)
    val_ds = CarlaBCDataset(val_df, ep_roots, preprocess, augment=None)
    test_ds = CarlaBCDataset(test_df, ep_roots, preprocess, augment=None) if len(test_df) else None

    loader_cfg = data_cfg.loader
    sampler_cfg = data_cfg.get("sampler") or {}

    sampler = None
    shuffle = True
    if sampler_cfg and sampler_cfg.get("enabled", False) and len(train_ds) > 0:
        w = steering_balanced_weights(
            train_ds.steer_values,
            num_bins=int(sampler_cfg.get("num_bins", 21)),
            smoothing=float(sampler_cfg.get("smoothing", 0.05)),
            max_weight=float(sampler_cfg.get("max_weight", 100.0)),
        )
        sampler = WeightedRandomSampler(weights=torch.from_numpy(w).double(),
                                        num_samples=len(train_ds), replacement=True)
        shuffle = False
        log.info("steering-balanced sampler enabled (bins=%d, max_w=%.1f)",
                 int(sampler_cfg.get("num_bins", 21)), float(sampler_cfg.get("max_weight", 100.0)))

    num_workers = int(loader_cfg.num_workers)
    common = dict(
        batch_size=int(loader_cfg.batch_size),
        num_workers=num_workers,
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
    )
    if num_workers > 0:
        common["persistent_workers"] = bool(loader_cfg.get("persistent_workers", True))
        common["prefetch_factor"] = int(loader_cfg.get("prefetch_factor", 2))

    train_loader = DataLoader(train_ds, shuffle=shuffle, sampler=sampler, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **common) if test_ds else None

    log.info("train samples: %d | val samples: %d | test samples: %d",
             len(train_ds), len(val_ds), 0 if test_ds is None else len(test_ds))

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_ds,
        "val_dataset": val_ds,
        "test_dataset": test_ds,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": 0 if test_ds is None else len(test_ds),
    }
