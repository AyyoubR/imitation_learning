"""Offline evaluation: predict on val/test set, compute metrics, emit plots."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import build_dataloaders
from models.model import BCModel, build_model, BCModel_SteeringOnly, build_model_steering_only
from utils.logging import get_logger


log = get_logger("bc.eval")


class Evaluator:
    def __init__(self, cfg, checkpoint: str | Path, split: str = "val",
                 output_dir: str | Path | None = None):
        self.cfg = cfg
        self.checkpoint_path = Path(checkpoint)
        self.split = split

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Rebuild data (uses the SAME preprocessing as training).
        data = build_dataloaders(cfg)
        if split == "val":
            self.loader: DataLoader = data["val_loader"]
        elif split == "train":
            self.loader = data["train_loader"]
        elif split == "test":
            if data["test_loader"] is None:
                raise RuntimeError("cfg.data.split.test_fraction=0 — no test set available")
            self.loader = data["test_loader"]
        else:
            raise ValueError(f"Unknown split: {split}")

        # Model
        if cfg.model.multi_task_steering_throttle_brake:
            self.model: BCModel = build_model(cfg).to(self.device)
        else: 
            self.model: BCModel_SteeringOnly = build_model_steering_only(cfg).to(self.device)

        state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
        log.info("loaded checkpoint %s (epoch=%s best_val=%.6f)",
                 self.checkpoint_path, state.get("epoch"), state.get("best_val", float("nan")))

        exp = cfg.experiment
        self.out_dir = Path(output_dir) if output_dir else \
            Path(exp.output_dir) / exp.name / f"eval_{split}"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def predict_all(self) -> tuple[np.ndarray, np.ndarray]:
        preds_all, targets_all = [], []
        for batch in self.loader:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["targets"]
            preds = self.model.predict_controls(images).cpu().numpy()
            preds_all.append(preds)
            targets_all.append(targets.numpy())
        return np.concatenate(preds_all, axis=0), np.concatenate(targets_all, axis=0)

    @staticmethod
    def _metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
        err = preds - targets
        abs_err = np.abs(err)
        mae = abs_err.mean(axis=0)
        rmse = np.sqrt((err ** 2).mean(axis=0))

        # Steering-specific checks
        steer_p, steer_t = preds[:, 0], targets[:, 0]
        steer_corr = float(np.corrcoef(steer_p, steer_t)[0, 1]) if np.std(steer_p) > 0 else 0.0
        # Direction agreement — ignore near-zero steers where direction is ambiguous.
        mask = np.abs(steer_t) > 0.02
        if mask.any():
            dir_acc = float(np.mean(np.sign(steer_p[mask]) == np.sign(steer_t[mask])))
        else:
            dir_acc = float("nan")

        return {
            "num_samples": int(preds.shape[0]),
            "mae": {"steer": float(mae[0]), "throttle": float(mae[1]), "brake": float(mae[2])},
            "rmse": {"steer": float(rmse[0]), "throttle": float(rmse[1]), "brake": float(rmse[2])},
            "overall_mae": float(abs_err.mean()),
            "steer_corr": steer_corr,
            "steer_direction_acc": dir_acc,
            "pred_steer_std": float(np.std(steer_p)),
            "target_steer_std": float(np.std(steer_t)),
        }

    def _plot(self, preds: np.ndarray, targets: np.ndarray) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            log.warning("matplotlib unavailable (%s); skipping plots", e)
            return

        names = ["steer", "throttle", "brake"]
        # 1) histograms: pred vs target
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i, (ax, name) in enumerate(zip(axes, names)):
            ax.hist(targets[:, i], bins=50, alpha=0.6, label="target", density=True)
            ax.hist(preds[:, i], bins=50, alpha=0.6, label="pred", density=True)
            ax.set_title(f"{name} distribution")
            ax.legend()
        fig.tight_layout()
        fig.savefig(self.out_dir / "histograms.png", dpi=120)
        plt.close(fig)

        # 2) scatter pred-vs-target
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i, (ax, name) in enumerate(zip(axes, names)):
            ax.scatter(targets[:, i], preds[:, i], s=2, alpha=0.3)
            lo = float(min(targets[:, i].min(), preds[:, i].min()))
            hi = float(max(targets[:, i].max(), preds[:, i].max()))
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
            ax.set_xlabel(f"target {name}")
            ax.set_ylabel(f"pred {name}")
            ax.set_title(f"{name} pred vs target")
        fig.tight_layout()
        fig.savefig(self.out_dir / "scatter.png", dpi=120)
        plt.close(fig)

        # 3) error over sample index (as a proxy for time-order within batch)
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i, (ax, name) in enumerate(zip(axes, names)):
            ax.plot(targets[:, i], label="target", linewidth=0.8)
            ax.plot(preds[:, i], label="pred", linewidth=0.8, alpha=0.8)
            ax.set_ylabel(name)
            ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("sample index")
        fig.tight_layout()
        fig.savefig(self.out_dir / "timeseries.png", dpi=120)
        plt.close(fig)

    def run(self) -> dict[str, Any]:
        preds, targets = self.predict_all()
        metrics = self._metrics(preds, targets)

        # Save artifacts
        np.save(self.out_dir / "predictions.npy", preds)
        np.save(self.out_dir / "targets.npy", targets)
        with (self.out_dir / "metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)

        if bool(self.cfg.evaluation.get("save_plots", True)):
            self._plot(preds, targets)

        log.info("eval done. overall_mae=%.4f mae_steer=%.4f corr=%.3f dir_acc=%.3f",
                 metrics["overall_mae"], metrics["mae"]["steer"],
                 metrics["steer_corr"], metrics["steer_direction_acc"])
        log.info("artifacts saved to %s", self.out_dir)
        return metrics


def run_evaluation(cfg, checkpoint: str | Path, split: str = "val",
                   output_dir: str | Path | None = None) -> dict[str, Any]:
    return Evaluator(cfg, checkpoint=checkpoint, split=split, output_dir=output_dir).run()
