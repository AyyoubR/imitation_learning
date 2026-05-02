"""End-to-end training loop with AMP, TB logging, checkpointing, and resume."""
from __future__ import annotations

import math
import os
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from data.dataset import build_dataloaders
from models.model import BCModel, build_model, BCModel_SteeringOnly, build_model_steering_only
from utils.config import save_config
from utils.logging import MetricTracker, get_logger


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class WeightedControlLoss(nn.Module):
    """Weighted per-head loss over (steer, throttle, brake)."""

    def __init__(self, kind: str = "smoothl1",
                 w_steer: float = 1.0, w_throttle: float = 0.5, w_brake: float = 0.5):
        super().__init__()
        self.kind = kind
        self.register_buffer("weights", torch.tensor([w_steer, w_throttle, w_brake],
                                                     dtype=torch.float32))

    def _per_element(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.kind == "mse":
            return (pred - target) ** 2
        if self.kind == "smoothl1":
            return torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
        raise ValueError(f"Unknown loss kind: {self.kind}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        elems = self._per_element(pred, target)            # (B, 3)
        per_head = elems.mean(dim=0)                       # (3,)
        total = (per_head * self.weights.to(pred.device)).sum()
        return total, {
            "loss_steer": per_head[0].item(),
            "loss_throttle": per_head[1].item(),
            "loss_brake": per_head[2].item(),
        }

class WeightedControlLoss_SteeringOnly(nn.Module):
    """Weighted per-head loss over (steer, throttle, brake)."""

    def __init__(self, kind: str = "smoothl1",
                 w_steer: float = 1.0):
        super().__init__()
        self.kind = kind
        self.register_buffer("weights", torch.tensor([w_steer],
                                                     dtype=torch.float32))

    def _per_element(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.kind == "mse":
            return (pred - target) ** 2
        if self.kind == "smoothl1":
            return torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
        raise ValueError(f"Unknown loss kind: {self.kind}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        elems = self._per_element(pred, target)            # (B, 3)
        per_head = elems.mean(dim=0)                       # (3,)
        total = (per_head * self.weights.to(pred.device)).sum()
        return total, {
            "loss_steer": per_head[0].item(),
        }
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_device(pref: str) -> torch.device:
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def _build_optimizer(model: nn.Module, opt_cfg) -> Optimizer:
    kind = str(opt_cfg.get("kind", "adam")).lower()
    lr = float(opt_cfg.lr)
    wd = float(opt_cfg.get("weight_decay", 0.0))
    betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
    params = [p for p in model.parameters() if p.requires_grad]
    if kind == "adam":
        return Adam(params, lr=lr, betas=betas, weight_decay=wd)
    if kind == "adamw":
        return AdamW(params, lr=lr, betas=betas, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {kind}")


def _build_scheduler(optimizer: Optimizer, sched_cfg, total_epochs: int):
    kind = str(sched_cfg.get("kind", "none")).lower() if sched_cfg else "none"
    if kind == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, total_epochs), eta_min=float(sched_cfg.get("min_lr", 0.0)))
    if kind == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2,
                                 min_lr=float(sched_cfg.get("min_lr", 0.0)))
    return None


def _summary_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter  # lazy import
    except Exception as e:
        get_logger().warning("TensorBoard unavailable (%s) — logging to console only", e)
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(log_dir))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg, resume: str | Path | None = None):
        self.cfg = cfg
        self.resume_path = Path(resume) if resume else None

        exp = cfg.experiment
        self.device = _pick_device(str(exp.get("device", "auto")))
        self.use_amp = bool(exp.get("amp", True)) and self.device.type == "cuda"

        self.run_dir = (Path(exp.output_dir) / exp.name).resolve()
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.log_dir = self.run_dir / "tb"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        log_file = self.run_dir / "train.log"
        self.logger = get_logger("bc", log_file=log_file)
        self.logger.info("device=%s amp=%s run_dir=%s", self.device, self.use_amp, self.run_dir)

        self._seed_all(int(exp.seed))

        # Data
        data = build_dataloaders(cfg)
        self.train_loader: DataLoader = data["train_loader"]
        self.val_loader: DataLoader = data["val_loader"]
        self.test_loader: DataLoader | None = data["test_loader"]

        # Model
        if cfg.model.multi_task_steering_throttle_brake:
            self.model: BCModel = build_model(cfg).to(self.device)
        else:
            self.model: BCModel_SteeringOnly = build_model_steering_only(cfg).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info("model=%s params=%.2fM activation=%s",
                         self.model.arch, n_params / 1e6, self.model.activation)

        # Loss / optim / scheduler
        loss_cfg = cfg.training.loss
        if cfg.model.get("multi_task_steering_throttle_brake", True):
            self.criterion = WeightedControlLoss(
                kind=str(loss_cfg.get("kind", "smoothl1")),
                w_steer=float(loss_cfg.get("steer_weight", 1.0)),
                w_throttle=float(loss_cfg.get("throttle_weight", 0.5)),
                w_brake=float(loss_cfg.get("brake_weight", 0.5)),
            ).to(self.device)
        else:
            self.criterion = WeightedControlLoss_SteeringOnly(
                kind=str(loss_cfg.get("kind", "smoothl1")),
                w_steer=float(loss_cfg.get("steer_weight", 1.0)),
            ).to(self.device)

        self.optimizer = _build_optimizer(self.model, cfg.training.optimizer)
        self.total_epochs = int(cfg.training.epochs)
        self.scheduler = _build_scheduler(self.optimizer, cfg.training.get("scheduler"), self.total_epochs)
        self.warmup_epochs = int(cfg.training.get("scheduler", {}).get("warmup_epochs", 0))
        self.base_lr = float(cfg.training.optimizer.lr)

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Checkpoint bookkeeping
        self.start_epoch = 0
        self.best_val = math.inf
        self.epochs_since_improve = 0
        self._ckpt_queue: deque[Path] = deque()

        # TensorBoard
        self.writer = _summary_writer(self.log_dir) if bool(cfg.training.get("tensorboard", True)) else None

        # Persist resolved config alongside the checkpoints (for reproducibility).
        save_config(cfg, self.run_dir / "config.resolved.yaml")

        if self.resume_path is not None:
            self._load_checkpoint(self.resume_path)

    # ---- utilities ------------------------------------------------------

    def _seed_all(self, seed: int) -> None:
        import random as _r
        _r.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _warmup_lr(self, epoch: int, step: int, steps_per_epoch: int) -> None:
        if self.warmup_epochs <= 0:
            return
        global_step = epoch * steps_per_epoch + step
        warmup_steps = self.warmup_epochs * steps_per_epoch
        if global_step >= warmup_steps:
            return
        lr = self.base_lr * (global_step + 1) / max(1, warmup_steps)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_val": self.best_val,
            "val_loss": val_loss,
            "config": self.cfg.to_dict(),
            "arch": self.model.arch,
            "activation": self.model.activation,
            "input_hw": self.model.input_hw,
        }
        path = self.ckpt_dir / f"epoch_{epoch:04d}.pt"
        torch.save(state, path)
        self._ckpt_queue.append(path)

        keep = int(self.cfg.training.checkpoint.get("keep_last", 3))
        while len(self._ckpt_queue) > keep:
            old = self._ckpt_queue.popleft()
            try:
                old.unlink()
            except FileNotFoundError:
                pass

        # Always maintain a `last.pt` pointer for easy resume.
        last = self.ckpt_dir / "last.pt"
        if last.exists() or last.is_symlink():
            last.unlink()
        shutil.copyfile(path, last)

        if is_best and bool(self.cfg.training.checkpoint.get("save_best", True)):
            shutil.copyfile(path, self.ckpt_dir / "best.pt")
            self.logger.info("new best val_loss=%.6f saved to best.pt", val_loss)

    def _load_checkpoint(self, path: Path) -> None:
        self.logger.info("resuming from checkpoint: %s", path)
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(state["scaler_state_dict"])
        if self.scheduler is not None and state.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.start_epoch = int(state.get("epoch", 0)) + 1
        self.best_val = float(state.get("best_val", math.inf))

    # ---- one epoch ------------------------------------------------------

    def _run_epoch(self, loader: DataLoader, epoch: int, train: bool) -> dict[str, float]:
        self.model.train(train)
        tracker = MetricTracker()

        prefix = "train" if train else "val"
        steps = len(loader)
        t0 = time.time()

        for step, batch in enumerate(loader):
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["targets"].to(self.device, non_blocking=True)

            if train:
                self._warmup_lr(epoch, step, steps)

            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                preds = self.model(images)
                loss, head_losses = self.criterion(preds, targets)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                clip = float(self.cfg.training.get("grad_clip", 0.0))
                if clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # MAE per head (in original units, not loss units)
            with torch.no_grad():
                mae = (preds - targets).abs().mean(dim=0)
                steer_std = preds[:, 0].detach().float().std().item() if preds.shape[0] > 1 else 0.0

            n = images.size(0)
            if self.cfg.model.get("multi_task_steering_throttle_brake", True):
                tracker.update({
                    "loss": loss.item(),
                    **head_losses,
                    "mae_steer": mae[0].item(),
                    "mae_throttle": mae[1].item(),
                    "mae_brake": mae[2].item(),
                    "pred_steer_std": steer_std,
                }, n=n)
                if train and step % int(self.cfg.training.get("log_interval", 50)) == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.logger.info(
                        "ep=%d %s step=%d/%d lr=%.2e loss=%.4f mae_s=%.4f mae_t=%.4f mae_b=%.4f",
                        epoch, prefix, step, steps, lr,
                        tracker.avg("loss"), tracker.avg("mae_steer"),
                        tracker.avg("mae_throttle"), tracker.avg("mae_brake"),
                    )                
            
            else:
                tracker.update({
                    "loss": loss.item(),
                    **head_losses,
                    "mae_steer": mae[0].item(),
                    "pred_steer_std": steer_std,
                }, n=n)
                if train and step % int(self.cfg.training.get("log_interval", 50)) == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.logger.info(
                        "ep=%d %s step=%d/%d lr=%.2e loss=%.4f mae_s=%.4f",
                        epoch, prefix, step, steps, lr,
                        tracker.avg("loss"), tracker.avg("mae_steer"),
                    )



        metrics = tracker.as_dict()
        metrics["epoch_time_s"] = time.time() - t0
        return metrics

    # ---- top-level ------------------------------------------------------

    def fit(self) -> None:
        self.logger.info("starting training: epochs=%d start_epoch=%d",
                         self.total_epochs, self.start_epoch)

        for epoch in range(self.start_epoch, self.total_epochs):
            train_metrics = self._run_epoch(self.train_loader, epoch, train=True)
            with torch.no_grad():
                val_metrics = self._run_epoch(self.val_loader, epoch, train=False)

            # Scheduler step
            if self.scheduler is not None and epoch >= self.warmup_epochs:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            if self.cfg.model.get("multi_task_steering_throttle_brake", True):
                self.logger.info(
                    "[epoch %d] train_loss=%.4f val_loss=%.4f val_mae_s=%.4f val_mae_t=%.4f val_mae_b=%.4f lr=%.2e",
                    epoch, train_metrics["loss"], val_metrics["loss"],
                    val_metrics["mae_steer"], val_metrics["mae_throttle"], val_metrics["mae_brake"], lr,
                )
            else:   
                
                self.logger.info(
                    "[epoch %d] train_loss=%.4f val_loss=%.4f val_mae_s=%.4f lr=%.2e",
                    epoch, train_metrics["loss"], val_metrics["loss"],
                    val_metrics["mae_steer"], lr,
                )

            if self.writer is not None:
                for k, v in train_metrics.items():
                    self.writer.add_scalar(f"train/{k}", v, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)
                self.writer.add_scalar("lr", lr, epoch)

            # Best tracking + checkpoint
            val_loss = float(val_metrics["loss"])
            improved = val_loss < self.best_val - float(self.cfg.training.early_stopping.get("min_delta", 0.0))
            if improved:
                self.best_val = val_loss
                self.epochs_since_improve = 0
            else:
                self.epochs_since_improve += 1

            if (epoch + 1) % int(self.cfg.training.checkpoint.get("every_epochs", 1)) == 0 \
                    or epoch == self.total_epochs - 1:
                self._save_checkpoint(epoch, val_loss, is_best=improved)

            # Early stopping
            es_cfg = self.cfg.training.get("early_stopping") or {}
            if es_cfg.get("enabled", False):
                patience = int(es_cfg.get("patience", 8))
                if self.epochs_since_improve >= patience:
                    self.logger.info("early stopping: no improvement for %d epochs", patience)
                    break

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        self.logger.info("training complete. best val_loss=%.6f", self.best_val)


def run_training(cfg, resume: str | Path | None = None) -> Trainer:
    trainer = Trainer(cfg, resume=resume)
    trainer.fit()
    return trainer
