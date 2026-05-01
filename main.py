"""Single CLI entry point for training and evaluation.

Examples
--------
    # Train with default config
    python main.py train --config configs/default.yaml

    # Train with overrides
    python main.py train --config configs/default.yaml \\
        experiment.name=exp_pilotnet_v2 training.epochs=60 data.loader.batch_size=128

    # Resume
    python main.py train --config configs/default.yaml --resume runs/exp_pilotnet_v2/checkpoints/last.pt

    # Evaluate a checkpoint on val split
    python main.py eval  --config configs/default.yaml --checkpoint runs/exp_pilotnet_v2/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure local modules are importable when running the file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import load_config, merge_overrides  # noqa: E402
from utils.logging import get_logger                    # noqa: E402


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config (relative paths resolved from CWD)")
    parser.add_argument("overrides", nargs="*",
                        help="Dotted overrides, e.g. training.epochs=60 data.loader.batch_size=32")


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CARLA behavioral cloning pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a model")
    _common_args(p_train)
    p_train.add_argument("--resume", type=str, default=None,
                         help="Path to checkpoint to resume from (optional)")

    p_eval = sub.add_parser("eval", help="Evaluate a trained checkpoint")
    _common_args(p_eval)
    p_eval.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint to load")
    p_eval.add_argument("--split", choices=["train", "val", "test"], default="val")
    p_eval.add_argument("--output-dir", type=str, default=None)

    p_data = sub.add_parser("inspect-data", help="Discover episodes and show label stats")
    _common_args(p_data)

    return parser.parse_args()


def main() -> int:
    args = _parse()
    cfg = load_config(args.config)
    if args.overrides:
        cfg = merge_overrides(cfg, args.overrides)

    log = get_logger("bc")
    log.info("command=%s config=%s", args.command, args.config)

    if args.command == "train":
        from training.train import run_training
        run_training(cfg, resume=args.resume)
        return 0

    if args.command == "eval":
        from evaluation.eval import run_evaluation
        run_evaluation(cfg, checkpoint=args.checkpoint, split=args.split,
                       output_dir=args.output_dir)
        return 0

    if args.command == "inspect-data":
        from data.dataset import discover_episodes, load_episode_labels
        eps = discover_episodes(cfg.data.root, label_formats=list(cfg.data.get("label_formats",
                                                                               ["parquet", "csv", "json"])))
        total = 0
        for ep in eps:
            df = load_episode_labels(ep, camera=cfg.data.get("camera"))
            total += len(df)
            log.info("%s: %d rows | steer mean=%.3f std=%.3f | buckets=%s",
                     ep.episode_id, len(df),
                     float(df["steer"].mean()) if len(df) else float("nan"),
                     float(df["steer"].std()) if len(df) else float("nan"),
                     sorted(df["bucket"].unique().tolist()) if "bucket" in df.columns else "n/a")
        log.info("TOTAL rows across %d episodes: %d", len(eps), total)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
