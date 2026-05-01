"""Logger + metric tracker shared by training and evaluation."""
from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path


_FORMAT = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"


def get_logger(name: str = "bc", log_file: str | Path | None = None,
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger  # already configured (e.g. re-imported)

    fmt = logging.Formatter(_FORMAT, datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


class MetricTracker:
    """Accumulate per-step metrics and emit running averages."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    def update(self, values: dict[str, float], n: int = 1) -> None:
        for k, v in values.items():
            if v is None:
                continue
            self._sums[k] += float(v) * n
            self._counts[k] += n

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()

    def avg(self, key: str) -> float:
        c = self._counts.get(key, 0)
        return self._sums[key] / c if c else float("nan")

    def as_dict(self) -> dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums if self._counts[k]}

    def format(self, keys: list[str] | None = None, precision: int = 4) -> str:
        d = self.as_dict()
        if keys is not None:
            d = {k: d[k] for k in keys if k in d}
        return " | ".join(f"{k}={v:.{precision}f}" for k, v in d.items())
