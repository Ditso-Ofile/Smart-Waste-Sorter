"""Reproducibility + experiment logging helpers."""
import json
import os
import random
import time
from datetime import datetime

import numpy as np

from config import CLASSES, MODELS, RUN_LOG


def seed_everything(seed: int = 42) -> None:
    """Fix all random seeds so runs are comparable across experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Optional: fully deterministic GPU ops (slower). Uncomment if needed.
        # tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass


def log_run(**fields) -> None:
    """Append one experiment's settings + metrics to results/runs.csv.

    New columns are added automatically, so older rows simply get blanks.

    Example:
        log_run(exp_id="E2", backbone="MobileNetV2", val_acc=0.87, macro_f1=0.84)
    """
    import pandas as pd

    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"timestamp": datetime.now().isoformat(timespec="seconds"), **fields}])
    if RUN_LOG.exists():
        row = pd.concat([pd.read_csv(RUN_LOG), row], ignore_index=True)
    row.to_csv(RUN_LOG, index=False)


def save_class_names(model_name: str) -> None:
    """Save the class order next to the model so the webcam app can load it."""
    MODELS.mkdir(parents=True, exist_ok=True)
    with open(MODELS / f"{model_name}_classes.json", "w") as f:
        json.dump(CLASSES, f)


class Timer:
    """Context manager: `with Timer() as t: ...; print(t.ms)`."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self._start) * 1000
