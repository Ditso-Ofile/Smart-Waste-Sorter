"""Step 3 - save figures of the augmentation pipeline for the report.

Run from the project root:
    python scripts/preview_augmentation.py

Saves to results/figures/:
    augmentation_<level>.png   - 1 original + 5 augmented versions, for 4 images
    background_replacement.png - original vs background-replaced, for 6 images
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from config import CLASSES, RESULTS  # noqa: E402
from data import _make_bg_replacer, build_augmentation, make_dataset  # noqa: E402

FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
tf.random.set_seed(0)

x, y = next(iter(make_dataset("train", training=True, batch_size=16)))

for level in ("standard", "strong"):
    aug = build_augmentation(level)
    rows, cols = 4, 6
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.9))
    for r in range(rows):
        img = x[r:r + 1]
        versions = [img[0]] + [aug(img, training=True)[0] for _ in range(cols - 1)]
        for c, v in enumerate(versions):
            ax = axes[r, c]
            ax.imshow(np.clip(v.numpy(), 0, 255).astype("uint8"))
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_title(f"original ({CLASSES[int(y[r])]})", fontsize=8)
    fig.suptitle(f"Augmentation: {level}", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"augmentation_{level}.png", dpi=110)
    plt.close(fig)
    print(f"saved augmentation_{level}.png")

replace = _make_bg_replacer(prob=1.0)
fig, axes = plt.subplots(2, 6, figsize=(12, 4.4))
for c in range(6):
    orig = x[c]
    swapped, _ = replace(tf.cast(orig, tf.uint8), y[c])
    for r, v in enumerate([orig, swapped]):
        ax = axes[r, c]
        ax.imshow(np.clip(v.numpy(), 0, 255).astype("uint8"))
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, c].set_title(CLASSES[int(y[c])], fontsize=9)
axes[0, 0].set_ylabel("original")
axes[1, 0].set_ylabel("background\nreplaced")
fig.tight_layout()
fig.savefig(FIG_DIR / "background_replacement.png", dpi=110)
plt.close(fig)
print("saved background_replacement.png")
