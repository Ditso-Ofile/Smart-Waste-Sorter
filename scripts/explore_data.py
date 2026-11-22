"""Step 2b - explore TrashNet and save figures for the report.

Run from the project root:
    python scripts/explore_data.py

Saves to results/figures/:
    class_distribution.png   - bar chart of images per class (imbalance)
    sample_grid.png          - 5 random images per class
and to results/:
    dataset_stats.csv        - per-class counts, image sizes, mean brightness
    near_duplicates.csv      - pairs of near-identical images (used by make_splits.py)
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # no window needed; just save the figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import CLASSES, DATA_RAW, RESULTS, ROOT, SEED  # noqa: E402

DATA_DIR = DATA_RAW / "trashnet"
FIG_DIR = RESULTS / "figures"
# Near-duplicate detection. TrashNet images are mostly flat white background, where a
# perceptual hash is unreliable (sensor noise flips its bits, and unrelated images look
# alike). Instead we compare 32x32 grayscale thumbnails with Pearson correlation, only
# within the SAME class (one object cannot be two materials). Tune with --min-corr.
THUMB = 32
MIN_CORR = 0.95


def imread(path):
    # np.fromfile + imdecode works with non-ASCII / spaced Windows paths, unlike cv2.imread
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def scan():
    rows = []
    for cls in CLASSES:
        for p in sorted((DATA_DIR / cls).glob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = imread(p)
            if img is None:
                print(f"  unreadable, skipped: {p}")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rows.append({
                "path": p.relative_to(ROOT).as_posix(),
                "label": cls,
                "height": img.shape[0],
                "width": img.shape[1],
                "brightness": float(gray.mean()),
                "thumb": cv2.resize(gray, (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype("float32").ravel(),
            })
    return pd.DataFrame(rows)


def plot_distribution(df):
    counts = df.label.value_counts().reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color="#4C72B0")
    ax.bar_label(bars)
    ax.set_ylabel("Images")
    ax.set_title(f"TrashNet class distribution (n = {len(df)})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_samples(df, per_class=5):
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(per_class * 2, len(CLASSES) * 1.6))
    for r, cls in enumerate(CLASSES):
        paths = df[df.label == cls].path.to_numpy()
        for c, rel in enumerate(rng.choice(paths, size=min(per_class, len(paths)), replace=False)):
            img = cv2.cvtColor(imread(ROOT / rel), cv2.COLOR_BGR2RGB)
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(cls, fontsize=11)
    fig.suptitle("Random TrashNet samples (note the uniform white background)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sample_grid.png", dpi=120)
    plt.close(fig)


def find_near_duplicates(df, min_corr):
    pairs = []
    for cls in CLASSES:                                   # same-class comparisons only
        sub = df[df.label == cls]
        t = np.stack(sub.thumb.to_numpy())
        t = (t - t.mean(1, keepdims=True)) / (t.std(1, keepdims=True) + 1e-6)
        corr = t @ t.T / t.shape[1]                       # all pairwise correlations at once
        ii, jj = np.nonzero(np.triu(corr, k=1) >= min_corr)
        for i, j in zip(ii, jj):
            pairs.append({"path_a": sub.path.iat[i], "path_b": sub.path.iat[j],
                          "label_a": cls, "label_b": cls, "corr": round(float(corr[i, j]), 3)})
    return pd.DataFrame(pairs, columns=["path_a", "path_b", "label_a", "label_b", "corr"])


def plot_duplicate_pairs(dups, n=8):
    """Side-by-side view of matched pairs so you can check them by eye."""
    if dups.empty:
        return
    show = dups.sample(min(n, len(dups)), random_state=SEED)
    fig, axes = plt.subplots(len(show), 2, figsize=(5, 2 * len(show)))
    axes = np.atleast_2d(axes)
    for r, row in enumerate(show.itertuples(index=False)):
        for c, rel in enumerate([row.path_a, row.path_b]):
            ax = axes[r, c]
            ax.imshow(cv2.cvtColor(imread(ROOT / rel), cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(Path(rel).name, fontsize=8)
        axes[r, 0].set_ylabel(f"r={row.corr}", fontsize=8)
    fig.suptitle("Random near-duplicate pairs (check: same object?)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "near_duplicate_pairs.png", dpi=110)
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-corr", type=float, default=MIN_CORR, help="min thumbnail correlation (0-1); higher = stricter")
    args = ap.parse_args()
    if not DATA_DIR.exists():
        sys.exit(f"{DATA_DIR} not found - run scripts/download_trashnet.py first")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Scanning images...")
    df = scan()

    stats = df.groupby("label").agg(images=("path", "size"),
                                    width=("width", "median"), height=("height", "median"),
                                    brightness=("brightness", "mean")).reindex(CLASSES)
    stats["share_%"] = (stats.images / stats.images.sum() * 100).round(1)
    stats.to_csv(RESULTS / "dataset_stats.csv")
    print("\n", stats.round(1).to_string(), sep="")
    ratio = stats.images.max() / stats.images.min()
    print(f"\nImbalance: largest / smallest class = {ratio:.1f}x")
    sizes = df.groupby(["width", "height"]).size()
    print(f"Image sizes: {dict(sizes)}")

    plot_distribution(df)
    plot_samples(df)

    print("\nLooking for near-duplicates...")
    dups = find_near_duplicates(df, args.min_corr)
    dups.to_csv(RESULTS / "near_duplicates.csv", index=False)
    plot_duplicate_pairs(dups)
    per_img = pd.concat([dups.path_a, dups.path_b]).value_counts()
    print(f"  {len(dups)} same-class near-duplicate pairs "
          f"(min corr={args.min_corr})")
    if len(dups):
        print(f"  Per class: {dups.label_a.value_counts().reindex(CLASSES, fill_value=0).to_dict()}")
        print(f"  Most-matched image: {per_img.index[0]} ({per_img.iloc[0]} matches)")
        print("  Check results/figures/near_duplicate_pairs.png - each row should show the same object.")

    print(f"\nSaved figures to {FIG_DIR} and tables to {RESULTS}")


if __name__ == "__main__":
    main()
