"""Step 2c - stratified 70/15/15 train/val/test split, saved as CSVs.

Run from the project root (after explore_data.py):
    python scripts/make_splits.py

Writes data/splits/train.csv, val.csv, test.csv with columns: path, label.
Paths are relative to the project root with forward slashes, so the same
CSVs work on Windows and on Colab. Commit these CSVs to Git - they are tiny
and guarantee both teammates (and every experiment) use identical splits.

Near-duplicate groups (from results/near_duplicates.csv) are kept together
in one split, so the same object cannot appear in both train and test.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import CLASSES, DATA_RAW, DATA_SPLITS, RESULTS, ROOT, SEED  # noqa: E402

DATA_DIR = DATA_RAW / "trashnet"


def list_images():
    rows = [{"path": p.relative_to(ROOT).as_posix(), "label": cls}
            for cls in CLASSES
            for p in sorted((DATA_DIR / cls).glob("*"))
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return pd.DataFrame(rows)


def duplicate_groups(df):
    """Union-find over near-duplicate pairs -> a group id per image."""
    parent = {p: p for p in df.path}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    dup_file = RESULTS / "near_duplicates.csv"
    n_pairs = 0
    if dup_file.exists():
        dups = pd.read_csv(dup_file)
        dups = dups[dups.label_a == dups.label_b]          # cross-class matches are false positives
        for a, b in dups[["path_a", "path_b"]].itertuples(index=False):
            if a in parent and b in parent:
                parent[find(a)] = find(b)
                n_pairs += 1
    else:
        print("near_duplicates.csv not found - run explore_data.py first for leak-free splits")
    groups = df.path.map(find)
    sizes = groups.value_counts()
    print(f"{n_pairs} near-duplicate pairs -> {groups.nunique()} groups for {len(df)} images "
          f"(largest group: {sizes.iloc[0]} images)")
    if sizes.iloc[0] > 15:
        print("  WARNING: a very large group usually means false matches - "
              "re-run explore_data.py with a higher --min-corr, e.g. 0.97")
    return groups


def split(df, groups):
    # StratifiedGroupKFold with 20 folds: 3 folds (15%) -> test, 3 -> val, 14 (70%) -> train.
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    fold = pd.Series(-1, index=df.index)
    for k, (_, idx) in enumerate(sgkf.split(df.path, df.label, groups)):
        fold.iloc[idx] = k
    df = df.assign(split="train")
    df.loc[fold.isin([0, 1, 2]), "split"] = "test"
    df.loc[fold.isin([3, 4, 5]), "split"] = "val"
    return df


def main():
    if not DATA_DIR.exists():
        sys.exit(f"{DATA_DIR} not found - run scripts/download_trashnet.py first")
    df = list_images()
    df = split(df, duplicate_groups(df))

    DATA_SPLITS.mkdir(parents=True, exist_ok=True)
    for name in ["train", "val", "test"]:
        df[df.split == name][["path", "label"]].to_csv(DATA_SPLITS / f"{name}.csv", index=False)

    table = pd.crosstab(df.label, df.split).reindex(CLASSES)[["train", "val", "test"]]
    table.loc["TOTAL"] = table.sum()
    table["train_%"] = (table.train / table.sum(axis=1) * 100).round(1)
    print("\n", table.to_string(), sep="")
    table.to_csv(RESULTS / "split_counts.csv")

    # Sanity check: no image in two splits
    assert df.path.is_unique, "duplicate paths in split"
    print(f"\nSaved {DATA_SPLITS / 'train.csv'}, val.csv, test.csv")


if __name__ == "__main__":
    main()
