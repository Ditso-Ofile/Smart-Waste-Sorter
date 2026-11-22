"""Step 2a - download TrashNet (resized, 512x384) and unpack it into data/raw/trashnet.

Run from the project root:
    python scripts/download_trashnet.py

Result:
    data/raw/trashnet/cardboard/*.jpg
    data/raw/trashnet/glass/*.jpg
    ... one folder per class
"""
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import CLASSES, DATA_RAW  # noqa: E402

URL = "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip"
ZIP_PATH = DATA_RAW / "dataset-resized.zip"
OUT_DIR = DATA_RAW / "trashnet"
EXPECTED = {"cardboard": 403, "glass": 501, "metal": 410, "paper": 594, "plastic": 482, "trash": 137}


def download():
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 40_000_000:
        print(f"Zip already downloaded: {ZIP_PATH}")
        return
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}\n  -> {ZIP_PATH}")

    def progress(blocks, block_size, total):
        done = blocks * block_size
        if total > 0:
            pct = min(100, done * 100 // total)
            print(f"\r  {pct:3d}%  {done / 1e6:5.1f} / {total / 1e6:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(URL, ZIP_PATH, reporthook=progress)
    print()


def extract():
    if OUT_DIR.exists():
        print(f"Already extracted: {OUT_DIR} (delete it to re-extract)")
        return
    tmp = DATA_RAW / "_tmp_extract"
    shutil.rmtree(tmp, ignore_errors=True)
    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(tmp)

    # The class folders may sit one or two levels deep; find them, skip macOS junk.
    OUT_DIR.mkdir(parents=True)
    for cls in CLASSES:
        candidates = [p for p in tmp.rglob(cls) if p.is_dir() and "__MACOSX" not in p.parts]
        if not candidates:
            sys.exit(f"Could not find folder '{cls}' inside the zip - check {tmp}")
        shutil.move(str(candidates[0]), OUT_DIR / cls)
    shutil.rmtree(tmp, ignore_errors=True)


def verify():
    print("\nImages per class:")
    total, ok = 0, True
    for cls in CLASSES:
        n = len([p for p in (OUT_DIR / cls).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        total += n
        flag = "" if n == EXPECTED[cls] else f"   <-- expected {EXPECTED[cls]}"
        ok &= n == EXPECTED[cls]
        print(f"  {cls:10s} {n:4d}{flag}")
    print(f"  {'TOTAL':10s} {total:4d}")
    print("\nDataset looks complete." if ok else "\nCounts differ from the published ones - check the download.")


if __name__ == "__main__":
    download()
    extract()
    verify()
