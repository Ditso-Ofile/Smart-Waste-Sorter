"""Single source of truth for paths and constants.

Every script (training, evaluation, webcam app) imports from here so that
class order, image size and paths can never drift apart.
"""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"            # TrashNet unzipped here (Step 2)
DATA_SPLITS = ROOT / "data" / "splits"      # train.csv / val.csv / test.csv
WEBCAM_TEST = ROOT / "data" / "webcam_test" # your own captured images (Step 8)
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
RUN_LOG = RESULTS / "runs.csv"              # one row per experiment

# Model used by default in the webcam app, benchmark and TFLite export.
# E6-lr1e-4 has the best test macro-F1 (0.886). E4-lr1e-4 is the faster alternative
# (0.881 macro-F1, best trash F1, ~25% lower latency): switch here if the demo laptop is slow.
BEST_MODEL = "E6-lr1e-4"

# --- Data ------------------------------------------------------------------
# Alphabetical order = the order Keras' image_dataset_from_directory uses.
# The webcam app MUST use this exact list, in this order.
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = (224, 224)

# --- Training defaults -----------------------------------------------------
SEED = 42
BATCH_SIZE = 32
SPLIT = (0.70, 0.15, 0.15)  # train / val / test
