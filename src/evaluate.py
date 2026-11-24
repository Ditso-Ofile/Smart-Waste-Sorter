"""Step 6 - final evaluation on the TEST split + inference time on THIS machine.

Run on the laptop you will demo on (after copying models/ from Colab):
    python src/evaluate.py                                  # every model in models/
    python src/evaluate.py --models E2 E6-lr1e-4 E7-eff     # selected models

The test split should be used once, for the final numbers: pick your configurations
on the validation results (results/runs.csv) first, then run this.

Outputs:
    results/test_results.csv                     one row per model (the main report table)
    results/eval/<EXP>/classification_report.txt precision / recall / F1 per class
    results/eval/<EXP>/confusion_matrix.png      row-normalised confusion matrix
    results/eval/<EXP>/errors.png                12 misclassified test images
    results/figures/accuracy_vs_latency.png      macro-F1 vs CPU latency for all models
"""
import argparse
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score, precision_score, recall_score,
                             top_k_accuracy_score)
from tensorflow import keras  # noqa: E402

from config import CLASSES, IMG_SIZE, MODELS, NUM_CLASSES, RESULTS  # noqa: E402
from data import load_split, make_dataset  # noqa: E402


def measure_latency(model, runs=100, warmup=10):
    """Batch-size-1 latency of a compiled forward pass, in milliseconds."""
    infer = tf.function(lambda x: model(x, training=False))
    x = tf.random.uniform((1,) + IMG_SIZE + (3,), 0, 255)
    for _ in range(warmup):
        infer(x).numpy()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        infer(x).numpy()
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


def plot_confusion(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES), normalize="true")
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", values_format=".2f",
                                                           colorbar=False)
    ax.set_title(title, fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_errors(df, y_true, y_pred, probs, path, n=12):
    wrong = np.nonzero(y_true != y_pred)[0]
    if len(wrong) == 0:
        return
    # most confident mistakes first - the most informative ones
    wrong = wrong[np.argsort(-probs[wrong, y_pred[wrong]])][:n]
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, i in zip(np.ravel(axes), wrong):
        img = tf.io.decode_jpeg(tf.io.read_file(df.abs_path.iat[i]), channels=3).numpy()
        ax.imshow(img)
        ax.set_title(f"true: {CLASSES[y_true[i]]}\npred: {CLASSES[y_pred[i]]} "
                     f"({probs[i, y_pred[i]]:.0%})", fontsize=8)
    fig.suptitle("Most confident test-set errors", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_tradeoff(res):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(res.latency_ms, res.test_macro_f1, s=40)
    for r in res.itertuples():
        ax.annotate(r.model, (r.latency_ms, r.test_macro_f1), textcoords="offset points",
                    xytext=(5, 4), fontsize=8)
    ax.set_xlabel("CPU latency per image, batch 1 (ms) - this laptop")
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Accuracy vs speed", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "figures" / "accuracy_vs_latency.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="model names without .keras (default: all)")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--runs", type=int, default=100, help="latency repetitions")
    args = ap.parse_args()

    names = args.models or sorted(p.stem for p in MODELS.glob("*.keras") if "quick" not in p.stem)
    if not names:
        raise SystemExit(f"No models found in {MODELS} - copy them from Colab first")

    df = load_split(args.split)
    ds = make_dataset(args.split)
    y_true = df.y.to_numpy()
    rows = []

    for name in names:
        path = MODELS / f"{name}.keras"
        print(f"\n=== {name} ===")
        model = keras.models.load_model(path)
        probs = model.predict(ds, verbose=0)
        y_pred = probs.argmax(axis=1)

        out = RESULTS / "eval" / name
        out.mkdir(parents=True, exist_ok=True)
        report = classification_report(y_true, y_pred, labels=range(NUM_CLASSES), target_names=CLASSES,
                                       digits=3, zero_division=0)
        (out / "classification_report.txt").write_text(report)
        print(report)
        plot_confusion(y_true, y_pred, out / "confusion_matrix.png", f"{name} - {args.split} set")
        plot_errors(df, y_true, y_pred, probs, out / "errors.png")

        lat_med, lat_p95 = measure_latency(model, runs=args.runs)
        print(f"latency: median {lat_med:.1f} ms, p95 {lat_p95:.1f} ms  (~{1000 / lat_med:.0f} images/s)")

        per_class = f1_score(y_true, y_pred, average=None, labels=range(NUM_CLASSES), zero_division=0)
        rows.append({
            "model": name,
            "test_acc": round(accuracy_score(y_true, y_pred), 4),
            "test_macro_precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "test_macro_recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "test_macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
            "test_top2_acc": round(top_k_accuracy_score(y_true, probs, k=2, labels=range(NUM_CLASSES)), 4),
            **{f"f1_{c}": round(float(f), 3) for c, f in zip(CLASSES, per_class)},
            "latency_ms": round(lat_med, 1),
            "latency_p95_ms": round(lat_p95, 1),
            "params_m": round(model.count_params() / 1e6, 2),
        })
        keras.backend.clear_session()

    res = pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False)
    out_csv = RESULTS / ("test_results.csv" if args.split == "test" else "val_results.csv")
    res.to_csv(out_csv, index=False)
    plot_tradeoff(res)
    print("\n", res[["model", "test_acc", "test_macro_f1", "f1_trash", "latency_ms"]].to_string(index=False),
          sep="")
    print(f"\nSaved {out_csv}, per-model reports in {RESULTS / 'eval'}, "
          f"and {RESULTS / 'figures' / 'accuracy_vs_latency.png'}")


if __name__ == "__main__":
    main()
