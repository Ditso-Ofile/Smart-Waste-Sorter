"""Step 9 - speed and size of the trained models (.keras and .tflite) on THIS machine.

Run from the project root, on the laptop you demo on:
    python scripts/benchmark.py                                  # every model in models/
    python scripts/benchmark.py --models models/E6-lr1e-4.keras models/E6-lr1e-4_dynamic.tflite

Writes results/benchmark.csv. Batch size 1, single random image, after warm-up.
TFLite runs on 1 thread (closest to a small device). Keras models run as a compiled
tf.function, the same way src/evaluate.py measures latency.
"""
import argparse
import csv
import sys
import time
import platform
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import MODELS, RESULTS, IMG_SIZE  # noqa: E402


def measure(predict, image, warmup, runs):
    for _ in range(warmup):
        predict(image)

    timings = []

    for _ in range(runs):
        start = time.perf_counter()
        predict(image)

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        timings.append(elapsed)

    return {
        "median_ms": round(
            float(np.median(timings)), 3
        ),
        "p95_ms": round(
            float(np.percentile(timings, 95)), 3
        ),
        "mean_ms": round(
            float(np.mean(timings)), 3
        )
    }


def load_keras(path):
    model = tf.keras.models.load_model(
        path, compile=False
    )

    infer = tf.function(lambda x: model(x, training=False))

    def predict(image):
        return infer(image).numpy()

    return predict


def load_tflite(path):
    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        num_threads=1
    )

    interpreter.allocate_tensors()

    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    def predict(image):
        interpreter.set_tensor(
            input_info["index"],
            image.astype(input_info["dtype"])
        )

        interpreter.invoke()

        return interpreter.get_tensor(
            output_info["index"]
        )

    return predict


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        type=Path
    )

    parser.add_argument(
        "--warmup", type=int, default=20
    )

    parser.add_argument(
        "--runs", type=int, default=200
    )

    args = parser.parse_args()

    if args.runs < 1 or args.warmup < 0:
        parser.error("Invalid benchmark settings")

    paths = args.models or sorted(
        p for p in
        list(MODELS.glob("*.keras")) + list(MODELS.glob("*.tflite"))
        if "quick" not in p.stem              # skip --quick smoke-test models
    )

    if not paths:
        print("No trained models available.")
        return

    rng = np.random.default_rng(42)

    image = rng.uniform(
        0, 255,
        (1, IMG_SIZE[1], IMG_SIZE[0], 3)
    ).astype(np.float32)

    results = []

    print("System:", platform.platform())
    print("Architecture:", platform.machine())
    print("TensorFlow:", tf.__version__)

    for path in paths:
        if not path.exists():
            print("Missing model:", path)
            continue

        print(f"\nBenchmarking: {path.name}")

        if path.suffix == ".keras":
            predict = load_keras(path)

        elif path.suffix == ".tflite":
            predict = load_tflite(path)

        else:
            continue

        metrics = measure(
            predict, image,
            args.warmup, args.runs
        )

        row = {
            "model": path.name,
            "size_mb": round(
                path.stat().st_size / (1024 ** 2), 3
            ),
            **metrics,
            "inferences_per_second": round(
                1000 / metrics["mean_ms"], 2
            ),
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
            "architecture": platform.machine(),
            "system": platform.platform(),
            "tensorflow": tf.__version__
        }

        results.append(row)

        for key, value in row.items():
            print(f"{key}: {value}")

    if results:
        RESULTS.mkdir(parents=True, exist_ok=True)
        output = RESULTS / "benchmark.csv"

        with output.open("w", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=list(results[0])
            )
            writer.writeheader()
            writer.writerows(results)

        print("\nResults saved:", output)


if __name__ == "__main__":
    main()
