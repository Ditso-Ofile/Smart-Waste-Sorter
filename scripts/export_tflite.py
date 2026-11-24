"""Step 9 - export a trained model to TensorFlow Lite (float32 + dynamic-range int8).

Run from the project root:
    python scripts/export_tflite.py                    # default model (config.BEST_MODEL)
    python scripts/export_tflite.py --model E4-lr1e-4

Writes models/<name>_float32.tflite and models/<name>_dynamic.tflite, then checks
each one against the Keras model on real validation images (agreement of the
predicted class + largest probability difference), so a broken conversion is
caught here and not in the demo.

Keras 3 (TensorFlow 2.16+) models are converted via an inference-only SavedModel
(model.export), because TFLiteConverter.from_keras_model often fails on Keras 3.
The augmentation layers at the start of the model are inactive at inference,
so they export as pass-through.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from config import BEST_MODEL, MODELS  # noqa: E402
from data import make_dataset  # noqa: E402


def convert(saved_model_dir: str, optimized: bool) -> bytes:
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if optimized:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]   # weights -> int8
    return converter.convert()


def tflite_predict(path: Path, images: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    preds = []
    for img in images:
        interpreter.set_tensor(inp["index"], img[None].astype(inp["dtype"]))
        interpreter.invoke()
        preds.append(interpreter.get_tensor(out["index"])[0])
    return np.stack(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BEST_MODEL,
                        help="model name in models/ or a path to a .keras file")
    parser.add_argument("--check-images", type=int, default=64,
                        help="validation images used to compare TFLite with Keras")
    args = parser.parse_args()

    model_path = Path(args.model)
    if model_path.suffix != ".keras":
        model_path = MODELS / f"{args.model}.keras"
    if not model_path.exists():
        sys.exit(f"Trained model not found: {model_path}")

    model = tf.keras.models.load_model(model_path, compile=False)

    # A few real validation images for the correctness check.
    images = np.concatenate([x.numpy() for x, _ in make_dataset("val", batch_size=32)])
    images = images[:args.check_images]
    reference = model.predict(images, verbose=0)

    keras_mb = model_path.stat().st_size / 1024 ** 2
    print(f"Keras model: {model_path.name}  {keras_mb:.2f} MB")

    with tempfile.TemporaryDirectory() as tmp:
        saved_model_dir = str(Path(tmp) / "saved_model")
        model.export(saved_model_dir)                  # inference-only SavedModel

        for name, optimized in [("float32", False), ("dynamic", True)]:
            output = MODELS / f"{model_path.stem}_{name}.tflite"
            output.write_bytes(convert(saved_model_dir, optimized))

            preds = tflite_predict(output, images)
            agree = float(np.mean(preds.argmax(1) == reference.argmax(1)))
            max_diff = float(np.abs(preds - reference).max())
            size = output.stat().st_size / 1024 ** 2
            print(f"Exported: {output.name}  {size:.2f} MB  "
                  f"| same class as Keras on {agree:.1%} of {len(images)} val images, "
                  f"max prob. diff {max_diff:.4f}")
            if agree < 0.95:
                print("  WARNING: TFLite disagrees with Keras on >5% of images - check the conversion.")

    print("Export completed. Benchmark with: python scripts/benchmark.py")


if __name__ == "__main__":
    main()
