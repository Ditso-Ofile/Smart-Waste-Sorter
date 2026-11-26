import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "webcam_test"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "domain_shift_figures"

CLASSES = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

CONDITIONS = {
    "C1": "Bright / White",
    "C2": "Dim / White",
    "C3": "Backlight / White",
    "C4": "Bright / Cluttered",
    "C5": "Hand holding object",
    "C6": "Warm / Cluttered"
}


def load_classes(path):
    json_path = path.with_name(
        path.stem + "_classes.json"
    )

    with json_path.open() as file:
        labels = json.load(file)

    if isinstance(labels, dict):
        if "classes" in labels:
            labels = labels["classes"]
        elif "class_names" in labels:
            labels = labels["class_names"]
        elif all(isinstance(v, int) for v in labels.values()):
            labels = sorted(labels, key=labels.get)
        elif all(k.isdigit() for k in labels):
            labels = [
                labels[k] for k in sorted(labels, key=int)
            ]

    if not isinstance(labels, list):
        raise ValueError("Unknown class JSON format")

    if len(labels) != 6 or set(labels) != set(CLASSES):
        raise ValueError("Invalid model class order")

    return labels


def load_dataset(condition, size):
    images = []
    labels = []

    for label in CLASSES:
        folder = DATA / condition / label

        for path in sorted(folder.glob("*")):
            if path.suffix.lower() not in [
                ".jpg", ".jpeg", ".png"
            ]:
                continue

            image = cv2.imread(str(path))

            if image is None:
                print("Unreadable image:", path)
                continue

            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            image = cv2.cvtColor(
                image, cv2.COLOR_BGR2RGB
            )

            images.append(image.astype(np.float32))
            labels.append(label)

    return np.asarray(images), labels


def evaluate(model_path, threshold):
    print("\nEvaluating:", model_path.name)

    model = tf.keras.models.load_model(
        model_path, compile=False
    )

    classes = load_classes(model_path)

    height, width = model.input_shape[1:3]

    rows = []
    worst = None

    for condition, description in CONDITIONS.items():
        images, actual = load_dataset(
            condition, (width, height)
        )

        if len(images) == 0:
            print(condition, "- No images, skipping")
            continue

        probabilities = model.predict(
            images, batch_size=16, verbose=0
        )

        predicted = [
            classes[i]
            for i in np.argmax(probabilities, axis=1)
        ]

        confidence = np.max(probabilities, axis=1)
        accepted = confidence >= threshold

        accuracy = accuracy_score(actual, predicted)

        precision = precision_score(
            actual, predicted,
            labels=CLASSES,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            actual, predicted,
            labels=CLASSES,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            actual, predicted,
            labels=CLASSES,
            average="macro",
            zero_division=0
        )

        coverage = float(np.mean(accepted))

        accepted_accuracy = (
            accuracy_score(
                np.asarray(actual)[accepted],
                np.asarray(predicted)[accepted]
            )
            if accepted.any() else None
        )

        row = {
            "model": model_path.stem,
            "condition": condition,
            "description": description,
            "images": len(actual),
            "classes_present": len(set(actual)),
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
            "mean_confidence": float(
                np.mean(confidence)
            ),
            "confidence_threshold": threshold,
            "coverage": coverage,
            "accepted_accuracy": accepted_accuracy
        }

        rows.append(row)

        print(
            f"{condition}: "
            f"{len(actual)} images | "
            f"Accuracy: {accuracy:.1%} | "
            f"F1: {f1:.1%}"
        )

        if len(set(actual)) < 6:
            print("Warning: incomplete class coverage")

        cm = confusion_matrix(
            actual, predicted,
            labels=CLASSES
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=CLASSES
        ).plot(
            ax=ax,
            cmap="Blues",
            xticks_rotation=45
        )

        ax.set_title(
            f"{model_path.stem} - {condition}"
        )

        fig.tight_layout()

        fig.savefig(
            FIGURES /
            f"{model_path.stem}_{condition}.png",
            dpi=150
        )

        plt.close(fig)

        if worst is None or accuracy < worst[1]:
            worst = (condition, accuracy)

    if worst:
        print("Worst condition:", worst[0])
        print("Accuracy:", f"{worst[1]:.1%}")

    return rows


def comparison_chart(df, metric, title, filename):
    table = df.pivot(
        index="condition",
        columns="model",
        values=metric
    ) * 100

    ax = table.plot(
        kind="bar", figsize=(10, 6)
    )

    ax.set_title(title)
    ax.set_xlabel("Testing condition")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.25)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models", nargs="+",
        type=Path
    )

    parser.add_argument(
        "--threshold", type=float,
        default=0.6
    )

    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("Threshold must be between 0 and 1")

    paths = args.models or sorted(
        MODELS.glob("*.keras")
    )

    if not paths:
        print("No trained models found.")
        return

    images = [
        p for p in DATA.rglob("*")
        if p.suffix.lower() in [
            ".jpg", ".jpeg", ".png"
        ]
    ]

    if not images:
        print("No webcam test images found.")
        print("Run scripts/capture.py first.")
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    rows = []

    for path in paths:
        rows.extend(evaluate(path, args.threshold))

    if not rows:
        print("No valid labelled dataset found.")
        return

    df = pd.DataFrame(rows)

    output = RESULTS / "domain_shift.csv"

    df.to_csv(output, index=False)

    comparison_chart(
        df, "accuracy",
        "Accuracy under different webcam conditions",
        "accuracy_comparison.png"
    )

    comparison_chart(
        df, "macro_f1",
        "Macro-F1 under different webcam conditions",
        "f1_comparison.png"
    )

    comparison_chart(
        df, "coverage",
        "Prediction coverage at confidence threshold",
        "coverage_comparison.png"
    )

    print("\nEvaluation completed.")
    print("CSV:", output)
    print("Figures:", FIGURES)


if __name__ == "__main__":
    main()
