"""Step 7 - real-time webcam classifier (+ Step 8 test-image capture).

Run from the project root:
    python app/webcam.py                          # default model (config.BEST_MODEL)
    python app/webcam.py --model E4-lr1e-4        # another model, by name or path
    python app/webcam.py --condition C2 --label glass   # S saves to data/webcam_test/C2/glass/

The model receives the square guide-box crop as float32 RGB 0-255, resized to 224x224:
the same geometry and pixel range as training (src/data.py). Scaling to the
backbone's range happens inside the saved model.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import BEST_MODEL, CLASSES, IMG_SIZE, MODELS, WEBCAM_TEST  # noqa: E402

WINDOW = "Smart Waste Sorter"

GREEN = (80, 220, 100)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
DARK = (30, 30, 30)
ORANGE = (0, 165, 255)


def draw_text(frame, text, x, y, color=WHITE, scale=0.6):
    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, 2, cv2.LINE_AA
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BEST_MODEL,
                        help="model name in models/ (e.g. E4-lr1e-4) or a path to a .keras file")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--condition", choices=[
        "C1", "C2", "C3", "C4", "C5", "C6"
    ], default="C1")
    parser.add_argument("--label", choices=CLASSES)
    args = parser.parse_args()

    model_path = Path(args.model)
    if model_path.suffix != ".keras":
        model_path = MODELS / f"{args.model}.keras"

    model = None

    if model_path.exists():
        import tensorflow as tf
        keras_model = tf.keras.models.load_model(
            model_path, compile=False
        )
        # Compiled forward pass: much faster per frame than eager calls.
        model = tf.function(lambda x: keras_model(x, training=False))
        print("Model loaded:", model_path)

        # The class order saved at training time must match config.CLASSES.
        classes_file = model_path.with_name(f"{model_path.stem}_classes.json")
        if classes_file.exists():
            saved = json.loads(classes_file.read_text())
            if saved != CLASSES:
                sys.exit(f"Class order mismatch: model {saved} vs config {CLASSES}")
    else:
        print(f"Camera-only mode: {model_path} not found.")

    history = deque(maxlen=8)
    fps_history = deque(maxlen=30)

    # DirectShow opens much faster and more reliably on Windows (same as check_env.py).
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)

    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    previous = time.perf_counter()
    saved = 0
    message = ""
    message_until = 0

    print("Q: Quit | S: Save image")

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("Cannot read camera frame.")
                break

            frame = cv2.resize(frame, (960, 540))
            h, w = frame.shape[:2]

            size = int(h * 0.65)
            x = (w - 280 - size) // 2
            y = (h - size) // 2

            roi = frame[y:y+size, x:x+size].copy()

            label = "Waiting for model"
            confidence = 0.0
            latency = 0.0
            probabilities = None

            if model is not None:
                # INTER_AREA = anti-aliased downscaling, matching tf.image.resize(antialias=True)
                image = cv2.resize(roi, IMG_SIZE, interpolation=cv2.INTER_AREA)
                image = cv2.cvtColor(
                    image, cv2.COLOR_BGR2RGB
                ).astype(np.float32)

                start = time.perf_counter()

                prediction = model(image[None]).numpy()[0]

                latency = (
                    time.perf_counter() - start
                ) * 1000

                if prediction.shape != (len(CLASSES),):
                    raise ValueError("Unexpected model output shape")

                history.append(prediction)
                probabilities = np.mean(history, axis=0)

                index = int(np.argmax(probabilities))
                confidence = float(probabilities[index])

                if confidence >= args.threshold:
                    label = CLASSES[index].upper()
                else:
                    label = "UNCERTAIN"

            now = time.perf_counter()
            fps_history.append(
                1 / max(now - previous, 0.000001)
            )
            previous = now
            fps = np.mean(fps_history)

            color = GREEN if (
                model is not None
                and confidence >= args.threshold
            ) else ORANGE

            # Object guide box
            cv2.rectangle(
                frame, (x, y),
                (x+size, y+size), color, 2
            )

            draw_text(
                frame, "Place waste inside the box",
                x, y-15, WHITE, 0.55
            )

            # Header
            cv2.rectangle(frame, (0, 0), (w, 52), DARK, -1)
            draw_text(
                frame, "SMART WASTE SORTER",
                20, 35, GREEN, 0.9
            )

            # Right-side information panel
            panel_x = w - 280

            cv2.rectangle(
                frame, (panel_x, 52), (w, h), DARK, -1
            )

            draw_text(frame, "LIVE CLASSIFICATION",
                      panel_x+15, 85, GRAY, 0.55)

            draw_text(frame, label,
                      panel_x+15, 125, color, 0.7)

            draw_text(frame, f"Confidence: {confidence:.1%}",
                      panel_x+15, 165)

            draw_text(frame, f"FPS: {fps:.1f}",
                      panel_x+15, 205, GREEN)

            draw_text(frame, f"Inference: {latency:.1f} ms",
                      panel_x+15, 240, GRAY, 0.5)

            draw_text(frame, "TOP 3 PREDICTIONS",
                      panel_x+15, 280, WHITE, 0.55)

            if probabilities is not None:
                top3 = np.argsort(probabilities)[-3:][::-1]

                for rank, index in enumerate(top3):
                    probability = float(probabilities[index])
                    bar_y = 310 + rank * 55

                    draw_text(
                        frame,
                        f"{CLASSES[index]}: {probability:.1%}",
                        panel_x+15, bar_y, WHITE, 0.5
                    )

                    cv2.rectangle(
                        frame,
                        (panel_x+15, bar_y+8),
                        (panel_x+245, bar_y+20),
                        (75, 75, 75), -1
                    )

                    cv2.rectangle(
                        frame,
                        (panel_x+15, bar_y+8),
                        (panel_x+15+int(
                            230*np.clip(probability, 0, 1)
                        ), bar_y+20),
                        GREEN, -1
                    )
            else:
                draw_text(
                    frame, "Model not available",
                    panel_x+15, 320, ORANGE, 0.5
                )

            draw_text(frame, f"Saved images: {saved}",
                      panel_x+15, 485, GRAY, 0.5)

            # Footer
            cv2.rectangle(
                frame, (0, h-35), (panel_x, h), DARK, -1
            )

            draw_text(
                frame, "Q: Exit  |  S: Save image",
                15, h-12, WHITE, 0.55
            )

            if time.perf_counter() < message_until:
                draw_text(frame, message,
                          15, h-50, GREEN, 0.55)

            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                if args.label:
                    folder = (
                        WEBCAM_TEST /
                        args.condition /
                        args.label
                    )
                else:
                    folder = WEBCAM_TEST / "unlabelled"

                folder.mkdir(parents=True, exist_ok=True)

                filename = folder / f"{time.time_ns()}.jpg"

                if cv2.imwrite(str(filename), roi):
                    saved += 1
                    message = "Image saved!"
                    print("Saved:", filename)
                else:
                    message = "Failed to save image"

                message_until = time.perf_counter() + 2

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
