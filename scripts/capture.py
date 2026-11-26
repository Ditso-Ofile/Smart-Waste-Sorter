import argparse
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "webcam_test"

CLASSES = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

CONDITIONS = {
    "C1": "Bright / White background",
    "C2": "Dim / White background",
    "C3": "Strong backlight / White background",
    "C4": "Bright / Cluttered background",
    "C5": "Hand holding object / Face in frame",
    "C6": "Warm light / Cluttered background"
}

TARGET = 10


def count_images(condition, label):
    folder = OUTPUT / condition / label
    return len(list(folder.glob("*.jpg")))


def total_images():
    return sum(
        count_images(condition, label)
        for condition in CONDITIONS
        for label in CLASSES
    )


def text(frame, value, x, y, color=(255, 255, 255),
         scale=0.55):
    cv2.putText(
        frame, value, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, 2, cv2.LINE_AA
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    condition_index = 0
    class_index = 0
    conditions = list(CONDITIONS)

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print("Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    message = ""
    message_until = 0

    print("1-6: Category | C: Condition")
    print("S: Save image | Q: Quit")

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("Cannot read camera frame.")
                break

            frame = cv2.resize(frame, (960, 540))
            h, w = frame.shape[:2]

            condition = conditions[condition_index]
            label = CLASSES[class_index]

            size = int(h * 0.65)
            x = (w - 280 - size) // 2
            y = (h - size) // 2

            roi = frame[y:y+size, x:x+size].copy()

            count = count_images(condition, label)

            color = (80, 220, 100)

            cv2.rectangle(
                frame, (x, y),
                (x+size, y+size), color, 2
            )

            cv2.rectangle(
                frame, (0, 0), (w, 52),
                (30, 30, 30), -1
            )

            text(
                frame, "SMART WASTE SORTER - DATA CAPTURE",
                15, 33, color, 0.75
            )

            cv2.rectangle(
                frame, (680, 52), (960, 540),
                (30, 30, 30), -1
            )

            text(frame, f"Condition: {condition}",
                 695, 85, color, 0.7)

            description = CONDITIONS[condition]

            text(frame, description[:29],
                 695, 115, scale=0.45)

            text(frame, description[29:],
                 695, 135, scale=0.45)

            text(frame, f"Category: {label.upper()}",
                 695, 175, color, 0.65)

            text(frame, f"Captured: {count}/{TARGET}",
                 695, 210)

            for i, category in enumerate(CLASSES):
                n = count_images(condition, category)

                category_color = (
                    color if n >= TARGET
                    else (200, 200, 200)
                )

                text(
                    frame,
                    f"{i+1}. {category}: {n}/{TARGET}",
                    695, 250 + i*30,
                    category_color
                )

            text(frame, f"Total: {total_images()}/360",
                 695, 465, color)

            cv2.rectangle(
                frame, (0, 505), (680, 540),
                (30, 30, 30), -1
            )

            text(
                frame,
                "1-6: Category | C: Condition | S: Save | Q: Quit",
                10, 528, scale=0.5
            )

            if time.perf_counter() < message_until:
                text(frame, message, 15, 485, color)

            cv2.imshow("Waste Dataset Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key in range(ord("1"), ord("7")):
                class_index = key - ord("1")

            elif key == ord("c"):
                condition_index = (
                    condition_index + 1
                ) % len(conditions)

            elif key == ord("s"):
                folder = OUTPUT / condition / label
                folder.mkdir(parents=True, exist_ok=True)

                filename = folder / f"{time.time_ns()}.jpg"

                if cv2.imwrite(str(filename), roi):
                    message = f"Saved {condition}/{label}"
                    print("Saved:", filename)
                else:
                    message = "Save failed"

                message_until = time.perf_counter() + 1.5

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Total collected images: {total_images()}")


if __name__ == "__main__":
    main()
