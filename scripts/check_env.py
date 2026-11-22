"""Step 1 sanity check. Run:  python scripts/check_env.py [--no-webcam]

Checks Python + library versions, GPU visibility, that MobileNetV2 /
EfficientNet-B0 ImageNet weights download, and that the webcam opens.
"""
import argparse
import sys
import time


def ok(msg):   print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-webcam", action="store_true", help="skip webcam test (e.g. on Colab)")
    ap.add_argument("--no-weights", action="store_true", help="skip pretrained-weights download")
    args = ap.parse_args()
    problems = 0

    print("\n1. Python")
    v = sys.version_info
    if (3, 10) <= (v.major, v.minor) <= (3, 13):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        warn(f"Python {v.major}.{v.minor} - check tensorflow.org/install/pip for supported versions (3.11 is the safe choice)")

    print("\n2. Libraries")
    for name in ["numpy", "pandas", "sklearn", "matplotlib", "seaborn", "cv2", "tensorflow"]:
        try:
            mod = __import__(name)
            ok(f"{name} {getattr(mod, '__version__', '')}")
        except ImportError:
            fail(f"{name} not installed -> pip install -r requirements.txt")
            problems += 1

    print("\n3. GPU")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            ok(f"{len(gpus)} GPU(s): {[g.name for g in gpus]}")
        else:
            warn("No GPU - fine for the webcam app, slow for training (use Colab/Kaggle)")
    except ImportError:
        pass

    if not args.no_weights:
        print("\n4. Pretrained weights (downloads ~14 MB + ~16 MB once)")
        try:
            import numpy as np
            import tensorflow as tf
            for name, ctor in [("MobileNetV2", tf.keras.applications.MobileNetV2),
                               ("EfficientNetB0", tf.keras.applications.EfficientNetB0)]:
                m = ctor(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
                x = np.zeros((1, 224, 224, 3), dtype="float32")
                m(x, training=False)  # warm-up
                t = time.perf_counter()
                for _ in range(10):
                    m(x, training=False)
                ms = (time.perf_counter() - t) / 10 * 1000
                ok(f"{name}: {m.count_params()/1e6:.1f}M params, ~{ms:.0f} ms/image on this machine")
        except Exception as e:  # noqa: BLE001
            fail(f"Could not load weights: {e}")
            problems += 1

    if not args.no_webcam:
        print("\n5. Webcam (a window opens for ~3 s; press q to close early)")
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                fail("Camera 0 did not open - try index 1, or check OS camera permissions")
                problems += 1
            else:
                frames, t0 = 0, time.time()
                while time.time() - t0 < 3:
                    good, frame = cap.read()
                    if not good:
                        break
                    frames += 1
                    cv2.imshow("check_env - webcam", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                h, w = frame.shape[:2] if frames else (0, 0)
                cap.release()
                cv2.destroyAllWindows()
                if frames:
                    ok(f"Webcam {w}x{h}, ~{frames / (time.time() - t0):.0f} FPS capture")
                else:
                    fail("Camera opened but returned no frames")
                    problems += 1
        except Exception as e:  # noqa: BLE001
            fail(f"Webcam error: {e}")
            problems += 1

    print("\n" + ("All checks passed." if problems == 0 else f"{problems} problem(s) - fix before Step 2."))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
