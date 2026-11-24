"""Step 3 - data pipeline: loading, preprocessing, augmentation, class weights.

Design:
  * tf.data loads images from the split CSVs, centre-crops them to a square
    (the webcam app also uses a square crop) and resizes to 224x224.
    Pixels stay in 0-255; each backbone's own preprocessing is added in train.py.
  * Augmentation is a Keras Sequential that becomes the FIRST block of the model.
    Keras applies it only during training, so the saved model can be used
    for inference unchanged.
  * Optional background replacement (for the domain-shift experiment E7) is done
    in tf.data, on training images only: the white TrashNet background is swapped
    for a real photo from data/backgrounds/ or a random synthetic background.

Usage (in train.py):
    from data import make_dataset, build_augmentation, class_weights
    train_ds = make_dataset("train", training=True, bg_prob=0.0)
    val_ds   = make_dataset("val")
    augment  = build_augmentation("standard")
    cw       = class_weights()
"""
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from config import BATCH_SIZE, CLASSES, DATA_SPLITS, IMG_SIZE, ROOT, SEED

AUTOTUNE = tf.data.AUTOTUNE
BACKGROUNDS_DIR = ROOT / "data" / "backgrounds"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_split(name: str) -> pd.DataFrame:
    """Read data/splits/<name>.csv -> DataFrame with absolute path + integer label."""
    df = pd.read_csv(DATA_SPLITS / f"{name}.csv")
    df["abs_path"] = [str(ROOT / p) for p in df.path]
    df["y"] = df.label.map({c: i for i, c in enumerate(CLASSES)})
    if df.y.isna().any():
        raise ValueError(f"Unknown labels in {name}.csv: {set(df.label) - set(CLASSES)}")
    return df


def _decode(path, label):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    # Centre square crop (512x384 -> 384x384), same geometry as the webcam ROI.
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    s = tf.minimum(h, w)
    img = tf.image.crop_to_bounding_box(img, (h - s) // 2, (w - s) // 2, s, s)
    img = tf.image.resize(img, IMG_SIZE, antialias=True)
    return tf.cast(tf.round(img), tf.uint8), label   # uint8 keeps the cache small (~270 MB)


# --------------------------------------------------------------------------- #
# Background replacement (training only, experiment E7)
# --------------------------------------------------------------------------- #
def _load_backgrounds():
    """Load data/backgrounds/*.jpg|png as a (N, 224, 224, 3) float tensor, or None."""
    if not BACKGROUNDS_DIR.exists():
        return None
    files = [p for p in BACKGROUNDS_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not files:
        return None
    imgs = []
    for p in files:
        img = tf.io.decode_image(tf.io.read_file(str(p)), channels=3, expand_animations=False)
        imgs.append(tf.image.resize(img, IMG_SIZE))
    print(f"Loaded {len(imgs)} background photos from {BACKGROUNDS_DIR}")
    return tf.stack(imgs)


def _synthetic_background():
    """Random colour gradient + low-frequency texture + sensor noise."""
    c1 = tf.random.uniform([1, 1, 3], 0.0, 255.0)
    c2 = tf.random.uniform([1, 1, 3], 0.0, 255.0)
    t = tf.linspace(0.0, 1.0, IMG_SIZE[0])
    a = tf.random.uniform([], 0.0, 1.0)
    grad = a * t[None, :, None] + (1.0 - a) * t[:, None, None]          # (224, 224, 1)
    base = c1 * (1.0 - grad) + c2 * grad
    blobs = tf.image.resize(tf.random.uniform([1, 6, 6, 3], -60.0, 60.0), IMG_SIZE, method="bicubic")[0]
    noise = tf.random.normal(IMG_SIZE + (3,), 0.0, 6.0)
    return tf.clip_by_value(base + blobs + noise, 0.0, 255.0)


def _background_mask(img):
    """1 where a pixel looks like the TrashNet studio background, 0 on the object.

    The background colour is estimated from the image border, so the mask adapts
    to the grey/white shade and shading of each photo.
    """
    border = tf.concat([tf.reshape(img[:8], [-1, 3]), tf.reshape(img[-8:], [-1, 3]),
                        tf.reshape(img[:, :8], [-1, 3]), tf.reshape(img[:, -8:], [-1, 3])], axis=0)
    bg_colour = tf.reduce_mean(border, axis=0)
    dist = tf.norm(img - bg_colour, axis=-1)
    mask = tf.cast(dist < 40.0, tf.float32)[None, ..., None]
    mask = tf.nn.avg_pool2d(mask, ksize=7, strides=1, padding="SAME")   # soft edges
    return mask[0]


def _make_bg_replacer(prob):
    real_bgs = _load_backgrounds()

    def replace(img, label):
        img = tf.cast(img, tf.float32)

        def swap():
            mask = _background_mask(img)
            frac = tf.reduce_mean(mask)
            if real_bgs is not None:
                idx = tf.random.uniform([], 0, tf.shape(real_bgs)[0], dtype=tf.int32)
                new_bg = tf.cond(tf.random.uniform([]) < 0.5,
                                 lambda: tf.image.random_flip_left_right(real_bgs[idx]),
                                 _synthetic_background)
            else:
                new_bg = _synthetic_background()
            out = img * (1.0 - mask) + new_bg * mask
            # Skip images where the "object" is itself background-coloured (e.g. a white
            # paper sheet: >90% masked) or where no clear background was found (<15%).
            return tf.cond((frac > 0.15) & (frac < 0.90), lambda: out, lambda: img)

        img = tf.cond(tf.random.uniform([]) < prob, swap, lambda: img)
        return img, label

    return replace


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def make_dataset(split: str, training: bool = False, bg_prob: float = 0.0,
                 batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    """Batched dataset of (float32 image 0-255, int label).

    training=True shuffles (and applies background replacement if bg_prob > 0).
    Geometric/colour augmentation is NOT applied here - it lives in the model
    (see build_augmentation) so it runs on the GPU.
    """
    df = load_split(split)
    ds = tf.data.Dataset.from_tensor_slices((df.abs_path.values, df.y.values.astype("int32")))
    ds = ds.map(_decode, num_parallel_calls=AUTOTUNE).cache()
    if training:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
        if bg_prob > 0:
            ds = ds.map(_make_bg_replacer(bg_prob), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def build_augmentation(level: str = "standard") -> keras.Sequential:
    """Augmentation block placed at the start of the model (active only in training).

    level: "none" | "standard" | "strong"
    """
    L = keras.layers
    if level == "none":
        return keras.Sequential([L.Identity()], name="augment_none")

    layers = [
        L.RandomFlip("horizontal_and_vertical", seed=SEED),
        L.RandomRotation(0.15, fill_mode="reflect", seed=SEED),
        L.RandomZoom((-0.2, 0.2), fill_mode="reflect", seed=SEED),
        L.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=SEED),
        L.RandomBrightness(0.3, value_range=(0, 255), seed=SEED),   # lighting changes
        L.RandomContrast(0.3, seed=SEED),
    ]
    if level == "strong":
        # Colour + sensor-noise augmentation; uses the newer Keras layers when present.
        if hasattr(L, "RandomSaturation"):
            layers.append(L.RandomSaturation(0.3, value_range=(0, 255), seed=SEED))
        if hasattr(L, "RandomHue"):
            layers.append(L.RandomHue(0.05, value_range=(0, 255), seed=SEED))
        # Webcam sensor noise. GaussianNoise only accepts stddev in [0, 1], so scale to 0-1 and back.
        layers += [L.Rescaling(1 / 255), L.GaussianNoise(0.03, seed=SEED), L.Rescaling(255)]
    elif level != "standard":
        raise ValueError(f"unknown augmentation level: {level}")
    return keras.Sequential(layers, name=f"augment_{level}")


def class_weights(split: str = "train") -> dict:
    """Inverse-frequency weights: n_samples / (n_classes * n_class). trash gets ~3x."""
    counts = load_split(split).y.value_counts().reindex(range(len(CLASSES)), fill_value=0)
    w = counts.sum() / (len(CLASSES) * counts.clip(lower=1))
    return {int(i): round(float(v), 3) for i, v in w.items()}


if __name__ == "__main__":
    # Quick self-test: python src/data.py
    print("Class weights:", {CLASSES[k]: v for k, v in class_weights().items()})
    ds = make_dataset("train", training=True, bg_prob=0.5)
    x, y = next(iter(ds))
    print("Batch:", x.shape, x.dtype, "min/max", float(tf.reduce_min(x)), float(tf.reduce_max(x)))
    aug = build_augmentation("strong")
    print("Augmented:", aug(x, training=True).shape)
    print("Train/val/test sizes:", [len(load_split(s)) for s in ("train", "val", "test")])
