"""Steps 4-5 - train one experiment (feature extraction or fine-tuning).

Run from the project root:
    python src/train.py --exp E2              # full run (use a GPU: Colab/Kaggle)
    python src/train.py --exp E3 --quick      # 1+1 epochs on a few batches: smoke test on a laptop
    python src/train.py --exp E7 --backbone efficientnetb0   # override the backbone

Experiments (see the project guide, Step 5):
    E1  MobileNetV2     feature extraction   no augmentation
    E2  MobileNetV2     feature extraction   standard augmentation          (baseline)
    E3  MobileNetV2     fine-tune top layers standard augmentation
    E4  MobileNetV2     fine-tune all layers standard augmentation
    E5  EfficientNetB0  feature extraction   standard augmentation
    E6  EfficientNetB0  fine-tune top layers standard augmentation
    E7  MobileNetV2*    fine-tune top layers strong aug + background replacement
    E8  = E7 without class weights (imbalance ablation)
    * set --backbone to whichever of E3/E6 did better

Fine-tuning runs train in two phases: (1) frozen backbone, head only, lr 1e-3;
(2) unfreeze the top of the backbone, lr 1e-5. BatchNorm layers stay frozen.

Outputs:
    models/<EXP>.keras                 best model (by val accuracy)
    models/<EXP>_classes.json          class order for the webcam app
    results/history/<EXP>.csv          per-epoch metrics
    results/figures/curves_<EXP>.png   training curves
    results/runs.csv                   one summary row per run (the report table)
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
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from tensorflow import keras  # noqa: E402

from config import CLASSES, IMG_SIZE, MODELS, NUM_CLASSES, RESULTS, SEED  # noqa: E402
from data import build_augmentation, class_weights, make_dataset  # noqa: E402
from utils import log_run, save_class_names, seed_everything  # noqa: E402

EXPERIMENTS = {
    "E1": dict(backbone="mobilenetv2", strategy="feature_extraction", aug="none", bg_prob=0.0, use_cw=True),
    "E2": dict(backbone="mobilenetv2", strategy="feature_extraction", aug="standard", bg_prob=0.0, use_cw=True),
    "E3": dict(backbone="mobilenetv2", strategy="fine_tune_top", aug="standard", bg_prob=0.0, use_cw=True),
    "E4": dict(backbone="mobilenetv2", strategy="fine_tune_all", aug="standard", bg_prob=0.0, use_cw=True),
    "E5": dict(backbone="efficientnetb0", strategy="feature_extraction", aug="standard", bg_prob=0.0, use_cw=True),
    "E6": dict(backbone="efficientnetb0", strategy="fine_tune_top", aug="standard", bg_prob=0.0, use_cw=True),
    "E7": dict(backbone="mobilenetv2", strategy="fine_tune_top", aug="strong", bg_prob=0.5, use_cw=True),
    "E8": dict(backbone="mobilenetv2", strategy="fine_tune_top", aug="strong", bg_prob=0.5, use_cw=False),
}

BACKBONES = {
    # preprocess "tf" = scale 0-255 -> [-1, 1]; EfficientNet rescales internally, so "none".
    "mobilenetv2": dict(ctor=keras.applications.MobileNetV2, preprocess="tf", unfreeze_top=40),
    "efficientnetb0": dict(ctor=keras.applications.EfficientNetB0, preprocess="none", unfreeze_top=60),
}


def build_model(backbone: str, aug_level: str):
    spec = BACKBONES[backbone]
    base = spec["ctor"](include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))
    base.trainable = False

    inputs = keras.Input(IMG_SIZE + (3,), name="image")          # float 0-255, RGB
    x = build_augmentation(aug_level)(inputs)                     # active only in training
    if spec["preprocess"] == "tf":
        x = keras.layers.Rescaling(1 / 127.5, offset=-1.0, name="preprocess")(x)
    x = base(x, training=False)                                   # BatchNorm always in inference mode
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name=f"{backbone}_{aug_level}"), base


def unfreeze(base, top_n=None):
    """Unfreeze the last top_n layers of the backbone (all if None), keeping BatchNorm frozen."""
    base.trainable = True
    cutoff = 0 if top_n is None else len(base.layers) - top_n
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= cutoff and not isinstance(layer, keras.layers.BatchNormalization)


def callbacks(exp, patience, best_so_far=None):
    # best_so_far: phase-1 best val accuracy, so phase 2 only overwrites the file if it improves.
    return [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 2),
                                          min_lr=1e-7, verbose=1),
        keras.callbacks.ModelCheckpoint(str(MODELS / f"{exp}.keras"), monitor="val_accuracy",
                                        mode="max", save_best_only=True,
                                        initial_value_threshold=best_so_far),
    ]


def plot_curves(hist: pd.DataFrame, ft_start, exp, title):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ep = np.arange(1, len(hist) + 1)
    for ax, key, label in [(a1, "accuracy", "Accuracy"), (a2, "loss", "Loss")]:
        ax.plot(ep, hist[key], label="train")
        ax.plot(ep, hist[f"val_{key}"], label="val")
        if ft_start is not None:
            ax.axvline(ft_start + 0.5, color="grey", ls="--", lw=1)
            ax.text(ft_start + 0.7, ax.get_ylim()[1], "fine-tune", va="top", fontsize=8, color="grey")
        ax.set_xlabel("Epoch"); ax.set_title(label)
        ax.spines[["top", "right"]].set_visible(False)
    a1.legend()
    fig.suptitle(f"{exp}: {title}", fontsize=10)
    fig.tight_layout()
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "figures" / f"curves_{exp}.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--backbone", choices=sorted(BACKBONES), help="override the experiment's backbone")
    ap.add_argument("--epochs-head", type=int, default=20, help="max epochs, frozen backbone")
    ap.add_argument("--epochs-ft", type=int, default=20, help="max epochs, fine-tuning phase")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tag", default="", help="suffix for the run name, e.g. --tag lr1e-4 -> E3-lr1e-4")
    ap.add_argument("--quick", action="store_true", help="smoke test: 1+1 epochs on a few batches")
    args = ap.parse_args()

    cfg = dict(EXPERIMENTS[args.exp])
    if args.backbone:
        cfg["backbone"] = args.backbone
    exp = args.exp + (f"-{args.tag}" if args.tag else "") + ("-quick" if args.quick else "")
    if args.quick:
        args.epochs_head, args.epochs_ft = 1, 1

    seed_everything(SEED)
    MODELS.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {exp}: {cfg} ===")
    print("GPUs:", [g.name for g in tf.config.list_physical_devices("GPU")] or "none (CPU)")

    train_ds = make_dataset("train", training=True, bg_prob=cfg["bg_prob"], batch_size=args.batch)
    val_ds = make_dataset("val", batch_size=args.batch)
    if args.quick:
        train_ds, val_ds = train_ds.take(3), val_ds.take(2)
    cw = class_weights() if cfg["use_cw"] else None

    model, base = build_model(cfg["backbone"], cfg["aug"])
    t0 = time.perf_counter()

    # Phase 1: frozen backbone, train the head only
    model.compile(keras.optimizers.Adam(args.lr_head), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head, class_weight=cw,
                   callbacks=callbacks(exp, patience=5), verbose=2)
    hist = pd.DataFrame(h1.history)
    ft_start = None

    # Phase 2: fine-tune (only for fine-tuning strategies)
    if cfg["strategy"] != "feature_extraction":
        top_n = None if cfg["strategy"] == "fine_tune_all" else BACKBONES[cfg["backbone"]]["unfreeze_top"]
        unfreeze(base, top_n)
        n_train = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
        print(f"\nFine-tuning: unfroze {'all' if top_n is None else top_n} backbone layers "
              f"-> {n_train / 1e6:.2f}M trainable params")
        model.compile(keras.optimizers.Adam(args.lr_ft), loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        ft_start = len(hist)
        h2 = model.fit(train_ds, validation_data=val_ds, initial_epoch=ft_start,
                       epochs=ft_start + args.epochs_ft, class_weight=cw,
                       callbacks=callbacks(exp, patience=8, best_so_far=float(hist.val_accuracy.max())),
                       verbose=2)
        hist = pd.concat([hist, pd.DataFrame(h2.history)], ignore_index=True)
    train_min = (time.perf_counter() - t0) / 60

    # Make sure the saved file is the best model, then evaluate it on the validation set.
    model_path = MODELS / f"{exp}.keras"
    if not model_path.exists():
        model.save(model_path)
    best = keras.models.load_model(model_path)
    y_true = np.concatenate([y.numpy() for _, y in val_ds])
    y_pred = best.predict(val_ds, verbose=0).argmax(axis=1)
    val_acc = accuracy_score(y_true, y_pred)
    val_f1 = f1_score(y_true, y_pred, average="macro")
    per_class = f1_score(y_true, y_pred, average=None, labels=range(NUM_CLASSES), zero_division=0)

    save_class_names(exp)
    (RESULTS / "history").mkdir(parents=True, exist_ok=True)
    hist.to_csv(RESULTS / "history" / f"{exp}.csv", index_label="epoch")
    plot_curves(hist, ft_start, exp, f"{cfg['backbone']}, {cfg['strategy']}, aug={cfg['aug']}, "
                                     f"bg={cfg['bg_prob']}, class weights={cfg['use_cw']}")

    total_params = best.count_params()
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    size_mb = model_path.stat().st_size / 1e6
    row = dict(exp_id=exp, backbone=cfg["backbone"], strategy=cfg["strategy"], augmentation=cfg["aug"],
            bg_prob=cfg["bg_prob"], class_weights=cfg["use_cw"], lr_head=args.lr_head,
            lr_ft=args.lr_ft if cfg["strategy"] != "feature_extraction" else "", epochs=len(hist),
            train_minutes=round(train_min, 1), val_acc=round(val_acc, 4), val_macro_f1=round(val_f1, 4),
            params_m=round(total_params / 1e6, 2), trainable_params_m=round(trainable / 1e6, 2),
            model_mb=round(size_mb, 1),
            **{f"f1_{c}": round(float(f), 3) for c, f in zip(CLASSES, per_class)})
    if not args.quick:                       # smoke tests stay out of the report table
        log_run(**row)

    print(f"\n=== {exp} done in {train_min:.1f} min ===")
    print(f"val accuracy {val_acc:.3f} | val macro-F1 {val_f1:.3f}")
    print("per-class F1:", {c: round(float(f), 3) for c, f in zip(CLASSES, per_class)})
    print(f"saved {model_path} ({size_mb:.1f} MB), curves, history, and a row in {RESULTS / 'runs.csv'}")


if __name__ == "__main__":
    main()
