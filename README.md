# Smart Waste Sorter

Real-time 6-class waste classification (cardboard, glass, metal, paper, plastic, trash)
with transfer learning on TrashNet and an OpenCV webcam app.

## Setup (laptop)

```bash
python3.12 -m venv .venv          # 3.11 or 3.12; TensorFlow does not support 3.14
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_env.py
```

## Setup (Google Colab, for training)

```python
!git clone https://github.com/Ditso-Ofile/Smart-Waste-Sorter.git
%cd Smart-Waste-Sorter
!pip install -q -r requirements.txt
!python scripts/check_env.py --no-webcam
```

Runtime -> Change runtime type -> GPU (T4) before running.

## Layout

| Path | Contents |
| --- | --- |
| `src/config.py` | Paths, class order, image size, seed, default model (`BEST_MODEL`) |
| `src/utils.py` | Seeding, run logging, timing |
| `src/data.py` | Loading, augmentation, background replacement, class weights (Step 3) |
| `src/train.py` | Training, experiment chosen by `--exp E1..E8` (Steps 4-5) |
| `src/evaluate.py` | Test-set metrics, confusion matrices, error grids, latency (Step 6) |
| `app/webcam.py` | Real-time OpenCV app + webcam test-image capture (Steps 7-8) |
| `scripts/` | Dataset download/exploration/splits, augmentation preview, TFLite export, benchmark |
| `data/` | Raw TrashNet, splits (committed), webcam test set (not committed) |
| `models/` | Saved models (not committed - share via Drive) |
| `results/` | `runs.csv` (validation), `test_results.csv` (test), `eval/`, `history/`, `figures/` |

## Workflow (run from the project root)

```bash
python scripts/download_trashnet.py        # Step 2: data/raw/trashnet
python scripts/explore_data.py             #   stats, figures, near-duplicates
python scripts/make_splits.py              #   data/splits/*.csv (already committed)
python scripts/preview_augmentation.py     # Step 3: augmentation figures
python src/train.py --exp E6 --lr-ft 1e-4 --tag lr1e-4   # Steps 4-5 (GPU / Colab)
python src/evaluate.py --models E6-lr1e-4 E4-lr1e-4      # Step 6: test set + latency
python app/webcam.py                       # Step 7: live demo (default model: BEST_MODEL)
python app/webcam.py --condition C2 --label glass        # Step 8: S saves test images
python scripts/export_tflite.py            # Step 9: .tflite float32 + dynamic int8
python scripts/benchmark.py                #   size + latency of every model
```

## Status (24 Nov 2026)

Best on the TrashNet test split: **E6-lr1e-4** (EfficientNetB0, top-60 fine-tuned, lr 1e-4):
accuracy 0.908, macro-F1 0.886, 18 ms/image on CPU. **E4-lr1e-4** (MobileNetV2, fully
fine-tuned) is close (macro-F1 0.881, best trash F1 0.821) and faster (14 ms).

To do: webcam test set + its evaluation (Step 8), TFLite export + benchmark on the demo
laptop (Step 9), report. Known limitation: in `results/near_duplicates.csv` the
paper class has many false matches (different white sheets on the white background), which
chain into one group of 20 paper images that all landed in the test split.

## Team

- Ditso
- Aws
- Jahongir
