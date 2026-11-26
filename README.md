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
python scripts/capture.py                  # Step 8: capture the webcam test set (6 conditions)
python scripts/eval_webcam.py              #   accuracy per condition -> results/domain_shift.csv
python scripts/export_tflite.py            # Step 9: .tflite float32 + dynamic int8
python scripts/benchmark.py                #   size + latency of every model
```

## Results (v2 splits, near-duplicate threshold r >= 0.98)

| Model | Test acc | Test macro-F1 | F1 trash | CPU latency |
| --- | --- | --- | --- | --- |
| **E6-lr1e-4** (EfficientNetB0, top-60 fine-tuned, lr 1e-4) | 0.900 | 0.884 | 0.769 | 19 ms |
| E4-lr1e-4 (MobileNetV2, fully fine-tuned, lr 1e-4) | 0.890 | 0.877 | 0.780 | 15 ms |

Full tables: `results/runs.csv` (validation), `results/test_results.csv` (test),
`results/domain_shift.csv` (webcam conditions, from `scripts/eval_webcam.py`).
Results on the first split (threshold 0.95, which put a falsely grouped set of 20 paper
images into the test split) are kept in `results/old_splits/` for reference.

## Team

- Ditso
- Aws
- Jahongir
