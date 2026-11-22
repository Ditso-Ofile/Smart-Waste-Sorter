# Smart Waste Sorter

Real-time 6-class waste classification (cardboard, glass, metal, paper, plastic, trash)
with transfer learning on TrashNet and an OpenCV webcam app.

## Setup (laptop)

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_env.py
```

## Setup (Google Colab, for training)

```python
!git clone https://github.com/<your-team>/waste-sorter.git
%cd waste-sorter
!pip install -q -r requirements.txt
!python scripts/check_env.py --no-webcam
```

Runtime -> Change runtime type -> GPU (T4) before running.

## Layout

| Path | Contents |
| --- | --- |
| `src/config.py` | Paths, class order, image size, seed |
| `src/utils.py` | Seeding, run logging, timing |
| `src/data.py` | Loading + augmentation (Step 3) |
| `src/train.py` | Training, strategy chosen by CLI flags (Steps 4-5) |
| `src/evaluate.py` | Metrics, confusion matrix, latency (Step 6) |
| `app/webcam.py` | Real-time OpenCV app (Step 7) |
| `data/` | Raw TrashNet, splits, webcam test set (not committed) |
| `models/` | Saved models (not committed) |
| `results/` | `runs.csv` experiment log, plots |

## Team

- Person A - models, training, evaluation
- Person B - webcam app, webcam test set, domain-shift study
