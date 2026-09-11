# libras-cv — LIBRAS Static Alphabet Recognition

Real-time Brazilian Sign Language (LIBRAS) static alphabet recognizer using Python and computer vision.

## Project Phases

| Phase | Approach | Status |
|---|---|---|
| A | MediaPipe hand landmarks → SVM | 🔜 In progress |
| C | MediaPipe crop → EfficientNet-B0 (Colab) | ⏳ Planned |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Collection

```bash
# Record all letters (50 samples each, ~20 min)
python collect_data.py

# Resume from a specific letter
python collect_data.py G
```

## Dataset (cnn-libras)

```bash
# Requires Kaggle API key (~/.kaggle/kaggle.json)
kaggle datasets download -d williansoliveira/cnn-libras -p data/cnn-libras --unzip
```

## Notebooks

| Notebook | Description |
|---|---|
| `notebooks/01_exploration.ipynb` | EDA, sample visualization |
| `notebooks/02_landmarks_svm.ipynb` | Phase A: landmark extraction + SVM |
| `notebooks/03_cnn_efficientnet.ipynb` | Phase C: EfficientNet fine-tuning (run on Colab) |

## Live Demo

```bash
python demo.py
```

## Letters Covered

`A B C D E F G I L M N O P Q R S T U V W Y` — 21 static signs (J and Z require motion; Phase 2).

## Controls (collect_data.py)

| Key | Action |
|---|---|
| `SPACE` | Capture frame |
| `n` | Skip to next letter |
| `r` | Restart current letter |
| `q` | Quit |
