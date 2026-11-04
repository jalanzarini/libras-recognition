# libras-cv — LIBRAS Static Alphabet Recognition

Real-time Brazilian Sign Language (LIBRAS) static alphabet recognizer using Python and computer vision.

## Project Phases

| Phase | Approach | Status |
|---|---|---|
| A | MediaPipe hand landmarks → SVM | Implemented; latest personal-data run reached 96.67% holdout accuracy and 0.9666 macro-F1 on 1,050 samples |
| C | MediaPipe crop → EfficientNet-B0 (Colab) | ⏳ Planned |

## Setup

Collector uses the MediaPipe Tasks Hand Landmarker API. On first run, it downloads the model to `~/.cache/libras-recognition/`.
Each successful capture saves a clean hand crop and paired raw landmark JSON sidecar.

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
| `notebooks/01_exploration.ipynb` | Planned; file not present |
| `notebooks/02_landmarks_svm.ipynb` | Phase A landmark extraction, evaluation, and SVM training |
| `notebooks/03_cnn_efficientnet.ipynb` | Planned for Colab; file not present |

## Live Demo

Run the saved Phase A model from a webcam:

```bash
python demo.py
```

Use `--camera 1` to select another camera. Press `q` or `Esc` to quit. The model must exist at `models/svm_model.pkl`; the MediaPipe task model is cached under `~/.cache/libras-recognition/`.

## Letters Covered

`A B C D E F G I L M N O P Q R S T U V W Y` — 21 static signs (J and Z require motion; Phase 2).

## Controls (collect_data.py)

| Key | Action |
|---|---|
| `SPACE` | Capture frame |
| `n` | Skip to next letter |
| `r` | Restart current letter |
| `q` | Quit |
