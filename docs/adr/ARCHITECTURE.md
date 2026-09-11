---
doc_type: adr
domain: architecture
stack: [python, opencv, mediapipe, scikit-learn, pytorch, torchvision]
node_id: "adr:architecture"
tags: [architecture, computer-vision, machine-learning, libras]
edges:
  - relation: references
    target: "adr:tests"
updated: 2026-09-10
---
# Project Architecture

## OVERVIEW

Flat-script pipeline for real-time LIBRAS static alphabet recognition. Phase A uses MediaPipe hand landmarks fed into an SVM classifier. Phase C fine-tunes EfficientNet-B0 on hand-crop images. Input is live webcam; output is predicted letter with confidence overlay.

## FOLDER STRUCTURE

```
libras-cv/
├── collect_data.py        # Guided webcam data-collection entrypoint
├── demo.py                # Live inference entrypoint (Phase A + C)
├── requirements.txt       # Pinned dependency manifest
├── data/
│   ├── cnn-libras/        # Kaggle dataset (gitignored, 21 static classes)
│   └── personal/          # Self-recorded samples per letter (gitignored)
├── models/                # Serialized SVM (.pkl) and CNN weights (.pt) — gitignored
├── notebooks/             # Jupyter notebooks for training pipelines
│   ├── 01_exploration.ipynb
│   ├── 02_landmarks_svm.ipynb
│   └── 03_cnn_efficientnet.ipynb
├── scripts/               # Utility scripts (graph generation, etc.)
├── docs/
│   ├── adr/               # Architecture Decision Records
│   └── feature/           # Feature-level documentation
└── src/                   # Shared helper modules (landmark utils, augmentation)
```

## LAYERS

- **Data layer** (`data/`): Raw images and personal recordings. PROHIBITED: committing image data to git.
- **Collection layer** (`collect_data.py`): Webcam capture with MediaPipe overlay. Saves hand-crop JPEGs.
- **Training layer** (`notebooks/`): Landmark extraction, SVM fitting, EfficientNet fine-tuning. Run 02 locally; run 03 on Colab.
- **Inference layer** (`demo.py`): Loads saved model, runs MediaPipe, renders OpenCV HUD with top-3 predictions.
- **Shared utilities** (`src/`): Landmark normalization, augmentation helpers, dataset loaders.

## MODULES

| Module | Responsibility | Location |
|---|---|---|
| collect_data | Webcam-guided data collection, hand-crop saving | `collect_data.py` |
| demo | Live real-time inference, OpenCV HUD rendering | `demo.py` |
| exploration | EDA, class distribution, sample visualization | `notebooks/01_exploration.ipynb` |
| landmarks_svm | MediaPipe landmark extraction, wrist normalization, SVM training | `notebooks/02_landmarks_svm.ipynb` |
| cnn_efficientnet | EfficientNet-B0 fine-tuning, data augmentation, evaluation | `notebooks/03_cnn_efficientnet.ipynb` |
| src utilities | Shared landmark utils, normalization, augmentation | `src/` |

## PATTERNS

```python
# REQUIRED: Wrist-relative landmark normalization (handedness-agnostic SVM input)
wrist = landmarks[0]
normalized = [(lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z) for lm in landmarks]

# WRONG: Raw landmark coordinates fed directly to classifier
features = [(lm.x, lm.y, lm.z) for lm in landmarks]  # breaks on left/right hand flip
```

```python
# REQUIRED: Hand-crop saved as JPEG for CNN training (mediapipe bbox + padding)
crop = frame[y_min:y_max, x_min:x_max]
cv2.imwrite(save_path, crop)

# WRONG: Full webcam frame saved — wastes storage, hurts CNN accuracy
cv2.imwrite(save_path, frame)
```

## INTEGRATIONS

| External Service / Component | Purpose | Connection / Authentication Method |
|---|---|---|
| MediaPipe Hands | Hand detection + 21-point 3D landmark extraction | Local Python package (`mediapipe>=0.10.0`) |
| Kaggle API | Download `cnn-libras` dataset | `~/.kaggle/kaggle.json` API key |
| Google Colab | GPU training for EfficientNet-B0 (Phase C) | Upload notebook; T4 runtime |
| OpenCV VideoCapture | Webcam frame read + HUD rendering | Device index (`CAMERA_INDEX=0`) |
| scikit-learn SVM | RBF-kernel SVM classifier on landmark vectors | Local; model serialized to `models/svm_model.pkl` |
| PyTorch / torchvision | EfficientNet-B0 fine-tuning | Local CPU / Colab T4; weights saved to `models/efficientnet_libras.pt` |

<!-- DOCUMENT MAP: omitted — this baseline ADR has exactly 1 edge. The ## REFERENCES section below carries the relation. -->

## REFERENCES

- [**README.md**](../README.md): Main documentation index.
- [**TESTS.md**](./TESTS.md): Testing strategies and commands.
