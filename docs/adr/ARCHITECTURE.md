---
doc_type: adr
domain: architecture
stack: [python, opencv, mediapipe, scikit-learn, pytorch, torchvision]
node_id: "adr:architecture"
tags: [architecture, computer-vision, machine-learning, libras]
edges:
  - relation: references
    target: "adr:tests"
updated: 2026-11-04
---
# Project Architecture

## OVERVIEW

Current repository implements guided MediaPipe capture, Phase A RBF-SVM training, and local webcam inference through `demo.py`. Phase C EfficientNet training remains planned.

## FOLDER STRUCTURE

```
libras-cv/
├── collect_data.py        # Implemented webcam collection entrypoint
├── requirements.txt       # Python dependency manifest
├── data/                  # Local ignored datasets and personal crops; do not track
├── models/                # Local ignored model artifacts; do not track
├── scripts/               # Add one-off developer utilities here
├── src/                   # Add shared feature extraction and model utilities here
├── notebooks/             # Add training and evaluation experiments here
└── docs/                  # Add project memory under adr/ or feature/
```

## LAYERS

- **Data layer** (`data/`): Local datasets, crop JPEGs, and landmark sidecars. PROHIBITED: committing image data to git.
- **Collection layer** (`collect_data.py`): Implemented MediaPipe Tasks capture. Saves a clean hand crop and paired raw landmark JSON.
- **Training layer** (`src/`, `notebooks/`): Phase A landmark extraction, wrist normalization, stratified SVM evaluation, and model saving implemented. Phase C EfficientNet training remains planned.
- **Inference layer** (`demo.py`): Implemented local CPU inference; uses wrist-normalized MediaPipe landmarks and the saved RBF-SVM artifact.
- **Shared utilities** (`src/`): Landmark sidecar handling, wrist normalization, dataset loading, SVM evaluation, and serialization.

## MODULES

| Module | Responsibility | Location and status |
|---|---|---|
| collect_data | Webcam-guided crop collection | `collect_data.py` — implemented |
| landmarks_svm | Landmark extraction, wrist normalization, SVM training | `src/landmark_svm.py`, `notebooks/02_landmarks_svm.ipynb` — implemented |
| cnn_efficientnet | EfficientNet fine-tuning and evaluation | `notebooks/03_cnn_efficientnet.ipynb` — planned, absent |
| demo | Live inference and OpenCV HUD | `demo.py` — implemented |

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
| MediaPipe Tasks Hand Landmarker | Hand detection + 21-point 3D landmark extraction | IMAGE mode for extraction and VIDEO mode for webcam inference; model cached in `~/.cache/libras-recognition/` |
| Kaggle CLI | Download `cnn-libras` dataset | Command documented in README; no project loader is present |
| Google Colab | Planned GPU training for EfficientNet-B0 (Phase C) | Colab T4; training notebook absent |
| OpenCV VideoCapture | Webcam frame read + HUD rendering | Device index (`CAMERA_INDEX=0`) |
| scikit-learn SVM | RBF classifier on wrist-relative landmark vectors | Local training utility; model saved to ignored `models/svm_model.pkl` |
| PyTorch / torchvision | Planned EfficientNet-B0 fine-tuning | Dependencies declared; train on Colab T4; notebook and weights absent |

<!-- DOCUMENT MAP: omitted — this baseline ADR has exactly 1 edge. The ## REFERENCES section below carries the relation. -->

## REFERENCES

- [**README.md**](../README.md): Main documentation index.
- [**TESTS.md**](./TESTS.md): Testing strategies and commands.
