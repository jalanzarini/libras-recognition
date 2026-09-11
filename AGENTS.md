# AGENTS.md — libras-cv

Guidelines for AI agents working on this project. Read this before making any change.

---

## QUICK ORIENTATION

1. Read `docs/.digest.md` — stack, constraints, test commands.
2. Read `docs/.graph.json` — document topology, 1-hop routing.
3. Read `docs/adr/ARCHITECTURE.md` — layers, patterns, integrations.
4. Read `docs/adr/TESTS.md` — testing strategy and commands.

Do not read source files until the above orient you to what is needed.

---

## PROJECT SUMMARY

Real-time LIBRAS static alphabet recognizer (21 letters; J and Z are motion-based and excluded from Phase A). Two pipeline phases:

| Phase | Input → Pipeline → Output | Where trained | Where runs |
|---|---|---|---|
| A | Webcam → MediaPipe 21-pt landmarks → wrist-normalize → SVM | Local CPU (`notebooks/02`) | Local CPU |
| C | Webcam → MediaPipe bbox crop → EfficientNet-B0 | Google Colab T4 (`notebooks/03`) | Local CPU |

---

## FILE MAP — WHERE TO PUT NEW CODE

| What you're adding | Where it goes |
|---|---|
| Shared utility function (normalization, augmentation, loaders) | `src/` |
| Data collection change | `collect_data.py` |
| Live inference change | `demo.py` |
| Training experiment | `notebooks/` |
| New ADR or architectural decision | `docs/adr/` |
| Feature-level documentation | `docs/feature/` |
| One-off utility or dev script | `scripts/` |

PROHIBITED: Adding business logic directly into notebooks without a corresponding utility in `src/`.
PROHIBITED: Adding any file to `data/`, `models/` — they are gitignored and managed locally.

---

## CODING CONSTRAINTS

### Landmark normalization — REQUIRED
```python
# CORRECT: wrist-relative normalization (makes SVM handedness-agnostic)
wrist = landmarks[0]
features = [(lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z)
            for lm in landmarks]
features = np.array(features).flatten()

# WRONG: raw coordinates — breaks on left/right hand flip
features = np.array([(lm.x, lm.y, lm.z) for lm in landmarks]).flatten()
```

### Hand-crop saving — REQUIRED
```python
# CORRECT: save the MediaPipe-bounded hand crop (with CROP_PADDING)
crop = frame[y_min:y_max, x_min:x_max]
cv2.imwrite(path, crop)

# WRONG: save the full webcam frame
cv2.imwrite(path, frame)
```

### Camera mock in tests — REQUIRED
```python
# CORRECT: patch VideoCapture and MediaPipe in tests
from unittest.mock import patch, MagicMock
with patch("cv2.VideoCapture") as mock_cap:
    mock_cap.return_value.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    ...

# WRONG: open a real camera in automated tests
cap = cv2.VideoCapture(0)  # PROHIBITED in pytest
```

### EfficientNet training — REQUIRED
- REQUIRED: Train `notebooks/03_cnn_efficientnet.ipynb` on **Google Colab (T4)**.
- PROHIBITED: Running PyTorch EfficientNet training on local AMD GPU — ROCm instability causes silent failures.

### Data augmentation for CNN — REQUIRED
- REQUIRED: Include horizontal flip in training augmentation (supports both hands).
- PROHIBITED: Applying horizontal flip to SVM training data — landmark normalization already handles handedness.

---

## GIT CONVENTIONS

### Commit message format
```
<type>(<scope>): <short description>

Types: feat, fix, docs, refactor, test, chore
Scope: optional — collect, demo, svm, cnn, src, docs, scripts

# Examples:
feat(svm): add wrist-relative normalization to landmark extractor
fix(collect): handle degenerate hand bbox in get_hand_crop
docs: update ARCHITECTURE.md with Phase C integrations
chore: add pyyaml to requirements.txt
```

### Gitignore rules — never commit
- `data/cnn-libras/` — Kaggle dataset (download via `kaggle datasets download`)
- `data/personal/` — personal recordings
- `models/*.pkl`, `models/*.pt` — serialized model files
- `__pycache__/`, `.ipynb_checkpoints/`, `venv/`, `.venv/`

### Files that MUST be tracked
- `docs/.digest.md` and `docs/.graph.json` — machine-readable project memory indexes.

---

## TESTING

```bash
# Run unit tests for src/ utilities
pytest src/

# With coverage
pytest --cov=src --cov-report=term-missing src/

# Visual smoke test (requires webcam)
python demo.py
```

Minimum coverage targets for `src/`:
- Global: 70%
- Landmark normalization module: 80%

---

## DOCUMENTATION UPDATES

After adding or removing a `docs/adr/` or `docs/feature/` file, regenerate the graph:

```bash
python scripts/generate_docs_graph.py docs
```

Then update `docs/.digest.md` manually with any new constraints or stack changes.
REQUIRED: Run the graph regeneration in the same commit as the documentation change.

---

## LETTERS COVERED

```
Phase A (static):  A B C D E F G I L M N O P Q R S T U V W Y  (21 letters)
Phase 2 (future):  J Z  (require motion — out of scope)
```

Dataset primary source: [cnn-libras](https://www.kaggle.com/datasets/williansoliveira/cnn-libras) (Kaggle, 21 classes).
Personal recordings: `data/personal/<LETTER>/frame_XXXX.jpg` via `collect_data.py`.

---

## CONTROLS — collect_data.py

| Key | Action |
|---|---|
| `SPACE` | Capture current frame |
| `n` | Skip to next letter |
| `r` | Restart current letter (discard this run's frames) |
| `q` | Quit |

Resume from a specific letter: `python collect_data.py G`
