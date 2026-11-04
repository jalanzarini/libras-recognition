---
doc_type: adr
domain: testing
stack: [pytest, scikit-learn, opencv-python, mediapipe]
node_id: "adr:tests"
tags: [testing, unit-tests, validation, accuracy]
edges: []
updated: 2026-11-04
---
# Testing Protocol

## OVERVIEW

`src/test_landmark_svm.py` covers landmark loading, wrist normalization, and SVM evaluation; `src/test_demo.py` covers prediction, no-hand frames, and webcam resource cleanup. No CI configuration is present; the local environment has no pytest installation.

## COMMANDS

| Type | Command | Status |
|---|---|---|
| Unit | `.venv/bin/python -m unittest discover -s src -p 'test_*.py' -v` | Verified local runner |
| Project guidance | `pytest src/` | Listed in `AGENTS.md`; pytest is not installed in the current virtualenv |
| Model accuracy (Phase A) | `notebooks/02_landmarks_svm.ipynb` | Run notebook cells; evaluation uses a stratified 20% holdout |
| Manual visual | `python demo.py` | Implemented; requires a webcam and saved model artifact |
| Coverage | `pytest --cov=src --cov-report=term-missing src/` | Listed in `AGENTS.md`; pytest-cov is not declared or installed |

## MINIMUM COVERAGE

The following are **targets stated in `AGENTS.md`**, not configured gates or measured results:

| Layer | Coverage | Description |
|---|---|---|
| Landmark normalization | 80% | Wrist-relative transform and edge cases |
| Hand-crop utility | 70% | Bounding-box extraction, degenerate-bbox fallback |
| Global (`src/`) | 70% | Average across all shared modules |

## PATTERNS & BEST PRACTICES

REQUIRED: Use Arrange-Act-Assert (AAA) structure in every unit test.
REQUIRED: Mock `cv2.VideoCapture` and `mp.tasks.vision.HandLandmarker` in automated tests; never open a real camera.
REQUIRED: Evaluate model accuracy on a stratified held-out split (20% of total samples per class).
REQUIRED: Report per-class accuracy via confusion matrix for the class report.
FORBIDDEN: Hardcoding expected pixel values or landmark coordinates as test assertions — use tolerance-based comparisons (`numpy.testing.assert_allclose`).
FORBIDDEN: Tests that depend on execution order or share mutable global state.

## TOOLING

- **Framework:** Python `unittest` in the current suite; pytest also named in `AGENTS.md`
- **Assertions:** pytest built-in + `numpy.testing`
- **Mocks/Stubs:** `unittest.mock` for OpenCV and MediaPipe Tasks
- **Coverage:** pytest-cov command documented in `AGENTS.md`; not declared in `requirements.txt`
- **CI Integration:** No CI configuration found

## TROUBLESHOOTING

- **Camera not opening in tests:** Confirm `cv2.VideoCapture` is patched; real camera access is PROHIBITED in automated tests.
- **MediaPipe import errors:** Install dependencies from `requirements.txt`; collector uses `mp.tasks`.
- **Debug mode:** `pytest -s -v src/` is documented in `AGENTS.md`; it requires the planned `src/` tests.

## REFERENCES

- [**README.md**](../README.md): Main documentation index.
