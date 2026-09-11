---
doc_type: adr
domain: testing
stack: [pytest, scikit-learn, opencv-python, mediapipe]
node_id: "adr:tests"
tags: [testing, unit-tests, validation, accuracy]
edges:
  - relation: references
    target: "adr:architecture"
updated: 2026-09-10
---
# Testing Protocol

## OVERVIEW

Class project — functional correctness is the primary quality goal. No automated CI pipeline. Tests focus on data-pipeline integrity, model accuracy on held-out test sets, and visual validation of the live demo.

## COMMANDS

| Type | Command | Description |
|---|---|---|
| Unit | `pytest src/` | Runs unit tests for shared utility functions in `src/` |
| Model accuracy (Phase A) | `python -m pytest notebooks/ -k svm` | Evaluates SVM on the held-out test split |
| Manual visual | `python demo.py` | Live webcam smoke-test — verify HUD renders and predictions are plausible |
| Coverage | `pytest --cov=src --cov-report=term-missing src/` | Coverage report for `src/` utilities |

## MINIMUM COVERAGE

REQUIRED: Maintain the following minimum coverage levels for `src/` utility code:

| Layer | Coverage | Description |
|---|---|---|
| Landmark normalization | 80% | Wrist-relative transform and edge cases |
| Hand-crop utility | 70% | Bounding-box extraction, degenerate-bbox fallback |
| Global (`src/`) | 70% | Average across all shared modules |

## PATTERNS & BEST PRACTICES

REQUIRED: Use Arrange-Act-Assert (AAA) structure in every unit test.
REQUIRED: Mock `cv2.VideoCapture` and `mediapipe.solutions.hands.Hands` in tests that exercise collection or inference logic — never open a real camera in automated tests.
REQUIRED: Evaluate model accuracy on a stratified held-out split (20% of total samples per class).
REQUIRED: Report per-class accuracy via confusion matrix for the class report.
FORBIDDEN: Hardcoding expected pixel values or landmark coordinates as test assertions — use tolerance-based comparisons (`numpy.testing.assert_allclose`).
FORBIDDEN: Tests that depend on execution order or share mutable global state.

## TOOLING

- **Framework:** pytest >= 7.4
- **Assertions:** pytest built-in + `numpy.testing`
- **Mocks/Stubs:** `unittest.mock.patch` for OpenCV and MediaPipe
- **Coverage:** `pytest-cov`; report format: `term-missing`
- **CI Integration:** None — run tests manually before each submission checkpoint

## TROUBLESHOOTING

- **Camera not opening in tests:** Confirm `cv2.VideoCapture` is patched; real camera access is PROHIBITED in automated tests.
- **MediaPipe import errors:** Verify `mediapipe>=0.10.0` is installed in the active venv (`pip show mediapipe`).
- **Debug mode:** `pytest -s -v src/` — shows stdout and verbose test names.

<!-- DOCUMENT MAP: omitted — this baseline ADR has exactly 1 edge. -->

## REFERENCES

- [**README.md**](../README.md): Main documentation index.
- [**ARCHITECTURE.md**](./ARCHITECTURE.md): System architecture and patterns.
