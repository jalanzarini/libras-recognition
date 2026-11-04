---
doc_type: feature
domain: landmark-svm
stack: [python, mediapipe, opencv, numpy, scikit-learn, joblib]
node_id: "feature:landmark-svm"
tags: [landmarks, svm, training, collection]
edges:
  - relation: references
    target: "adr:architecture"
    read: must
  - relation: tested_by
    target: "adr:tests"
    read: must
updated: 2026-11-04
---
```graph
{"node_id":"feature:landmark-svm","domain":"landmark-svm","implements":[],"tested_by":["adr:tests"],"entrypoints":["notebooks/02_landmarks_svm.ipynb","demo.py"],"registration_files":[],"reference_files":[],"code_files":["collect_data.py","src/landmark_svm.py"],"test_files":["src/test_demo.py","src/test_landmark_svm.py"],"knowledge":{"schema_version":1,"entities":[{"id":"capability:collect-landmark-samples","type":"capability","label":"Collect hand samples","definition":"Capture a clean hand crop and its MediaPipe landmark coordinates under a letter label.","aliases":[]},{"id":"capability:train-static-svm","type":"capability","label":"Train static-sign SVM","definition":"Use labeled hand samples to extract wrist-relative features, evaluate an RBF SVM, and save the fitted model.","aliases":[]},{"id":"rule:wrist-relative-features","type":"rule","label":"Wrist-relative features","definition":"Subtract landmark zero from all 21 points before flattening SVM features.","aliases":[]},{"id":"contract:sample-sidecar","type":"contract","label":"Image and landmark sidecar","definition":"A crop JPEG is paired by basename with JSON containing 21 raw x, y, z triples.","aliases":[]},{"id":"contract:dataset-result","type":"contract","label":"Landmark dataset result","definition":"Dataset loading returns an N by 63 feature matrix, N labels, and rejected image paths with reasons.","aliases":[]},{"id":"contract:svm-artifact","type":"contract","label":"SVM model artifact","definition":"A joblib file containing the fitted estimator and ordered class labels.","aliases":[]},{"id":"capability:live-static-sign-recognition","type":"capability","label":"Recognize static signs live","definition":"Classify one detected hand from webcam frames with the saved Phase A SVM.","aliases":[]},{"id":"rule:live-resource-cleanup","type":"rule","label":"Live resource cleanup","definition":"Release the webcam, close the MediaPipe landmarker, and close OpenCV windows when the demo exits or fails.","aliases":[]},{"id":"contract:live-inference-session","type":"contract","label":"Live inference session","definition":"Read from the selected webcam, display a predicted class or no-hand prompt, and quit on q or Escape; report camera open failure.","aliases":[]}],"claims":[{"id":"claim:collector-sidecar","subject":"capability:collect-landmark-samples","relation":"exposes","object":"contract:sample-sidecar","statement":"Each successful capture saves a clean hand crop and paired JSON sidecar with raw MediaPipe landmarks.","kind":"observation","status":"supported","evidence":[{"kind":"code","source":"collect_data.py","locator":"collect_letter: copy clean frame, save crop, call save_landmark_points","snapshot":null}],"derived_from":[],"gap":null},{"id":"claim:training-uses-collected-samples","subject":"capability:train-static-svm","relation":"depends_on","object":"capability:collect-landmark-samples","statement":"The training loader reads the letter-labeled JPEGs and prefers each matching landmark sidecar.","kind":"observation","status":"supported","evidence":[{"kind":"code","source":"src/landmark_svm.py","locator":"load_landmark_dataset: class folder traversal and matching sidecar branch","snapshot":null}],"derived_from":[],"gap":null},{"id":"claim:wrist-normalization","subject":"capability:train-static-svm","relation":"constrained_by","object":"rule:wrist-relative-features","statement":"Training validates 21 finite x, y, z points, subtracts point zero from every point, and flattens to 63 values.","kind":"observation","status":"supported","evidence":[{"kind":"code","source":"src/landmark_svm.py","locator":"normalize_landmarks validation and points minus points[0]","snapshot":null}],"derived_from":[],"gap":null},{"id":"claim:loader-result-contract","subject":"capability:train-static-svm","relation":"exposes","object":"contract:dataset-result","statement":"When sidecars are absent, the loader removes the collector overlay and runs MediaPipe detection; unreadable or undetected images are returned as rejected path/reason pairs.","kind":"observation","status":"supported","evidence":[{"kind":"code","source":"src/landmark_svm.py","locator":"load_landmark_dataset fallback, rejected list, and LandmarkDataset return","snapshot":null}],"derived_from":[],"gap":null},{"id":"claim:evaluation-and-save","subject":"capability:train-static-svm","relation":"exposes","object":"contract:svm-artifact","statement":"Evaluation uses a stratified 20% holdout and per-class metrics; final fit uses StandardScaler with RBF SVC and saves estimator plus class labels.","kind":"observation","status":"supported","evidence":[{"kind":"code","source":"src/landmark_svm.py","locator":"evaluate_svm, _make_svm, fit_svm, save_svm_model","snapshot":null}],"derived_from":[],"gap":null},{"id":"claim:legacy-training-result","subject":"capability:train-static-svm","relation":null,"object":null,"statement":"One run accepted 920 legacy crops, rejected 130, and measured 0.8913 holdout accuracy across 21 classes; class C had 18 accepted samples.","kind":"observation","status":"supported","evidence":[{"kind":"execution","source":"execution:exec_command-session-48641","locator":"stdout: feature matrix, class counts, rejected count, holdout accuracy","snapshot":null}],"derived_from":[],"gap":null},{"id":"claim:updated-training-result","subject":"capability:train-static-svm","relation":null,"object":null,"statement":"On data/personal-updated (1,050 samples; 50/class), the candidate scored 0.9666 macro-F1/0.9667 accuracy versus 0.0403/0.1143 for the prior model on the shared stratified 20% holdout; it was selected by macro-F1 and refit on all samples.","kind":"observation","status":"supported","evidence":[{"kind":"execution","source":"execution:current-session","locator":"stdout: shared holdout metrics, selection, and full-data refit","snapshot":null}],"derived_from":[],"gap":null},{"id":"claim:live-inference-session","subject":"capability:live-static-sign-recognition","relation":"exposes","object":"contract:live-inference-session","statement":"The demo reads the selected webcam, detects one hand in VIDEO mode, displays its class or a no-hand prompt, exits on q/Escape, and raises RuntimeError if the camera cannot open.","kind":"observation","status":"supported","evidence":[{"kind":"code","source":"demo.py","locator":"run_demo: camera guard, frame loop, HUD, and quit handling","snapshot":null}],"derived_from":[],"gap":null},{"id":"claim:live-inference-features","subject":"capability:live-static-sign-recognition","relation":"constrained_by","object":"rule:wrist-relative-features","statement":"predict_sign wrist-normalizes all 21 landmarks before SVM prediction.","kind":"observation","status":"supported","evidence":[{"kind":"code","source":"demo.py","locator":"predict_sign: normalize_landmarks before model.predict","snapshot":null}],"derived_from":[],"gap":null},{"id":"claim:live-inference-artifact","subject":"capability:live-static-sign-recognition","relation":"depends_on","object":"contract:svm-artifact","statement":"The demo requires a joblib estimator and ordered class labels matching the estimator classes.","kind":"observation","status":"supported","evidence":[{"kind":"code","source":"demo.py","locator":"load_model: artifact keys and class-order validation","snapshot":null}],"derived_from":[],"gap":null},{"id":"claim:live-inference-cleanup","subject":"capability:live-static-sign-recognition","relation":"constrained_by","object":"rule:live-resource-cleanup","statement":"run_demo finally releases the camera, closes an initialized landmarker, and destroys windows on exit or error.","kind":"observation","status":"supported","evidence":[{"kind":"code","source":"demo.py","locator":"run_demo: finally cleanup block","snapshot":null}],"derived_from":[],"gap":null}]}}
```

# Phase A Landmark SVM

Phase A collects hand crops and landmarks, trains the static-sign SVM, and supports webcam inference.

## DATA CONTRACT

- REQUIRED: Pair each JPEG with a same-basename `.landmarks.json` sidecar.
- REQUIRED: Store 21 finite raw points; wrist-normalize before SVM use.
- REQUIRED: Review class counts and rejected samples before fitting.
- PROHIBITED: Feed raw coordinates to the SVM.

## TRAINING FLOW

1. Run `python collect_data.py` to collect paired samples.
2. Run `notebooks/02_landmarks_svm.ipynb` from the project root.
3. Review holdout metrics and confusion matrix.
4. Fit all accepted samples and save `models/svm_model.pkl`.

Latest run: `data/personal-updated`, 1,050 samples (50/class). On the shared stratified 20% holdout, the candidate scored 0.9666 macro-F1 / 0.9667 accuracy versus 0.0403 / 0.1143 for the prior model. It won by macro-F1 and was refit on all samples. This single-run result is not a generalization guarantee.

Historical legacy run: 920 accepted, 130 rejected, 0.8913 holdout accuracy; class C had 18 accepted samples.

## LIVE INFERENCE

1. **Run** `python demo.py`; it loads `models/svm_model.pkl`.
2. Mirrored frames feed MediaPipe VIDEO tracking and wrist-relative SVM features; the HUD shows a predicted letter or “No hand detected.”
3. Press **`q`** or **`Esc`** to quit; use `--camera N`, `--model`, or `--hand-model` to override defaults.
4. The demo closes camera, MediaPipe, and OpenCV resources on exit or failure. Tests mock these boundaries; visual behavior needs a webcam run.

## REFERENCES

- [**ARCHITECTURE.md**](../adr/ARCHITECTURE.md): Project layers and integration constraints.
- [**TESTS.md**](../adr/TESTS.md): Test strategy and validation targets.
