"""Landmark extraction and SVM training utilities for Phase A."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np


LETTERS = tuple("ABCDEFGILMNOPQRSTUVWY")
FEATURE_COUNT = 21 * 3
DEFAULT_DATA_DIR = Path("data/personal")
DEFAULT_MODEL_PATH = Path("models/svm_model.pkl")
DEFAULT_HAND_LANDMARKER_MODEL = (
    Path.home() / ".cache" / "libras-recognition" / "hand_landmarker.task"
)


@dataclass(frozen=True)
class SVMEvaluation:
    """Metrics from a stratified holdout evaluation."""

    accuracy: float
    classification_report: dict
    confusion_matrix: np.ndarray
    classes: tuple[str, ...]
    test_samples_per_class: dict[str, int]


@dataclass(frozen=True)
class LandmarkDataset:
    """Landmark features, labels, and samples rejected during extraction."""

    features: np.ndarray
    labels: np.ndarray
    rejected: tuple[tuple[Path, str], ...]


def normalize_landmarks(landmarks: Iterable) -> np.ndarray:
    """Flatten 21 (x, y, z) points after subtracting the wrist coordinates."""
    try:
        landmarks = list(landmarks)
        if all(hasattr(point, "x") for point in landmarks):
            landmarks = [(point.x, point.y, point.z) for point in landmarks]
        points = np.asarray(landmarks, dtype=np.float32)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Expected 21 landmarks with x, y, and z coordinates") from error

    if points.shape != (21, 3):
        raise ValueError(f"Expected 21 landmarks with x, y, and z coordinates; got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Landmark coordinates must be finite")

    return (points - points[0]).reshape(FEATURE_COUNT)


def save_landmark_points(image_path: Path | str, landmarks: Iterable) -> Path:
    """Save raw MediaPipe coordinates beside a sample image as JSON."""
    points = np.asarray(
        [
            (point.x, point.y, point.z)
            if hasattr(point, "x")
            else point
            for point in landmarks
        ],
        dtype=np.float32,
    )
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Expected 21 finite landmarks with x, y, and z coordinates")

    image_path = Path(image_path)
    sidecar_path = image_path.with_suffix(".landmarks.json")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        json.dumps({"landmarks": points.tolist()}, separators=(",", ":")),
        encoding="utf-8",
    )
    return sidecar_path


def remove_landmark_overlay(image_bgr: np.ndarray) -> np.ndarray:
    """Inpaint the collector's red points and green lines in older crops."""
    import cv2

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, np.array([0, 140, 130]), np.array([12, 255, 255]))
    red_mask |= cv2.inRange(hsv, np.array([168, 140, 130]), np.array([180, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([90, 255, 255]))
    mask = red_mask | green_mask
    if not cv2.countNonZero(mask):
        return image_bgr

    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8))
    return cv2.inpaint(image_bgr, mask, 3, cv2.INPAINT_TELEA)


def create_hand_landmarker(
    model_path: Path | str = DEFAULT_HAND_LANDMARKER_MODEL,
    min_hand_detection_confidence: float = 0.1,
):
    """Create an image-mode MediaPipe Tasks Hand Landmarker."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Hand model not found at {model_path}; run collect_data.py to download it."
        )

    import mediapipe as mp
    import cv2  # Load OpenCV before the Tasks runtime starts native workers.

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=min_hand_detection_confidence,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def load_landmark_dataset(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    hand_landmarker=None,
    letters: Sequence[str] = LETTERS,
) -> LandmarkDataset:
    """Load saved landmarks or extract missing ones from labeled JPEG crops."""
    if hand_landmarker is None:
        raise ValueError("hand_landmarker is required")

    import mediapipe as mp
    import cv2

    data_dir = Path(data_dir)
    feature_rows = []
    labels = []
    rejected = []
    total_images = sum(len(list((data_dir / letter).glob("*.jpg"))) for letter in letters)
    processed = 0

    for letter in letters:
        class_dir = data_dir / letter
        image_paths = sorted(class_dir.glob("*.jpg"))
        if not image_paths:
            raise ValueError(f"No .jpg images found for class {letter} in {class_dir}")

        for image_path in image_paths:
            sidecar_path = image_path.with_suffix(".landmarks.json")
            if sidecar_path.is_file():
                try:
                    raw_landmarks = json.loads(sidecar_path.read_text(encoding="utf-8"))["landmarks"]
                except (KeyError, json.JSONDecodeError) as error:
                    raise ValueError(f"Invalid landmark sidecar: {sidecar_path}") from error
                feature_rows.append(normalize_landmarks(raw_landmarks))
                labels.append(letter)
            else:
                image_bgr = cv2.imread(str(image_path))
                if image_bgr is None:
                    rejected.append((image_path, "Could not read image"))
                else:
                    image_rgb = cv2.cvtColor(
                        remove_landmark_overlay(image_bgr), cv2.COLOR_BGR2RGB
                    )
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                    result = hand_landmarker.detect(image)
                    if not result.hand_landmarks:
                        rejected.append((image_path, "No hand detected"))
                    else:
                        feature_rows.append(normalize_landmarks(result.hand_landmarks[0]))
                        labels.append(letter)

            processed += 1
            if processed % 100 == 0 or processed == total_images:
                print(f"Processed {processed}/{total_images} images")

    return LandmarkDataset(
        features=np.asarray(feature_rows, dtype=np.float32).reshape(-1, FEATURE_COUNT),
        labels=np.asarray(labels),
        rejected=tuple(rejected),
    )


def _make_svm() -> object:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return make_pipeline(StandardScaler(), SVC(kernel="rbf"))


def _validate_training_data(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)
    if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
        raise ValueError(f"Expected features with shape (samples, {FEATURE_COUNT})")
    if labels.ndim != 1 or len(labels) != len(features):
        raise ValueError("Expected one class label for each feature row")
    if not len(labels):
        raise ValueError("Training data must not be empty")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        raise ValueError("SVM training requires at least two classes")
    if np.any(counts < 2):
        raise ValueError("Each class needs at least two samples for a stratified split")
    return features, labels


def evaluate_svm(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SVMEvaluation:
    """Evaluate RBF SVM with stratified holdout and per-class metrics."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    features, labels = _validate_training_data(features, labels)
    classes = tuple(str(label) for label in np.unique(labels))
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    model = _make_svm()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions, labels=classes)
    report = classification_report(
        y_test,
        predictions,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    _, class_counts = np.unique(y_test, return_counts=True)

    return SVMEvaluation(
        accuracy=float(accuracy_score(y_test, predictions)),
        classification_report=report,
        confusion_matrix=matrix,
        classes=classes,
        test_samples_per_class=dict(zip(classes, map(int, class_counts))),
    )


def fit_svm(features: np.ndarray, labels: np.ndarray):
    """Fit the final RBF SVM using all available samples."""
    features, labels = _validate_training_data(features, labels)
    model = _make_svm()
    model.fit(features, labels)
    return model


def save_svm_model(model, classes: Sequence[str], path: Path | str = DEFAULT_MODEL_PATH) -> Path:
    """Serialize a fitted model and its class order to the requested path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": tuple(classes)}, path)
    return path
