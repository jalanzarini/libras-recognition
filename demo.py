"""Run live Phase A sign prediction from a webcam."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mediapipe as mp
import cv2
import joblib

from src.landmark_svm import (
    DEFAULT_HAND_LANDMARKER_MODEL,
    DEFAULT_MODEL_PATH,
    normalize_landmarks,
)


CAMERA_INDEX = 0
WINDOW_NAME = "LIBRAS Static Sign Recognition"


def load_model(model_path: Path | str = DEFAULT_MODEL_PATH):
    """Load and validate the saved Phase A model artifact."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"SVM model not found at {model_path}; train Phase A first."
        )

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "model" not in artifact or "classes" not in artifact:
        raise ValueError(f"Invalid SVM model artifact at {model_path}")

    model = artifact["model"]
    saved_classes = tuple(str(label) for label in artifact["classes"])
    model_classes = tuple(str(label) for label in getattr(model, "classes_", ()))
    if not saved_classes or saved_classes != model_classes:
        raise ValueError(f"SVM model classes do not match the artifact at {model_path}")
    return model


def create_video_landmarker(model_path: Path | str = DEFAULT_HAND_LANDMARKER_MODEL):
    """Create a MediaPipe Tasks landmarker configured for webcam video."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Hand model not found at {model_path}; run collect_data.py to download it."
        )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def to_mp_image(frame_bgr):
    """Convert an OpenCV frame to a MediaPipe SRGB image."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)


def predict_sign(model, landmarks) -> str:
    """Predict a sign from one hand's 21 MediaPipe landmarks."""
    features = normalize_landmarks(landmarks).reshape(1, -1)
    return str(model.predict(features)[0])


def run_demo(
    camera_index: int = CAMERA_INDEX,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    hand_landmarker_model_path: Path | str = DEFAULT_HAND_LANDMARKER_MODEL,
) -> None:
    """Show live predictions until q or Escape is pressed."""
    model = load_model(model_path)
    capture = cv2.VideoCapture(camera_index)
    landmarker = None
    try:
        if not capture.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {camera_index}; check that a camera is connected."
            )

        landmarker = create_video_landmarker(hand_landmarker_model_path)
        last_timestamp_ms = -1

        while True:
            success, frame = capture.read()
            if not success:
                break

            # Match the mirrored signing view used during data collection.
            frame = cv2.flip(frame, 1)
            timestamp_ms = max(int(time.monotonic() * 1000), last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms
            result = landmarker.detect_for_video(to_mp_image(frame), timestamp_ms)

            if result.hand_landmarks:
                label = predict_sign(model, result.hand_landmarks[0])
                message, color = f"LIBRAS: {label}", (0, 220, 0)
            else:
                message, color = "No hand detected", (0, 0, 255)

            cv2.putText(
                frame,
                message,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "q / Esc: quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        try:
            if landmarker is not None:
                landmarker.close()
        finally:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live LIBRAS static-sign recognition.")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="webcam device index")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="saved SVM model")
    parser.add_argument(
        "--hand-model",
        type=Path,
        default=DEFAULT_HAND_LANDMARKER_MODEL,
        help="MediaPipe Hand Landmarker task model",
    )
    args = parser.parse_args()
    run_demo(args.camera, args.model, args.hand_model)


if __name__ == "__main__":
    main()
