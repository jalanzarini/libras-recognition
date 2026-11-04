import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import mediapipe

from src.landmark_svm import (
    evaluate_svm,
    load_landmark_dataset,
    normalize_landmarks,
    remove_landmark_overlay,
    save_landmark_points,
)


class LandmarkNormalizationTests(unittest.TestCase):
    def test_normalizes_all_coordinates_relative_to_wrist(self):
        landmarks = [
            SimpleNamespace(x=i / 10, y=i / 20, z=-i / 30)
            for i in range(21)
        ]

        features = normalize_landmarks(landmarks)

        self.assertEqual(features.shape, (63,))
        np.testing.assert_allclose(features[:3], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(features[3:6], [0.1, 0.05, -1 / 30])

    def test_rejects_landmark_sets_that_do_not_have_21_points(self):
        landmarks = [SimpleNamespace(x=0, y=0, z=0)] * 20

        with self.assertRaisesRegex(ValueError, "21 landmarks"):
            normalize_landmarks(landmarks)


class OverlayCleaningTests(unittest.TestCase):
    def test_removes_collector_red_and_green_marks(self):
        image = np.full((24, 24, 3), 100, dtype=np.uint8)
        image[5:19, 10] = (0, 200, 0)
        image[5:19, 14] = (0, 0, 255)

        cleaned = remove_landmark_overlay(image)

        self.assertEqual(cleaned.shape, image.shape)
        self.assertFalse(np.any(np.all(cleaned == (0, 200, 0), axis=2)))
        self.assertFalse(np.any(np.all(cleaned == (0, 0, 255), axis=2)))


class DatasetLoadingTests(unittest.TestCase):
    def test_uses_saved_landmarks_without_rerunning_detection(self):
        points = [
            SimpleNamespace(x=i / 100, y=i / 200, z=i / 300)
            for i in range(21)
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            class_dir = Path(temp_dir) / "A"
            class_dir.mkdir()
            image_path = class_dir / "frame_0000.jpg"
            image_path.touch()
            save_landmark_points(image_path, points)
            detector = SimpleNamespace(detect=Mock(side_effect=AssertionError))

            dataset = load_landmark_dataset(
                temp_dir, detector, letters=("A",)
            )

        self.assertEqual(dataset.features.shape, (1, 63))
        self.assertEqual(dataset.labels.tolist(), ["A"])
        np.testing.assert_allclose(dataset.features[0, :3], 0.0)
        detector.detect.assert_not_called()

    def test_loads_images_by_folder_label_and_extracts_wrist_relative_features(self):
        points = [SimpleNamespace(x=i / 100, y=i / 200, z=i / 300) for i in range(21)]
        result = SimpleNamespace(hand_landmarks=[points])
        fake_landmarker = SimpleNamespace(detect=lambda image: result)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for letter in ("A", "B"):
                class_dir = root / letter
                class_dir.mkdir()
                (class_dir / "frame_0000.jpg").touch()
                (class_dir / "frame_0001.jpg").touch()

            with patch("cv2.imread", return_value=np.zeros((20, 20, 3), dtype=np.uint8)):
                dataset = load_landmark_dataset(
                    root, fake_landmarker, letters=("A", "B")
                )

        self.assertEqual(dataset.features.shape, (4, 63))
        self.assertEqual(dataset.labels.tolist(), ["A", "A", "B", "B"])
        self.assertEqual(dataset.rejected, ())
        np.testing.assert_allclose(dataset.features[:, :3], 0.0)

    def test_records_images_without_detected_hands_as_rejected(self):
        fake_landmarker = SimpleNamespace(
            detect=lambda image: SimpleNamespace(hand_landmarks=[])
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            class_dir = Path(temp_dir) / "A"
            class_dir.mkdir()
            image_path = class_dir / "frame_0000.jpg"
            image_path.touch()

            with patch("cv2.imread", return_value=np.zeros((20, 20, 3), dtype=np.uint8)):
                dataset = load_landmark_dataset(
                    temp_dir, fake_landmarker, letters=("A",)
                )

        self.assertEqual(dataset.features.shape, (0, 63))
        self.assertEqual(dataset.labels.size, 0)
        self.assertEqual(len(dataset.rejected), 1)
        self.assertEqual(dataset.rejected[0][0], image_path)


class SVMTrainingTests(unittest.TestCase):
    def test_evaluates_with_stratified_20_percent_holdout_and_per_class_metrics(self):
        features = []
        labels = []
        for class_index, label in enumerate(("A", "B", "C")):
            for sample_index in range(10):
                row = np.zeros(63, dtype=float)
                row[class_index] = 10 + sample_index / 100
                features.append(row)
                labels.append(label)

        result = evaluate_svm(np.asarray(features), np.asarray(labels))

        self.assertEqual(result.confusion_matrix.shape, (3, 3))
        self.assertEqual(int(result.confusion_matrix.sum()), 6)
        self.assertEqual(result.test_samples_per_class, {"A": 2, "B": 2, "C": 2})
        self.assertTrue(all(label in result.classification_report for label in ("A", "B", "C")))


if __name__ == "__main__":
    unittest.main()
