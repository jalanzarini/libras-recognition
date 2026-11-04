import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from demo import predict_sign, run_demo


def make_landmarks():
    return [
        SimpleNamespace(x=index / 100, y=index / 200, z=index / 300)
        for index in range(21)
    ]


class PredictionTests(unittest.TestCase):
    def test_predict_sign_uses_wrist_relative_features(self):
        model = Mock()
        model.predict.return_value = np.asarray(["A"])

        prediction = predict_sign(model, make_landmarks())

        self.assertEqual(prediction, "A")
        features = model.predict.call_args.args[0]
        self.assertEqual(features.shape, (1, 63))
        np.testing.assert_allclose(features[0, :6], [0, 0, 0, 0.01, 0.005, 1 / 300])


class DemoLoopTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((32, 32, 3), dtype=np.uint8)
        self.capture = Mock()
        self.capture.isOpened.return_value = True
        self.capture.read.return_value = (True, self.frame)
        self.model = Mock()
        self.model.predict.return_value = np.asarray(["B"])
        self.landmarker = Mock()
        self.image = object()

    @patch("demo.cv2.putText")
    @patch("demo.cv2.destroyAllWindows")
    @patch("demo.cv2.waitKey", return_value=ord("q"))
    @patch("demo.cv2.imshow")
    @patch("demo.to_mp_image")
    @patch("demo.create_video_landmarker")
    @patch("demo.load_model")
    @patch("demo.cv2.VideoCapture")
    def test_displays_prediction_and_releases_resources_on_quit(
        self,
        video_capture,
        load_model,
        create_landmarker,
        to_mp_image,
        imshow,
        wait_key,
        destroy_windows,
        put_text,
    ):
        video_capture.return_value = self.capture
        load_model.return_value = self.model
        create_landmarker.return_value = self.landmarker
        to_mp_image.return_value = self.image
        self.landmarker.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[make_landmarks()]
        )

        run_demo()

        self.model.predict.assert_called_once()
        self.landmarker.detect_for_video.assert_called_once()
        self.assertTrue(any("B" in args[1] for args, _ in put_text.call_args_list))
        self.capture.release.assert_called_once()
        self.landmarker.close.assert_called_once()
        destroy_windows.assert_called_once()

    @patch("demo.cv2.putText")
    @patch("demo.cv2.destroyAllWindows")
    @patch("demo.cv2.waitKey", return_value=ord("q"))
    @patch("demo.cv2.imshow")
    @patch("demo.to_mp_image")
    @patch("demo.create_video_landmarker")
    @patch("demo.load_model")
    @patch("demo.cv2.VideoCapture")
    def test_no_hand_displays_prompt_without_predicting(
        self,
        video_capture,
        load_model,
        create_landmarker,
        to_mp_image,
        imshow,
        wait_key,
        destroy_windows,
        put_text,
    ):
        video_capture.return_value = self.capture
        load_model.return_value = self.model
        create_landmarker.return_value = self.landmarker
        to_mp_image.return_value = self.image
        self.landmarker.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[])

        run_demo()

        self.model.predict.assert_not_called()
        self.assertTrue(any("No hand" in args[1] for args, _ in put_text.call_args_list))
        self.capture.release.assert_called_once()
        self.landmarker.close.assert_called_once()

    @patch("demo.load_model")
    @patch("demo.cv2.VideoCapture")
    def test_closed_camera_raises_and_releases_capture(self, video_capture, load_model):
        capture = Mock()
        capture.isOpened.return_value = False
        video_capture.return_value = capture
        load_model.return_value = self.model

        with self.assertRaisesRegex(RuntimeError, "camera"):
            run_demo()

        capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
