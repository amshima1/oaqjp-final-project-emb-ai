"""Unit tests for the IBM Watson emotion detection client."""

import unittest
from unittest.mock import Mock, patch

import requests

from EmotionDetection import emotion_detector
from EmotionDetection.emotion_detection import API_URL, API_VERSION, MODEL_ID
from server import app


class EmotionDetectorTests(unittest.TestCase):
    """Tests the API request and emotion result handling."""

    def setUp(self):
        self.emotion_scores = {
            "anger": 0.1,
            "disgust": 0.05,
            "fear": 0.15,
            "joy": 0.6,
            "sadness": 0.1,
        }
        self.response = Mock()
        self.response.status_code = 200
        self.response.json.return_value = {
            "emotionPredictions": [{"emotion": self.emotion_scores}]
        }

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_returns_all_emotion_scores(self, post):
        post.return_value = self.response

        result = emotion_detector("A test sentence.")

        self.assertEqual(
            result,
            {
                **self.emotion_scores,
                "dominant_emotion": "joy",
            },
        )

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_uses_highest_score_as_dominant_emotion(self, post):
        scores = {**self.emotion_scores, "fear": 0.8}
        self.response.json.return_value = {
            "emotionPredictions": [{"emotion": scores}]
        }
        post.return_value = self.response

        result = emotion_detector("Another test sentence.")

        self.assertEqual(result["dominant_emotion"], "fear")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_selects_each_emotion_as_dominant(self, post):
        post.return_value = self.response

        for dominant_emotion in (
            "anger",
            "disgust",
            "fear",
            "joy",
            "sadness",
        ):
            with self.subTest(dominant_emotion=dominant_emotion):
                scores = {
                    emotion: 0.9 if emotion == dominant_emotion else 0.01
                    for emotion in self.emotion_scores
                }
                self.response.json.return_value = {
                    "emotionPredictions": [{"emotion": scores}]
                }

                result = emotion_detector("A test sentence.")

                self.assertEqual(result["dominant_emotion"], dominant_emotion)

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_sends_expected_api_request(self, post):
        post.return_value = self.response
        text = "Send this text."

        emotion_detector(text)

        post.assert_called_once_with(
            f"{API_URL}?version={API_VERSION}",
            json={"raw_document": {"text": text}},
            headers={"grpc-metadata-mm-model-id": MODEL_ID},
            timeout=30,
        )

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_rejects_blank_text_without_calling_api(self, post):
        with self.assertRaises(ValueError):
            emotion_detector("  \n")

        post.assert_not_called()

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_rejects_invalid_api_response(self, post):
        self.response.json.return_value = {"emotionPredictions": []}
        post.return_value = self.response

        with self.assertLogs("EmotionDetection.emotion_detection", level="ERROR") as logs:
            with self.assertRaisesRegex(ValueError, "invalid response"):
                emotion_detector("Text with an invalid API response.")

        self.assertIn("HTTP 200", logs.output[0])
        self.assertIn("emotionPredictions[0].emotion", logs.output[0])

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_logs_http_status_and_redacts_credentials(self, post):
        response = Mock(
            ok=False,
            status_code=503,
            reason="Service Unavailable",
            text=(
                '{"message":"temporarily unavailable","api_key":"do-not-log",'
                '"password":"never print this phrase"}'
            ),
        )
        response.raise_for_status.side_effect = requests.HTTPError(
            "503 Server Error"
        )
        post.return_value = response

        with self.assertLogs(
            "EmotionDetection.emotion_detection", level="ERROR"
        ) as logs:
            with self.assertRaises(requests.HTTPError):
                emotion_detector("A test sentence.")

        self.assertIn("HTTP 503", logs.output[0])
        self.assertIn("temporarily unavailable", logs.output[0])
        self.assertIn("[REDACTED]", logs.output[0])
        self.assertNotIn("do-not-log", logs.output[0])
        self.assertNotIn("never print this phrase", logs.output[0])

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_returns_empty_emotions_for_http_400(self, post):
        response = Mock(
            ok=False,
            status_code=400,
            reason="Bad Request",
            text="Invalid input.",
        )
        response.raise_for_status.side_effect = requests.HTTPError("400 Bad Request")
        post.return_value = response

        with self.assertLogs(
            "EmotionDetection.emotion_detection", level="ERROR"
        ) as logs:
            result = emotion_detector("A test sentence.")

        self.assertEqual(
            result,
            {
                "anger": None,
                "disgust": None,
                "fear": None,
                "joy": None,
                "sadness": None,
                "dominant_emotion": None,
            },
        )
        self.assertIn("HTTP 400", logs.output[0])
        response.raise_for_status.assert_not_called()


class EmotionDetectorRouteTests(unittest.TestCase):
    """Tests the Flask endpoint with its external detector isolated."""

    def setUp(self):
        self.client = app.test_client()

    @patch("server.emotion_detector")
    def test_endpoint_returns_detected_emotions(self, detector):
        expected = {
            "anger": 0.1,
            "disgust": 0.05,
            "fear": 0.15,
            "joy": 0.6,
            "sadness": 0.1,
            "dominant_emotion": "joy",
        }
        detector.return_value = expected

        response = self.client.get(
            "/emotionDetector",
            query_string={"textToAnalyze": "A test sentence."},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_data(as_text=True),
            "anger: 0.1, disgust: 0.05, fear: 0.15, joy: 0.6, "
            "sadness: 0.1, dominant_emotion: joy",
        )
        self.assertEqual(response.mimetype, "text/plain")
        detector.assert_called_once_with("A test sentence.")

    @patch("server.emotion_detector")
    def test_endpoint_reports_invalid_text(self, detector):
        detector.return_value = {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

        response = self.client.get(
            "/emotionDetector",
            query_string={"textToAnalyze": "Invalid text."},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_data(as_text=True),
            "Invalid text! Please try again!",
        )
        self.assertEqual(response.mimetype, "text/plain")
        detector.assert_called_once_with("Invalid text.")

    @patch("server.emotion_detector")
    def test_endpoint_rejects_empty_text(self, detector):
        response = self.client.get(
            "/emotionDetector",
            query_string={"textToAnalyze": "  "},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("must not be blank", response.get_json()["error"])
        detector.assert_not_called()

    @patch("server.emotion_detector", side_effect=ValueError("Invalid API response"))
    def test_endpoint_reports_detector_failure(self, _detector):
        response = self.client.get(
            "/emotionDetector",
            query_string={"textToAnalyze": "A test sentence."},
        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.get_json(),
            {"error": "Emotion detection service failed."},
        )


if __name__ == "__main__":
    unittest.main()
