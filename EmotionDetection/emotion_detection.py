"""Emotion detection using the IBM Watson NLP EmotionPredict API."""

import logging
import re

import requests

LOGGER = logging.getLogger(__name__)
API_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
MODEL_ID = "emotion_aggregated-workflow_lang_en_stock"
API_VERSION = "2022-04-07"
REQUEST_TIMEOUT = 30
EMOTION_NAMES = ("anger", "disgust", "fear", "joy", "sadness")
MAX_ERROR_BODY_LENGTH = 1000
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+\S+")
_SENSITIVE_FIELD = re.compile(
    r"""(?i)(["']?(?:authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|secret|credential)["']?\s*[:=]\s*)(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^,\s;&}]+)"""
)


def _safe_error_body(response):
    """Return bounded response details without exposing credential-like values."""
    body = response.text[:MAX_ERROR_BODY_LENGTH]
    body = _BEARER_TOKEN.sub("Bearer [REDACTED]", body)
    body = _SENSITIVE_FIELD.sub(r"\1[REDACTED]", body)
    if len(response.text) > MAX_ERROR_BODY_LENGTH:
        body += "..."
    return body or "<empty>"


def emotion_detector(text_to_analyze):
    """Return emotion scores and the dominant emotion for the supplied text."""
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        raise ValueError("Text to analyze must not be blank.")

    headers = {"grpc-metadata-mm-model-id": MODEL_ID}
    response = requests.post(
        f"{API_URL}?version={API_VERSION}",
        json={"raw_document": {"text": text_to_analyze}},
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    if not response.ok:
        LOGGER.error(
            "Watson EmotionPredict API returned HTTP %s %s; response body: %s",
            response.status_code,
            response.reason,
            _safe_error_body(response),
        )
    if response.status_code == 400:
        return {**dict.fromkeys(EMOTION_NAMES), "dominant_emotion": None}

    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as error:
        LOGGER.error(
            "Watson EmotionPredict API returned invalid JSON with HTTP %s; "
            "response body: %s",
            response.status_code,
            _safe_error_body(response),
        )
        raise ValueError("The emotion API returned invalid JSON.") from error

    try:
        emotion_scores = payload["emotionPredictions"][0]["emotion"]
        result = {name: emotion_scores[name] for name in EMOTION_NAMES}
    except (KeyError, IndexError, TypeError) as error:
        top_level_keys = (
            sorted(payload.keys()) if isinstance(payload, dict) else "<not an object>"
        )
        LOGGER.error(
            "Watson EmotionPredict API returned HTTP %s with an unexpected "
            "'emotionPredictions[0].emotion' structure; top-level keys: %s",
            response.status_code,
            top_level_keys,
        )
        raise ValueError("The emotion API returned an invalid response.") from error

    result["dominant_emotion"] = max(EMOTION_NAMES, key=result.__getitem__)
    return result
