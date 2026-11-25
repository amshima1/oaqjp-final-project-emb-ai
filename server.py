"""Flask application for IBM Watson emotion detection."""

from flask import Flask, Response, jsonify, render_template, request
import requests

from EmotionDetection import emotion_detector

app = Flask(__name__)


@app.route("/")
def home():
    """Render the emotion detection page."""
    return render_template("index.html")


@app.route("/emotionDetector")
def detect_emotion():
    """Analyze the submitted text and return its emotion scores."""
    text_to_analyze = request.args.get("textToAnalyze", "")
    if not text_to_analyze.strip():
        return jsonify({"error": "Text to analyze must not be blank."}), 400

    try:
        result = emotion_detector(text_to_analyze)
    except (requests.RequestException, ValueError) as error:
        app.logger.error("Emotion detection failed: %s", error)
        return jsonify({"error": "Emotion detection service failed."}), 502

    if result["dominant_emotion"] is None:
        return Response("Invalid text! Please try again!", mimetype="text/plain")

    formatted_result = (
        f"anger: {result['anger']}, "
        f"disgust: {result['disgust']}, "
        f"fear: {result['fear']}, "
        f"joy: {result['joy']}, "
        f"sadness: {result['sadness']}, "
        f"dominant_emotion: {result['dominant_emotion']}"
    )
    return Response(formatted_result, mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# Run static code analysis using: pylint server.py
