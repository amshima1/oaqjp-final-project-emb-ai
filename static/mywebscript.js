let RunSentimentAnalysis = () => {
    const textToAnalyze = document.getElementById("textToAnalyze").value;
    const systemResponse = document.getElementById("system_response");

    if (!textToAnalyze.trim()) {
        systemResponse.textContent = "Please enter text to analyze.";
        return;
    }

    systemResponse.textContent = "Analyzing...";

    const xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
        if (this.readyState !== XMLHttpRequest.DONE) {
            return;
        }

        if (this.status !== 200) {
            let response;
            try {
                response = JSON.parse(this.responseText);
            } catch (error) {
                systemResponse.textContent = "Emotion detection failed. Please try again.";
                return;
            }
            systemResponse.textContent = response.error || "Emotion detection failed. Please try again.";
            return;
        }

        if (!this.responseText) {
            systemResponse.textContent = "Emotion detection returned incomplete results.";
            return;
        }
        systemResponse.textContent = this.responseText;
    };
    xhttp.onerror = function() {
        systemResponse.textContent = "Unable to reach the emotion detection service. Please try again.";
    };
    xhttp.open(
        "GET",
        `/emotionDetector?textToAnalyze=${encodeURIComponent(textToAnalyze)}`,
        true
    );
    xhttp.send();
};
