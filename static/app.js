document.getElementById("analyzeBtn").addEventListener("click", async () => {
    const text = document.getElementById("reviewText").value.trim();

    if (!text) {
        alert("Please enter a review!");
        return;
    }

    const response = await fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text })
    });

    const data = await response.json();

    document.getElementById("resultBox").style.display = "block";

    // Classical
    document.getElementById("classicalOutput").textContent =
        `Prediction: ${data.classical.prediction}`;

    // BERT
    document.getElementById("bertOutput").textContent =
        `Prediction: ${data.bert.prediction}`;
});
