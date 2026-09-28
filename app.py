from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Fine-tuned RoBERTa AI-text detector
MODEL_NAME = "drasti02/ai-text-detector-roberta"

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detector")
def detector():
    return render_template("detector.html")


@app.route("/research")
def research():
    return render_template("research.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():

    text = request.form.get("text", "").strip()

    if not text:
        return render_template(
            "detector.html",
            error="Please enter some text."
        )

    word_count = len(text.split())

    if word_count < 10:
        return render_template(
            "detector.html",
            error="Please enter at least 10 words.",
            text=text
        )

    try:
        # The model can handle long text by truncating it.
        result = classifier(
            text,
            truncation=True,
            max_length=256
        )[0]

        raw_label = result["label"]
        score = result["score"]

        # Model labels:
        # LABEL_0 = Human-written
        # LABEL_1 = AI-generated
        # Some model versions may return descriptive labels.

        if raw_label in ["LABEL_1", "AI", "AI-generated", "AI Generated"]:
            prediction = "AI-GENERATED"
        else:
            prediction = "HUMAN-WRITTEN"

        confidence = round(score * 100, 2)

        return render_template(
            "detector.html",
            prediction=prediction,
            confidence=confidence,
            text=text,
            word_count=word_count
        )

    except Exception as e:

        return render_template(
            "detector.html",
            error=f"Model error: {str(e)}",
            text=text
        )


if __name__ == "__main__":
    app.run(debug=False)