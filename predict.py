from transformers import pipeline


# Load the trained/fine-tuned RoBERTa AI text detector
classifier = pipeline(
    "text-classification",
    model="drasti02/ai-text-detector-roberta"
)


def predict_text(text):
    """
    Predict whether the given text is AI-generated or human-written.
    """

    text = text.strip()

    if not text:
        return {
            "label": "ERROR",
            "confidence": 0
        }

    result = classifier(text)[0]

    label = result["label"]
    confidence = round(result["score"] * 100, 2)

    # Convert model output to readable labels
    label_upper = label.upper()

    if label_upper in ["LABEL_1", "AI", "AI-GENERATED", "AI_GENERATED"]:
        final_label = "AI-GENERATED"
    else:
        final_label = "HUMAN-WRITTEN"

    return {
        "label": final_label,
        "confidence": confidence
    }


# Test the model directly
if __name__ == "__main__":

    sample_text = """
    Artificial intelligence is transforming modern education by
    providing students with personalized learning experiences.
    These systems can analyze information and help students
    understand difficult concepts more efficiently.
    """

    result = predict_text(sample_text)

    print("\nAI Text Detection Result")
    print("------------------------")
    print("Prediction:", result["label"])
    print("Confidence:", str(result["confidence"]) + "%")