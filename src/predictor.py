from pathlib import Path
import joblib

from src.config import MODELS_DIR
from src.preprocessing import preprocess_text


MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
    "linear_svm": "Linear SVM",
}


class SentimentPredictor:

    def __init__(self, model_name="logistic_regression"):

        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model_name = model_name

        model_path = MODELS_DIR / f"{model_name}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        self.pipeline = joblib.load(model_path)

    def predict(self, review: str):
        raw_review = review

        review = preprocess_text(review)

        prediction = self.pipeline.predict([review])[0]

        result = {
            "model": MODEL_NAMES[self.model_name],
            "prediction": prediction.capitalize(),
        }

        if hasattr(self.pipeline, "predict_proba"):
            probabilities = self.pipeline.predict_proba([review])[0]
            class_probabilities = {
                cls.lower(): float(prob)
                for cls, prob in zip(self.pipeline.classes_, probabilities)
            }

            result["confidence"] = round(
                max(class_probabilities.values()) * 100,
                2,
            )

            result["probabilities"] = {
                "Negative": round(class_probabilities.get("negative", 0.0) * 100, 2),
                "Positive": round(class_probabilities.get("positive", 0.0) * 100, 2),
            }
        else:
            result["confidence"] = 100.0
            result["probabilities"] = {
                "Positive": 100.0 if prediction.lower() == "positive" else 0.0,
                "Negative": 100.0 if prediction.lower() == "negative" else 0.0,
            }

        return result