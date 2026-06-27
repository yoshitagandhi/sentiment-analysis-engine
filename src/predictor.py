from pathlib import Path
import joblib

from src.config import MODELS_DIR
from src.preprocessing import preprocess_text


class SentimentPredictor:

    def __init__(self, model_name="logistic_regression"):

        model_path = MODELS_DIR / f"{model_name}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        self.pipeline = joblib.load(model_path)

    def predict(self, review):

        cleaned = preprocess_text(review)

        prediction = self.pipeline.predict([cleaned])[0]

        result = {
            "prediction": prediction
        }

        if hasattr(self.pipeline, "predict_proba"):
            probabilities = self.pipeline.predict_proba([cleaned])[0]

            result["confidence"] = float(max(probabilities))

            result["probabilities"] = probabilities.tolist()

        return result