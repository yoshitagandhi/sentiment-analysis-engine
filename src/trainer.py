from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.config import DATASET_PATH, MODELS_DIR
from src.data_loader import load_dataset
from src.preprocessing import preprocess_text

def train_all_models():

    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    df = load_dataset(DATASET_PATH)

    df["review"] = df["review"].apply(preprocess_text)

    X = df["review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "naive_bayes": MultinomialNB(),
        "linear_svm": LinearSVC(),
    }

    results = []

    MODELS_DIR.mkdir(exist_ok=True)

    for name, model in models.items():

        print(f"\nTraining {name}...")

        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        stop_words="english",
                        max_features=5000,
                    ),
                ),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            predictions,
            average="weighted",
        )

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )

        model_path = MODELS_DIR / f"{name}.pkl"

        joblib.dump(pipeline, model_path)

        print(f"Saved -> {model_path}")

    print("\n")
    print("=" * 60)
    print("Training Summary")
    print("=" * 60)

    results_df = pd.DataFrame(results)

    print(results_df.sort_values("Accuracy", ascending=False))

    return results_df