import json

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.config import (
    DATASET_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
)
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
        "logistic_regression": GridSearchCV(
            LogisticRegression(max_iter=1000),
            {
                "C": [0.1, 1, 10],
                "solver":["liblinear"],
            },
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        ),
        "naive_bayes": GridSearchCV(
            MultinomialNB(),
            {
                "alpha": [0.1, 0.5, 1.0],
            },
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        ),
        "linear_svm": GridSearchCV(
            CalibratedClassifierCV(
                LinearSVC(
                    random_state=42
                )
            ),
            {
                "estimator__C": [0.1, 1, 10],
            },
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        ),
    }
    
    results = []

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    for name, model in models.items():

        print(f"\nTraining {name}...")

        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        max_features=20000,
                        ngram_range=(1, 2),
                        min_df=3,
                        max_df=0.90,
                        sublinear_tf=True,
                    ),
                ),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        report = classification_report(
            y_test,
            predictions,
        )

        report_path = OUTPUTS_DIR / f"{name}_classification_report.txt"

        with open(report_path, "w") as file:
            file.write(report)

        print(f"Saved -> {report_path}")
        
        cm = confusion_matrix(
            y_test,
            predictions,
        )

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
        )

        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.tight_layout()

        cm_path = OUTPUTS_DIR / f"{name}_confusion_matrix.png"

        plt.savefig(cm_path)

        plt.close()

        print(f"Saved -> {cm_path}")

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

    results_df = results_df.sort_values(
        "Accuracy",
        ascending=False,
    )

    csv_path = OUTPUTS_DIR / "model_comparison.csv"

    results_df.to_csv(
        csv_path,
        index=False,
    )

    print(results_df)

    print(f"Saved -> {csv_path}")

    return results_df