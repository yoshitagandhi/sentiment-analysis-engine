from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.utils import setup_logger

logger = setup_logger(__name__)


class ModelEvaluator:
    """
    Evaluates trained ML models.
    """

    @staticmethod
    def evaluate(y_true, y_pred):

        accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
        )

        report = classification_report(
            y_true,
            y_pred,
            output_dict=True,
        )

        matrix = confusion_matrix(
            y_true,
            y_pred,
        )

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "report": report,
            "confusion_matrix": matrix,
        }

        return results