"""
Project Configuration
---------------------

Stores all project paths and global constants.
"""

from pathlib import Path

# =========================
# Project Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

MODELS_DIR = PROJECT_ROOT / "models"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

TESTS_DIR = PROJECT_ROOT / "tests"

# =========================
# Files
# =========================

DATASET_PATH = RAW_DATA_DIR / "demo_imdb.csv"

TFIDF_VECTORIZER = MODELS_DIR / "tfidf_vectorizer.pkl"

LOGISTIC_MODEL = MODELS_DIR / "logistic_regression.pkl"

NAIVE_BAYES_MODEL = MODELS_DIR / "naive_bayes.pkl"

LINEAR_SVM_MODEL = MODELS_DIR / "linear_svm.pkl"

# =========================
# Random State
# =========================

RANDOM_STATE = 42

TEST_SIZE = 0.2