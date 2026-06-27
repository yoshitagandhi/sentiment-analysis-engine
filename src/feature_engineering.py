"""
Feature Engineering Module
--------------------------

Converts cleaned text into TF-IDF feature vectors.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import TFIDF_VECTORIZER
from src.utils import load_pickle, save_pickle, setup_logger

logger = setup_logger(__name__)


class FeatureEngineer:
    """
    Handles TF-IDF feature extraction.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )

    def fit_transform(self, texts):
        logger.info("Training TF-IDF vectorizer...")
        X = self.vectorizer.fit_transform(texts)
        save_pickle(self.vectorizer, TFIDF_VECTORIZER)
        logger.info("Vectorizer saved.")
        return X

    def transform(self, texts):
        logger.info("Loading TF-IDF vectorizer...")
        self.vectorizer = load_pickle(TFIDF_VECTORIZER)
        return self.vectorizer.transform(texts)