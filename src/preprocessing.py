"""
Text Preprocessing Module
-------------------------

Provides utilities for cleaning movie reviews before feature extraction.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.utils import setup_logger

logger = setup_logger(__name__)

# Download resources (only first time)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """
    Clean a movie review.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Clean review.
    """

    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\\S+|www\\S+", " ", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Stopword removal + Lemmatization
    tokens = [
        LEMMATIZER.lemmatize(word)
        for word in tokens
        if word not in STOPWORDS and len(word) > 1
    ]

    return " ".join(tokens)