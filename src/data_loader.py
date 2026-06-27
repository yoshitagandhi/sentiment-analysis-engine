"""
Data Loader
-----------

Loads and validates the IMDB sentiment dataset.
"""

from pathlib import Path

import pandas as pd

from src.config import DATASET_PATH
from src.utils import setup_logger

logger = setup_logger(__name__)


REQUIRED_COLUMNS = [
    "review",
    "sentiment",
]


def load_dataset(filepath: Path = DATASET_PATH) -> pd.DataFrame:
    """
    Load and validate the dataset.

    Parameters
    ----------
    filepath : Path
        CSV dataset location.

    Returns
    -------
    pd.DataFrame
        Clean dataframe.
    """

    logger.info("Loading dataset...")

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found:\n{filepath}")

    df = pd.read_csv(filepath)

    logger.info(f"Dataset shape: {df.shape}")

    return validate_dataset(df)


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate dataset structure.
    """

    logger.info("Validating dataset...")

    # Required columns
    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Drop missing values
    before = len(df)

    df = df.dropna()

    logger.info(
        f"Removed {before - len(df)} rows with missing values."
    )

    # Remove duplicates
    before = len(df)

    df = df.drop_duplicates()

    logger.info(
        f"Removed {before - len(df)} duplicate rows."
    )

    # Normalize labels
    df["sentiment"] = (
        df["sentiment"]
        .str.lower()
        .str.strip()
    )

    valid_labels = {"positive", "negative"}

    invalid = (
        ~df["sentiment"].isin(valid_labels)
    ).sum()

    if invalid > 0:
        raise ValueError(
            f"Found {invalid} invalid sentiment labels."
        )

    logger.info("Dataset validation successful.")

    return df.reset_index(drop=True)