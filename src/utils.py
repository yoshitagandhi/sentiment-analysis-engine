"""
Utility Functions
-----------------

Reusable helper functions used across the project.
"""

import logging
import pickle
import time
from pathlib import Path


def setup_logger(name: str) -> logging.Logger:
    """
    Create and return a logger.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    return logging.getLogger(name)


def create_directory(path: Path) -> None:
    """
    Create a directory if it doesn't exist.
    """

    path.mkdir(parents=True, exist_ok=True)


def save_pickle(obj, filepath: Path) -> None:
    """
    Save a Python object using pickle.
    """

    with open(filepath, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(filepath: Path):
    """
    Load a pickle object.
    """

    with open(filepath, "rb") as file:
        return pickle.load(file)


class Timer:
    """
    Simple execution timer.
    """

    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        elapsed = time.perf_counter() - self.start_time
        return elapsed