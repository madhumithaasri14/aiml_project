"""
Data loading utilities for SmartHire.
"""
import pandas as pd
from pathlib import Path
from typing import Optional

from src.config import JOB_POSTINGS_RAW, JOB_CORPUS_CLEAN, JOB_CORPUS_PROCESSED


def load_raw_jobs(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw job postings CSV dataset."""
    path = filepath or JOB_POSTINGS_RAW
    if not path.exists():
        raise FileNotFoundError(f"Raw job postings file not found at: {path}")
    df = pd.read_csv(path)
    return df


def load_clean_jobs(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load the cleaned job corpus CSV dataset."""
    path = filepath or JOB_CORPUS_CLEAN
    if not path.exists():
        raise FileNotFoundError(f"Clean job corpus file not found at: {path}")
    df = pd.read_csv(path)
    return df


def load_processed_jobs(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load the processed job corpus CSV dataset."""
    path = filepath or JOB_CORPUS_PROCESSED
    if not path.exists():
        raise FileNotFoundError(f"Processed job corpus file not found at: {path}")
    df = pd.read_csv(path)
    return df
