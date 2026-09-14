"""
SmartHire Configuration: Paths, constants, and parameters.
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
APP_DIR = PROJECT_ROOT / "app"

# Data Files
JOB_POSTINGS_RAW = RAW_DATA_DIR / "job_postings.csv"
JOB_CORPUS_CLEAN = INTERIM_DATA_DIR / "job_corpus_clean.csv"
JOB_CORPUS_PROCESSED = PROCESSED_DATA_DIR / "job_corpus_processed.csv"

# Model Artifacts
CLASSIFIER_PATH = MODELS_DIR / "classifier.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
KMEANS_MODEL_PATH = MODELS_DIR / "job_kmeans.pkl"
FIT_PREDICTOR_PATH = MODELS_DIR / "fit_predictor.pkl"

# Constants
RANDOM_STATE = 42
CATEGORIES = [
    "Android Development",
    "Business Analyst",
    "Civil Engineer",
    "Data Science",
    "DevOps Engineer",
    "Digital Marketing",
    "Graphic Designer",
    "HR",
    "Health and Fitness",
    "Java Developer",
    "Mechanical Engineer",
    "Network Security Engineer",
    "Python Developer",
    "Sales",
    "Web Development",
]

# Feature Extraction Parameters
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
