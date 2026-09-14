"""
Text cleaning, feature parsing, and data preprocessing for SmartHire.
"""
import re
from typing import List, Tuple
import pandas as pd

from src.config import JOB_CORPUS_CLEAN


def clean_text(text: str) -> str:
    """Clean and normalize raw text: lowercasing, removing special characters/urls, normalizing whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove emails
    text = re.sub(r"\S+@\S+", " ", text)
    # Replace non-alphanumeric (keep spaces, dashes, plus signs for C++, etc.)
    text = re.sub(r"[^a-zA-Z0-9\s\+\#\./\-]", " ", text)
    # Normalize multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_skills(skills_str: str) -> List[str]:
    """Parse semicolon/comma separated skills into a normalized, stripped list."""
    if not isinstance(skills_str, str) or not skills_str.strip():
        return []
    # Split by semicolon or comma
    delims = re.split(r"[;,]", skills_str)
    skills = [s.strip() for s in delims if s.strip()]
    return skills


def parse_experience(exp_str: str) -> Tuple[float, float, float]:
    """Parse experience string (e.g. '5-8 years', '2+ years', '0-3 years') into (min_exp, max_exp, avg_exp)."""
    if not isinstance(exp_str, str):
        return (0.0, 0.0, 0.0)
    numbers = [float(n) for n in re.findall(r"\d+", exp_str)]
    if len(numbers) >= 2:
        min_e, max_e = numbers[0], numbers[1]
    elif len(numbers) == 1:
        min_e = numbers[0]
        max_e = numbers[0] + 2.0
    else:
        min_e, max_e = 0.0, 1.0
    avg_e = (min_e + max_e) / 2.0
    return min_e, max_e, avg_e


def parse_salary(salary_str: str) -> Tuple[float, float, float]:
    """Parse salary range string (e.g. '16-24', '5-12') in LPA into (min_sal, max_sal, avg_sal)."""
    if not isinstance(salary_str, str):
        return (0.0, 0.0, 0.0)
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", salary_str)]
    if len(numbers) >= 2:
        min_s, max_s = numbers[0], numbers[1]
    elif len(numbers) == 1:
        min_s = numbers[0]
        max_s = numbers[0]
    else:
        min_s, max_s = 0.0, 0.0
    avg_s = (min_s + max_s) / 2.0
    return min_s, max_s, avg_s


def preprocess_job_corpus(df: pd.DataFrame, save_path=JOB_CORPUS_CLEAN) -> pd.DataFrame:
    """Preprocess the full job dataframe and enrich with parsed numerical and clean text columns."""
    df_clean = df.copy()

    # Parse skills
    df_clean["parsed_skills"] = df_clean["Skills"].apply(parse_skills)
    df_clean["skills_joined"] = df_clean["parsed_skills"].apply(lambda lst: " ".join(lst))

    # Parse experience & salary
    exp_tuples = df_clean["Experience_Required"].apply(parse_experience)
    df_clean["min_exp"] = [t[0] for t in exp_tuples]
    df_clean["max_exp"] = [t[1] for t in exp_tuples]
    df_clean["avg_exp"] = [t[2] for t in exp_tuples]

    sal_tuples = df_clean["Salary_LPA"].apply(parse_salary)
    df_clean["min_salary_lpa"] = [t[0] for t in sal_tuples]
    df_clean["max_salary_lpa"] = [t[1] for t in sal_tuples]
    df_clean["avg_salary_lpa"] = [t[2] for t in sal_tuples]

    # Clean text description
    df_clean["clean_description"] = df_clean["Description"].apply(clean_text)

    # Combined text for training and matching: Description + Skills + Title
    df_clean["composite_text"] = (
        df_clean["Title"].fillna("")
        + " "
        + df_clean["clean_description"]
        + " "
        + df_clean["skills_joined"]
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(save_path, index=False)

    return df_clean
