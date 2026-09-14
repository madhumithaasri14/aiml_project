"""
Match features, cosine similarity, skill overlap, and readiness scoring.
"""
from typing import List, Set, Tuple, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarity(resume_vector, job_vectors) -> np.ndarray:
    """Compute cosine similarity between a single resume vector and multiple job vectors."""
    sims = cosine_similarity(resume_vector, job_vectors)
    return sims.flatten()


def extract_skills_from_text(text: str, skill_universe: Set[str]) -> List[str]:
    """Extract known skills present in the resume text using exact token/phrase matching."""
    text_lower = f" {text.lower()} "
    found_skills = []
    for skill in skill_universe:
        skill_clean = skill.strip()
        if not skill_clean:
            continue
        # Check word boundary for single or multi-word skill
        pattern = rf"\b{re.escape(skill_clean.lower())}\b"
        if re.search(pattern, text_lower):
            found_skills.append(skill_clean)
    return sorted(list(set(found_skills)))


import re


def compute_skill_overlap(candidate_skills: List[str], target_skills: List[str]) -> Dict[str, Any]:
    """
    Compute overlap metrics between candidate skills and target role/cluster skills.
    Returns:
        matched_skills: list of skills both have
        missing_skills: list of target skills candidate lacks
        readiness_score: percentage of target skills candidate possesses (0-100%)
        jaccard_score: intersection over union
    """
    cand_set = set(s.strip().lower() for s in candidate_skills if s.strip())
    target_set = set(s.strip().lower() for s in target_skills if s.strip())

    if not target_set:
        return {
            "matched_skills": [],
            "missing_skills": [],
            "readiness_score": 100.0,
            "jaccard_score": 1.0,
            "primary_gap": "None",
        }

    matched_lower = cand_set.intersection(target_set)
    missing_lower = target_set.difference(cand_set)

    # Preserve order of target_skills
    matched = [s.strip() for s in target_skills if s.strip() and s.strip().lower() in matched_lower]
    missing = [s.strip() for s in target_skills if s.strip() and s.strip().lower() in missing_lower]

    readiness = (len(matched_lower) / len(target_set)) * 100.0
    union_len = len(cand_set.union(target_set))
    jaccard = len(matched_lower) / union_len if union_len > 0 else 0.0

    primary_gap = missing[0] if missing else "None"

    return {
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "readiness_score": round(readiness, 1),
        "jaccard_score": round(jaccard, 3),
        "primary_gap": primary_gap,
    }
