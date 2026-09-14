"""
Unsupervised Job Recommendation Engine using TF-IDF and Cosine Similarity.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from src.features.text_features import load_vectorizer
from src.features.match_features import compute_cosine_similarity, compute_skill_overlap, extract_skills_from_text


class JobRecommender:
    """Content-based job recommender matching CV text against job postings corpus."""

    def __init__(self, jobs_df: pd.DataFrame, vectorizer=None):
        self.jobs_df = jobs_df.reset_index(drop=True)
        self.vectorizer = vectorizer or load_vectorizer()
        # Precompute job vectors
        self.job_vectors = self.vectorizer.transform(self.jobs_df["composite_text"].fillna(""))
        # Build skill universe from all jobs
        all_skills = set()
        for s_list in self.jobs_df["parsed_skills"]:
            if isinstance(s_list, list):
                all_skills.update(s_list)
        self.skill_universe = all_skills

    def recommend(
        self,
        resume_text: str,
        top_n: int = 5,
        filter_category: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Rank jobs by cosine similarity against candidate resume text.
        Returns top-N matching jobs enriched with match score and skill overlap.
        """
        resume_vec = self.vectorizer.transform([resume_text])
        sims = compute_cosine_similarity(resume_vec, self.job_vectors)

        candidate_skills = extract_skills_from_text(resume_text, self.skill_universe)

        scores_df = self.jobs_df.copy()
        scores_df["match_score"] = np.round(sims * 100, 1)

        if filter_category:
            scores_df = scores_df[scores_df["Category"].str.lower() == filter_category.strip().lower()]

        if min_similarity > 0.0:
            scores_df = scores_df[scores_df["match_score"] >= (min_similarity * 100)]

        ranked = scores_df.sort_values(by="match_score", ascending=False).head(top_n)

        results = []
        for _, row in ranked.iterrows():
            target_skills = row.get("parsed_skills", [])
            overlap = compute_skill_overlap(candidate_skills, target_skills)

            results.append({
                "job_id": row["Job_ID"],
                "title": row["Title"],
                "category": row["Category"],
                "company": row["Company"],
                "location": row["Location"],
                "experience_required": row["Experience_Required"],
                "salary_lpa": row["Salary_LPA"],
                "description": row["Description"],
                "match_score": float(row["match_score"]),
                "skills": target_skills,
                "candidate_skills": candidate_skills,
                "matched_skills": overlap["matched_skills"],
                "missing_skills": overlap["missing_skills"],
                "readiness_score": overlap["readiness_score"],
                "primary_gap": overlap["primary_gap"],
            })

        return results
