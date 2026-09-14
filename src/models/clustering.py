"""
Unsupervised K-Means Clustering and Cluster-based Skill-Gap Engine for SmartHire.
"""
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib

from src.config import KMEANS_MODEL_PATH, RANDOM_STATE
from src.features.text_features import load_vectorizer
from src.features.match_features import extract_skills_from_text, compute_skill_overlap


class JobClusterEngine:
    """Clustering and skill-gap diagnostic engine."""

    def __init__(self, k: int = 15, random_state: int = RANDOM_STATE):
        self.k = k
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        self.vectorizer = None
        self.cluster_top_skills: Dict[int, List[str]] = {}
        self.cluster_top_terms: Dict[int, List[str]] = {}
        self.cluster_dominant_categories: Dict[int, str] = {}
        self.skill_universe = set()

    def fit(self, df: pd.DataFrame, vectorizer=None):
        """Fit K-Means on jobs text vectors and compute cluster profiles."""
        self.vectorizer = vectorizer or load_vectorizer()
        X = self.vectorizer.transform(df["composite_text"].fillna(""))
        labels = self.kmeans.fit_predict(X)
        df_clustered = df.copy()
        df_clustered["cluster"] = labels

        # Populate skill universe
        for s_list in df_clustered["parsed_skills"]:
            if isinstance(s_list, list):
                self.skill_universe.update(s_list)

        # Profile each cluster
        terms = np.array(self.vectorizer.get_feature_names_out())
        for c in range(self.k):
            # Dominant category
            sub = df_clustered[df_clustered["cluster"] == c]
            if not sub.empty:
                dom_cat = sub["Category"].mode().iloc[0]
            else:
                dom_cat = "General"
            self.cluster_dominant_categories[c] = dom_cat

            # Top TF-IDF centroid terms
            centroid = self.kmeans.cluster_centers_[c]
            top_term_indices = centroid.argsort()[::-1][:15]
            self.cluster_top_terms[c] = [terms[i] for i in top_term_indices]

            # Top skills by frequency in cluster
            cluster_skills = []
            for slist in sub["parsed_skills"]:
                if isinstance(slist, list):
                    cluster_skills.extend(slist)
            skill_counts = pd.Series(cluster_skills).value_counts()
            self.cluster_top_skills[c] = list(skill_counts.head(10).index)

        return df_clustered

    def get_skill_gap_report(self, candidate_text: str) -> Dict[str, Any]:
        """
        Analyze a candidate's CV against the closest job cluster.
        Returns readiness score, matched skills, missing skills, and a one-line action recommendation.
        """
        cand_vec = self.vectorizer.transform([candidate_text])
        cluster_id = int(self.kmeans.predict(cand_vec)[0])

        target_skills = self.cluster_top_skills.get(cluster_id, [])
        candidate_skills = extract_skills_from_text(candidate_text, self.skill_universe)

        overlap = compute_skill_overlap(candidate_skills, target_skills)
        dom_cat = self.cluster_dominant_categories.get(cluster_id, "Target Role")

        # Generate single-sentence actionable explanation
        if overlap["missing_skills"]:
            top_gap = overlap["missing_skills"][0]
            readiness = overlap["readiness_score"]
            potential_readiness = round(readiness + (100.0 / len(target_skills)), 1)
            one_line_gap = (
                f"Single biggest gap to close: Master {top_gap} to elevate role readiness from "
                f"{readiness}% to {min(100.0, potential_readiness)}% for {dom_cat}."
            )
        else:
            one_line_gap = f"Strong alignment: Candidate demonstrates 100% core skill coverage for {dom_cat}."

        return {
            "cluster_id": cluster_id,
            "target_role_domain": dom_cat,
            "candidate_skills": candidate_skills,
            "cluster_core_skills": target_skills,
            "matched_skills": overlap["matched_skills"],
            "missing_skills": overlap["missing_skills"],
            "readiness_score": overlap["readiness_score"],
            "single_line_explanation": one_line_gap,
            "primary_gap": overlap["primary_gap"],
        }

    def save(self, path=KMEANS_MODEL_PATH):
        """Serialize cluster engine to disk."""
        path = Path(path) if path else KMEANS_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path=KMEANS_MODEL_PATH) -> "JobClusterEngine":
        """Load serialized cluster engine."""
        path = Path(path) if path else KMEANS_MODEL_PATH
        return joblib.load(path)
