"""
Unit tests for SmartHire text cleaning, feature engineering, and metrics.
"""
import unittest
import numpy as np

from src.data.preprocess import clean_text, parse_skills, parse_experience, parse_salary
from src.features.match_features import compute_skill_overlap, compute_cosine_similarity
from src.evaluate import evaluate_recommender_precision_at_k


class TestPreprocess(unittest.TestCase):
    def test_clean_text(self):
        sample = "Senior Python Developer with AWS & Docker! Check https://github.com"
        cleaned = clean_text(sample)
        self.assertNotIn("https", cleaned)
        self.assertIn("python", cleaned)
        self.assertIn("aws", cleaned)
        self.assertIn("docker", cleaned)

    def test_parse_skills(self):
        skills_str = "Python; Machine Learning; SQL, Pandas; "
        parsed = parse_skills(skills_str)
        self.assertEqual(len(parsed), 4)
        self.assertIn("Python", parsed)
        self.assertIn("Machine Learning", parsed)

    def test_parse_experience(self):
        min_e, max_e, avg_e = parse_experience("3-6 years")
        self.assertEqual(min_e, 3.0)
        self.assertEqual(max_e, 6.0)
        self.assertEqual(avg_e, 4.5)

    def test_parse_salary(self):
        min_s, max_s, avg_s = parse_salary("12-18")
        self.assertEqual(min_s, 12.0)
        self.assertEqual(max_s, 18.0)
        self.assertEqual(avg_s, 15.0)


class TestMatchFeatures(unittest.TestCase):
    def test_compute_skill_overlap(self):
        cand = ["Python", "SQL", "Git"]
        target = ["Python", "SQL", "Docker", "Kubernetes"]
        res = compute_skill_overlap(cand, target)
        self.assertEqual(res["matched_skills"], ["Python", "SQL"])
        self.assertEqual(res["missing_skills"], ["Docker", "Kubernetes"])
        self.assertEqual(res["readiness_score"], 50.0)
        self.assertEqual(res["primary_gap"], "Docker")

    def test_cosine_similarity(self):
        v1 = np.array([[1.0, 0.0]])
        v2 = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]])
        sims = compute_cosine_similarity(v1, v2)
        self.assertAlmostEqual(sims[0], 1.0, places=4)
        self.assertAlmostEqual(sims[1], 0.0, places=4)
        self.assertGreater(sims[2], 0.7)

    def test_precision_at_k(self):
        recs = ["Data Science", "Data Science", "Web Development", "Data Science", "Sales"]
        prec = evaluate_recommender_precision_at_k(recs, "Data Science", k=5)
        self.assertEqual(prec, 0.6)


if __name__ == "__main__":
    unittest.main()
