"""
Script to train all models, generate all figures, and build executed notebooks for Phases 2, 3, and 4.
"""
import os
import re
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nbformat as nbf

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Set paths
ROOT_DIR = Path(__file__).resolve().parent.parent
import sys
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_INTERIM = ROOT_DIR / "data" / "interim" / "job_corpus_clean.csv"
DATA_PROCESSED = ROOT_DIR / "data" / "processed" / "job_corpus_clustered.csv"
MODELS_DIR = ROOT_DIR / "models"
FIG_DIR = ROOT_DIR / "reports" / "figures"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)

# Imports from src
from src.features.text_features import get_tfidf_vectorizer
from src.features.match_features import (
    compute_cosine_similarity,
    compute_skill_overlap,
    extract_skills_from_text,
)
from src.models.recommender import JobRecommender
from src.models.clustering import JobClusterEngine


def main():
    print("--- Starting Execution for Phases 2, 3, and 4 ---")
    df = pd.read_csv(DATA_INTERIM)
    # Ensure parsed_skills is list
    import ast
    def safe_parse_skills(val):
        if isinstance(val, list): return val
        if isinstance(val, str) and val.startswith("["):
            try: return ast.literal_eval(val)
            except: pass
        return [s.strip() for s in str(val).split(";") if s.strip()]

    df["parsed_skills"] = df["parsed_skills"].apply(safe_parse_skills)
    labels = sorted(list(df["Category"].unique()))

    # =========================================================================
    # PHASE 2: SUPERVISED CATEGORY CLASSIFIER (MODEL A)
    # =========================================================================
    print("\n--- PHASE 2: Training Category Classifier ---")
    X_raw = df["composite_text"].fillna("")
    y = df["Category"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )

    vectorizer = get_tfidf_vectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    # Model 1: Logistic Regression
    clf_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf_lr.fit(X_train, y_train)
    y_pred_lr = clf_lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr, average="weighted")
    print(f"Logistic Regression: Accuracy = {acc_lr:.4f}, Weighted F1 = {f1_lr:.4f}")

    # Model 2: Random Forest
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average="weighted")
    print(f"Random Forest: Accuracy = {acc_rf:.4f}, Weighted F1 = {f1_rf:.4f}")

    # Select best model (Logistic Regression typically achieves near 100% on domain TF-IDF)
    best_model = clf_lr if acc_lr >= acc_rf else clf_rf
    best_name = "Logistic Regression" if best_model == clf_lr else "Random Forest"
    best_preds = y_pred_lr if best_model == clf_lr else y_pred_rf

    # Save artifacts
    joblib.dump(best_model, MODELS_DIR / "classifier.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    print(f"Saved best model ({best_name}) and TF-IDF vectorizer to {MODELS_DIR}")

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, best_preds, labels=labels)
    plt.figure(figsize=(13, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(f"Confusion Matrix: Model A ({best_name}) — Accuracy: {max(acc_lr, acc_rf)*100:.1f}%", weight="bold", pad=12)
    plt.xlabel("Predicted Category", weight="bold")
    plt.ylabel("Actual Category", weight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8.5)
    plt.yticks(fontsize=8.5)
    plt.tight_layout()
    cm_path = FIG_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=180)
    plt.close()
    print(f"Saved {cm_path}")

    # Build notebooks/02_resume_classifier.ipynb
    build_classifier_notebook(acc_lr, f1_lr, acc_rf, f1_rf, best_name, labels, y_test, best_preds)

    # =========================================================================
    # PHASE 3: UNSUPERVISED RECOMMENDATION ENGINE
    # =========================================================================
    print("\n--- PHASE 3: Recommender Engine Sanity & Evaluation ---")
    recommender = JobRecommender(df, vectorizer=vectorizer)

    # Build notebooks/03_recommender.ipynb
    build_recommender_notebook(df)

    # =========================================================================
    # PHASE 4: CLUSTERING & SKILL-GAP MODULE
    # =========================================================================
    print("\n--- PHASE 4: K-Means Clustering & Skill-Gap Diagnostic ---")
    X_all = vectorizer.transform(df["composite_text"].fillna(""))

    # Evaluate k from 5 to 20
    k_range = range(5, 21)
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        k_labels = km.fit_predict(X_all)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_all, k_labels))

    best_k_idx = int(np.argmax(silhouettes))
    chosen_k = k_range[best_k_idx]
    print(f"K-Means Evaluation: Optimal K by Silhouette Score = {chosen_k} (Silhouette = {silhouettes[best_k_idx]:.4f})")

    # Fit final clustering engine with 15 clusters (matching 15 domains) or chosen_k
    k_final = 15
    cluster_engine = JobClusterEngine(k=k_final, random_state=42)
    df_clustered = cluster_engine.fit(df, vectorizer=vectorizer)
    df_clustered.to_csv(DATA_PROCESSED, index=False)
    cluster_engine.save(MODELS_DIR / "job_kmeans.pkl")
    print(f"Saved clustered dataset to {DATA_PROCESSED} and engine to {MODELS_DIR / 'job_kmeans.pkl'}")

    # PCA 2D Visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_all.toarray())
    df_pca = pd.DataFrame({"PCA1": X_pca[:, 0], "PCA2": X_pca[:, 1], "Cluster": df_clustered["cluster"], "Category": df_clustered["Category"]})

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="Category", palette="tab20", alpha=0.85, s=65)
    plt.title(f"SmartHire Job Space: PCA 2D Projection (k={k_final})", weight="bold", pad=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=8.5)
    plt.tight_layout()
    pca_path = FIG_DIR / "job_clusters_pca.png"
    plt.savefig(pca_path, dpi=180)
    plt.close()
    print(f"Saved {pca_path}")

    # t-SNE 2D Visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X_all.toarray())
    df_tsne = pd.DataFrame({"tSNE1": X_tsne[:, 0], "tSNE2": X_tsne[:, 1], "Cluster": df_clustered["cluster"], "Category": df_clustered["Category"]})

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_tsne, x="tSNE1", y="tSNE2", hue="Category", palette="tab20", alpha=0.85, s=65)
    plt.title(f"SmartHire Job Space: t-SNE 2D Manifold Embedding", weight="bold", pad=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=8.5)
    plt.tight_layout()
    tsne_path = FIG_DIR / "job_clusters_tsne.png"
    plt.savefig(tsne_path, dpi=180)
    plt.close()
    print(f"Saved {tsne_path}")

    # Build notebooks/04_clustering_topics.ipynb
    build_clustering_notebook(k_range, inertias, silhouettes, chosen_k)

    print("\n--- All modeling and notebook templates constructed successfully! ---")


def build_classifier_notebook(acc_lr, f1_lr, acc_rf, f1_rf, best_name, labels, y_test, best_preds):
    nb = nbf.v4.new_notebook()
    cells = []
    cells.append(nbf.v4.new_markdown_cell("""# SmartHire — Phase 2: Supervised Category Classifier (Model A)
### Objective:
Train a high-precision multi-class text classifier to map any input resume / job description into one of **15 technical job categories** using classical machine learning (TF-IDF + Logistic Regression / Random Forest).
"""))

    cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
if str(Path("..").resolve()) not in sys.path:
    sys.path.insert(0, str(Path("..").resolve()))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.features.text_features import get_tfidf_vectorizer
from src.evaluate import evaluate_classifier

# Load Clean Dataset
df = pd.read_csv("../data/interim/job_corpus_clean.csv")
print(f"Corpus Loaded: {df.shape[0]} records, {df['Category'].nunique()} categories")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Feature Extraction: TF-IDF Vectorization
We use unigrams and bigrams (`ngram_range=(1, 2)`), sublinear term frequency scaling (`sublinear_tf=True`), and English stop word removal.
"""))

    cells.append(nbf.v4.new_code_cell("""X_raw = df['composite_text'].fillna('')
y = df['Category']

# Stratified 80/20 train-test split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)

vectorizer = get_tfidf_vectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

print(f"TF-IDF Feature Space Dimensions: {X_train.shape[1]} terms")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Model Training & Comparative Evaluation
We train both **Multinomial Logistic Regression** (convex linear optimization) and **Random Forest Classifier** (non-linear ensemble).
"""))

    cells.append(nbf.v4.new_code_cell(f"""# Model 1: Logistic Regression
clf_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)

# Model 2: Random Forest
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

results = pd.DataFrame([
    {{"Model": "Logistic Regression", "Accuracy": accuracy_score(y_test, y_pred_lr), "F1 (Weighted)": {f1_lr:.4f}}},
    {{"Model": "Random Forest", "Accuracy": accuracy_score(y_test, y_pred_rf), "F1 (Weighted)": {f1_rf:.4f}}}
])
results
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. Confusion Matrix & Detailed Metrics
Visualizing classification performance across all 15 sectors.
"""))

    cells.append(nbf.v4.new_code_cell("""labels = sorted(list(y.unique()))
cm = confusion_matrix(y_test, y_pred_lr, labels=labels)

plt.figure(figsize=(12, 9))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Model A: Logistic Regression Confusion Matrix", weight='bold')
plt.xlabel("Predicted Category")
plt.ylabel("True Category")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred_lr, zero_division=0))
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Inference Test on a Synthetic Resume
Simulating an incoming uploaded resume to verify real-time category prediction and confidence scoring.
"""))

    cells.append(nbf.v4.new_code_cell("""sample_cv = \"\"\"
Experienced Software Engineer specializing in scalable backend systems.
Proficient in Python, Django, REST APIs, Celery, Docker, and PostgreSQL.
Led microservices migration and automated CI/CD pipeline deployments.
\"\"\"

cv_vec = vectorizer.transform([sample_cv])
pred_cat = clf_lr.predict(cv_vec)[0]
probs = clf_lr.predict_proba(cv_vec)[0]
conf = max(probs) * 100

print(f"Predicted Role Domain: {pred_cat}")
print(f"Confidence Score: {conf:.1f}%")
"""))

    nb.cells = cells
    with open(NOTEBOOKS_DIR / "02_resume_classifier.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Saved notebooks/02_resume_classifier.ipynb")


def build_recommender_notebook(df):
    nb = nbf.v4.new_notebook()
    cells = []
    cells.append(nbf.v4.new_markdown_cell("""# SmartHire — Phase 3: Unsupervised Job Recommendation Engine
### Objective:
Build a content-based recommendation system that ranks job postings by computing **Cosine Similarity** between a candidate's CV and the pre-indexed vector space of job descriptions and required skill stacks.
"""))

    cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
if str(Path("..").resolve()) not in sys.path:
    sys.path.insert(0, str(Path("..").resolve()))

import pandas as pd
import numpy as np
import joblib
from src.models.recommender import JobRecommender
from src.features.text_features import load_vectorizer
from src.evaluate import evaluate_recommender_precision_at_k

# Ingest corpus
df = pd.read_csv("../data/interim/job_corpus_clean.csv")
import ast
df['parsed_skills'] = df['parsed_skills'].apply(lambda v: ast.literal_eval(v) if isinstance(v, str) and v.startswith('[') else [s.strip() for s in str(v).split(';') if s.strip()])

vectorizer = load_vectorizer("../models/tfidf_vectorizer.pkl")
recommender = JobRecommender(df, vectorizer=vectorizer)
print(f"Recommender Initialized with {len(df)} jobs and {len(recommender.skill_universe)} distinct skills.")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Sanity Check 1: Data Science Candidate CV
Testing recommendations on an analytical profile with Python, Machine Learning, and Tableau.
"""))

    cells.append(nbf.v4.new_code_cell("""ds_cv = \"\"\"
Senior Data Scientist with 5 years experience in machine learning, statistical modeling, and data visualization.
Technical Skills: Python, Scikit-learn, TensorFlow, Pandas, NumPy, SQL, Tableau, A/B Testing.
Experienced in training predictive models, feature engineering, and deploying data solutions.
\"\"\"

recs_ds = recommender.recommend(ds_cv, top_n=5)
rec_df_ds = pd.DataFrame(recs_ds)[['title', 'category', 'company', 'location', 'match_score', 'readiness_score', 'primary_gap']]
rec_df_ds
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Sanity Check 2: DevOps & Cloud Engineer CV
Testing recommendations on an infrastructure and automation profile with AWS, Kubernetes, and Terraform.
"""))

    cells.append(nbf.v4.new_code_cell("""devops_cv = \"\"\"
Cloud Infrastructure Engineer with 4 years experience in CI/CD automation, cloud architecture, and container orchestration.
Core Skills: AWS, Docker, Kubernetes, Terraform, Ansible, Linux, Shell Scripting, Jenkins, Monitoring.
Built scalable Kubernetes clusters and automated infrastructure provisioning with Terraform.
\"\"\"

recs_devops = recommender.recommend(devops_cv, top_n=5)
rec_df_devops = pd.DataFrame(recs_devops)[['title', 'category', 'company', 'location', 'match_score', 'readiness_score', 'primary_gap']]
rec_df_devops
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. Sanity Check 3: UI/UX & Graphic Designer CV
Testing recommendations on a creative design profile with Figma, UI/UX, and Typography.
"""))

    cells.append(nbf.v4.new_code_cell("""designer_cv = \"\"\"
Creative Graphic Designer & UI/UX Specialist with expertise in digital product design and branding.
Proficient in Figma, Photoshop, Illustrator, InDesign, UI/UX, Typography, Branding, Creativity.
Crafted brand design systems, high-fidelity prototypes, and user research interfaces.
\"\"\"

recs_design = recommender.recommend(designer_cv, top_n=5)
rec_df_design = pd.DataFrame(recs_design)[['title', 'category', 'company', 'location', 'match_score', 'readiness_score', 'primary_gap']]
rec_df_design
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Recommender Precision@K Evaluation
Simulating Precision@5 across sample profiles:
"""))

    cells.append(nbf.v4.new_code_cell("""p5_ds = evaluate_recommender_precision_at_k([r['category'] for r in recs_ds], 'Data Science', k=5)
p5_devops = evaluate_recommender_precision_at_k([r['category'] for r in recs_devops], 'DevOps Engineer', k=5)
p5_design = evaluate_recommender_precision_at_k([r['category'] for r in recs_design], 'Graphic Designer', k=5)

eval_df = pd.DataFrame([
    {"Profile": "Data Scientist", "Target Category": "Data Science", "Precision@5": f"{p5_ds*100:.0f}%"},
    {"Profile": "DevOps Engineer", "Target Category": "DevOps Engineer", "Precision@5": f"{p5_devops*100:.0f}%"},
    {"Profile": "Graphic Designer", "Target Category": "Graphic Designer", "Precision@5": f"{p5_design*100:.0f}%"}
])
eval_df
"""))

    nb.cells = cells
    with open(NOTEBOOKS_DIR / "03_recommender.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Saved notebooks/03_recommender.ipynb")


def build_clustering_notebook(k_range, inertias, silhouettes, chosen_k):
    nb = nbf.v4.new_notebook()
    cells = []
    cells.append(nbf.v4.new_markdown_cell("""# SmartHire — Phase 4: Unsupervised Clustering & Skill-Gap Diagnostic
### Objective:
Discover natural job families using **K-Means Clustering** on job vectors, determine optimal $k$ via Elbow Method and Silhouette Analysis, project the manifold in 2D with **PCA and t-SNE**, and build the **Skill-Gap Module** that calculates candidate readiness scores and identifies actionable skill gaps.
"""))

    cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
if str(Path("..").resolve()) not in sys.path:
    sys.path.insert(0, str(Path("..").resolve()))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import joblib

from src.features.text_features import load_vectorizer
from src.models.clustering import JobClusterEngine

# Load dataset
df = pd.read_csv("../data/interim/job_corpus_clean.csv")
import ast
df['parsed_skills'] = df['parsed_skills'].apply(lambda v: ast.literal_eval(v) if isinstance(v, str) and v.startswith('[') else [s.strip() for s in str(v).split(';') if s.strip()])
vectorizer = load_vectorizer("../models/tfidf_vectorizer.pkl")
X = vectorizer.transform(df['composite_text'].fillna(''))
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Optimal Cluster Selection: Elbow & Silhouette Analysis
Evaluating $k \in [5, 20]$ using Inertia (within-cluster sum of squares) and Silhouette Score.
"""))

    cells.append(nbf.v4.new_code_cell(f"""k_vals = list(range(5, 21))
inertias = {inertias}
silhouettes = {silhouettes}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# Elbow Method
ax1.plot(k_vals, inertias, 'bo-', linewidth=2, markersize=6)
ax1.set_title("Elbow Method: Inertia vs Number of Clusters k", weight='bold')
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia")
ax1.grid(True)

# Silhouette Score
ax2.plot(k_vals, silhouettes, 'ro-', linewidth=2, markersize=6)
ax2.set_title("Silhouette Score vs Number of Clusters k", weight='bold')
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Silhouette Score")
ax2.grid(True)

plt.tight_layout()
plt.show()

print(f"Optimal k by Silhouette: k = {chosen_k} (Silhouette = {max(silhouettes):.4f})")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. 2D Manifold Visualizations: PCA & t-SNE
Mapping the high-dimensional TF-IDF vector space into two dimensions.
"""))

    cells.append(nbf.v4.new_code_cell("""# Fit final model with k=15
cluster_engine = JobClusterEngine(k=15, random_state=42)
df_clustered = cluster_engine.fit(df, vectorizer=vectorizer)

# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X.toarray())

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_clustered['Category'], palette='tab20', alpha=0.8)
plt.title("PCA 2D Projection of Job Clusters", weight='bold')
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. Skill-Gap Diagnostic Module in Action
Testing the candidate skill-gap report:
- Maps candidate to nearest cluster centroid
- Calculates Readiness Score (% overlap against cluster core skills)
- Produces a concise, single-sentence career guidance action item highlighting the single biggest gap to close.
"""))

    cells.append(nbf.v4.new_code_cell("""test_resume = \"\"\"
Aspiring Data Scientist with foundational skills in Python and SQL.
Completed projects in Pandas data cleaning and basic exploratory data analysis.
Looking for junior analytical and machine learning roles.
\"\"\"

gap_report = cluster_engine.get_skill_gap_report(test_resume)

print("="*65)
print(f"Target Cluster Role: {gap_report['target_role_domain']}")
print(f"Candidate Skills Found: {gap_report['candidate_skills']}")
print(f"Cluster Core Competencies: {gap_report['cluster_core_skills']}")
print(f"Matched Skills: {gap_report['matched_skills']}")
print(f"Missing Skills: {gap_report['missing_skills']}")
print(f"Readiness Score: {gap_report['readiness_score']}%")
print(f"Recommendation: {gap_report['single_line_explanation']}")
print("="*65)
"""))

    nb.cells = cells
    with open(NOTEBOOKS_DIR / "04_clustering_topics.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Saved notebooks/04_clustering_topics.ipynb")


if __name__ == "__main__":
    main()
