# SmartHire — Resume-to-Job Matching & Career Guidance Engine

A machine learning system combining supervised and unsupervised classical machine learning techniques to recommend jobs, predict role categories, and generate personalized skill-gap and readiness reports.

Built strictly with **classical machine learning** (no LLMs, no generative AI, no live scraping) in compliance with the industrial project specification.

---

## Key Features

1. **Category Classification (Supervised - Model A)**:
   - Predicts candidate job domain across 15 technical sectors using TF-IDF feature extraction and Multinomial Logistic Regression / Random Forest.
   - Outputs domain category and class confidence distributions.
2. **Content-Based Job Recommendation (Unsupervised - Core Engine)**:
   - High-dimensional vector space modeling of job descriptions, required skills, and candidate resumes.
   - Cosine similarity ranking returning Top-N relevant jobs with match scores.
3. **Job Clustering & Skill-Gap Analysis (Unsupervised)**:
   - K-Means clustering identifying natural job families with PCA/t-SNE cluster visualization.
   - Calculates a candidate **Readiness Score (%)** against cluster core competencies.
   - Delivers actionable, single-sentence career guidance highlighting the primary skill gap to close.
4. **SmartHire Web Portal (Streamlit)**:
   - Interactive portal with modern visual aesthetics (deep slate & emerald theme, metric cards, skill badges).
   - Instant PDF, DOCX, and plain-text CV parsing.

---

## Directory Structure

```
smarthire/
├── README.md                          # Project intro, setup, instructions
├── requirements.txt                   # Pinned project dependencies
├── .gitignore                         # Git exclusion rules
│
├── data/
│   ├── raw/                           # Original job postings & datasets
│   ├── interim/                       # Cleaned and parsed job corpus
│   └── processed/                     # Clustered and model-ready data
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis & insights
│   ├── 02_resume_classifier.ipynb     # Supervised category classifier
│   ├── 03_recommender.ipynb           # Unsupervised similarity ranking
│   └── 04_clustering_topics.ipynb     # Unsupervised K-Means clustering & skill-gap
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # Global paths, constants, parameters
│   ├── data/
│   │   ├── load_data.py               # Data loading routines
│   │   └── preprocess.py              # Text cleaning & feature parsing
│   ├── features/
│   │   ├── text_features.py           # TF-IDF vectorizers
│   │   └── match_features.py          # Skill overlap & similarity metrics
│   ├── models/
│   │   ├── classifier.py              # Supervised category model
│   │   ├── recommender.py             # Cosine similarity recommender
│   │   └── clustering.py              # K-Means clustering & skill gap
│   ├── parsing/
│   │   └── resume_parser.py           # PDF / DOCX text extraction
│   └── evaluate.py                    # Evaluation metrics & reports
│
├── models/                            # Saved .pkl serialized artifacts
│   ├── classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   └── job_kmeans.pkl
│
├── app/
│   └── streamlit_app.py               # Polished web UI demo
│
├── reports/
│   └── figures/                       # EDA charts, confusion matrix, PCA/t-SNE plots
│
└── tests/
    └── test_features.py               # Unit tests
```

---

## Setup & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 3. Launch Streamlit Web Portal
```bash
streamlit run app/streamlit_app.py
```
