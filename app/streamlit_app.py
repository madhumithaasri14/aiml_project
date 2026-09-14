"""
SmartHire — Classical ML-Powered Career Matching & Readiness Engine.
A Streamlit Web Portal for Resume Classification, Top-N Job Recommendations,
and Cluster-Based Skill-Gap Diagnostics.
"""
import sys
from pathlib import Path
import ast

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from src.config import (
    JOB_CORPUS_CLEAN,
    CLASSIFIER_PATH,
    TFIDF_VECTORIZER_PATH,
    KMEANS_MODEL_PATH,
)
from src.features.text_features import load_vectorizer
from src.features.match_features import extract_skills_from_text, compute_skill_overlap
from src.models.classifier import predict_category
from src.models.recommender import JobRecommender
from src.models.clustering import JobClusterEngine
from src.parsing.resume_parser import parse_resume

# ==============================================================================
# PAGE CONFIGURATION & METADATA
# ==============================================================================
st.set_page_config(
    page_title="SmartHire | Career Matching & Readiness Engine",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CUSTOM MODERN CSS THEME (Deep Slate, Emerald, Indigo, Glassmorphism)
# ==============================================================================
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0d1527 100%);
        color: #e2e8f0;
    }
    
    /* Header Container */
    .hero-container {
        padding: 1.8rem 2.2rem;
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        backdrop-filter: blur(12px);
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }
    
    .hero-tagline {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
    }
    
    .badge-classical-ml {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        background: rgba(99, 102, 241, 0.18);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #a5b4fc;
        margin-top: 0.5rem;
    }
    
    /* Metrics Cards */
    .metric-box {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-box:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.45);
    }
    
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0.2rem 0;
    }
    .metric-caption {
        font-size: 0.82rem;
        color: #64748b;
    }
    
    /* Callout Recommendation */
    .callout-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 95, 70, 0.18) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-left: 5px solid #10b981;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1.2rem 0;
    }
    .callout-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #34d399;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .callout-text {
        font-size: 1rem;
        color: #ecfdf5;
        font-weight: 500;
        line-height: 1.4;
    }
    
    /* Job Cards */
    .job-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        transition: all 0.2s ease-in-out;
    }
    .job-card:hover {
        border-color: #38bdf8;
        box-shadow: 0 8px 20px -4px rgba(56, 189, 248, 0.2);
    }
    
    .job-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.6rem;
    }
    .job-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .job-company {
        font-size: 0.95rem;
        color: #94a3b8;
        font-weight: 500;
    }
    .score-badge {
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
    }
    
    /* Skill Badges */
    .skill-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .skill-matched {
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.5);
        color: #6ee7b7;
    }
    .skill-missing {
        background: rgba(245, 158, 11, 0.18);
        border: 1px solid rgba(245, 158, 11, 0.45);
        color: #fcd34d;
    }
    .skill-neutral {
        background: rgba(148, 163, 184, 0.15);
        border: 1px solid rgba(148, 163, 184, 0.25);
        color: #cbd5e1;
    }
    
    /* Meta details */
    .meta-tag {
        font-size: 0.85rem;
        color: #cbd5e1;
        margin-right: 1.2rem;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# RESOURCE LOADERS (Cached)
# ==============================================================================
@st.cache_resource
def load_corpus():
    """Load cleaned job postings corpus."""
    df = pd.read_csv(JOB_CORPUS_CLEAN)
    def parse_s(val):
        if isinstance(val, list): return val
        if isinstance(val, str) and val.startswith("["):
            try: return ast.literal_eval(val)
            except: pass
        return [s.strip() for s in str(val).split(";") if s.strip()]
    df["parsed_skills"] = df["parsed_skills"].apply(parse_s)
    return df

@st.cache_resource
def load_recommender(_df):
    """Load fitted TF-IDF and recommender engine."""
    vectorizer = load_vectorizer(TFIDF_VECTORIZER_PATH)
    rec = JobRecommender(_df, vectorizer=vectorizer)
    return rec, vectorizer

@st.cache_resource
def load_models():
    """Load classifier and clustering engine."""
    from pathlib import Path
    import joblib
    classifier = joblib.load(CLASSIFIER_PATH)
    cluster_engine = JobClusterEngine.load(KMEANS_MODEL_PATH)
    return classifier, cluster_engine


import traceback

try:
    df_jobs = load_corpus()
    recommender, vectorizer = load_recommender(df_jobs)
    classifier, cluster_engine = load_models()
    engine_ready = True
except Exception as e:
    engine_ready = False
    load_err = f"{e}\n{traceback.format_exc()}"


# ==============================================================================
# SAMPLE RESUMES FOR QUICK DEMONSTRATION
# ==============================================================================
SAMPLE_RESUMES = {
    "Select a preloaded CV or upload your own...": "",
    "Data Scientist / ML Engineer": """
Data Scientist with 4 years of hands-on experience in predictive modeling, statistical analysis, and machine learning pipelines.
Technical Proficiencies: Python, Scikit-learn, TensorFlow, SQL, Pandas, NumPy, Tableau, A/B Testing, Data Visualization.
Experience:
- Built supervised machine learning classifiers and recommendation systems.
- Engineered ETL data pipelines and conducted hypothesis testing for growth metrics.
- Designed executive dashboards in Tableau and optimized SQL queries on big data warehouses.
Certifications: Google Data Analytics Professional Certificate.
""",
    "DevOps & Cloud Engineer": """
DevOps Engineer with 5 years experience automating infrastructure, CI/CD pipelines, and multi-cloud container orchestration.
Core Competencies: AWS, Docker, Kubernetes, Terraform, Ansible, Linux, Shell Scripting, Jenkins, CI/CD, Monitoring.
Key Achievements:
- Automated cloud infrastructure deployments using Terraform and Ansible across AWS VPCs.
- Architected Kubernetes microservice clusters with zero-downtime rolling updates.
- Configured Jenkins CI/CD automation and Prometheus/Grafana system health monitoring.
""",
    "Full-Stack Web Developer": """
Full Stack Web Developer with 3 years experience building responsive web applications and RESTful backend APIs.
Core Skills: JavaScript, TypeScript, React, Node.js, Express.js, MongoDB, CSS, HTML, Responsive Design, Git, REST API.
Experience:
- Developed single-page applications with React and TypeScript, integrating high-throughput REST APIs.
- Built scalable backend microservices using Node.js and Express with MongoDB NoSQL storage.
- Maintained clean version control workflows in Git and styled modern component libraries.
""",
    "Graphic Designer & UI/UX": """
Creative Graphic Designer and UI/UX Designer with 4 years experience in digital branding, web UI, and visual identity.
Key Strengths: Figma, Photoshop, Illustrator, InDesign, UI/UX, Typography, Branding, Creativity, Communication.
Experience:
- Created end-to-end design systems, typography guidelines, and design libraries in Figma.
- Designed marketing collateral, social media banners, and vector assets using Illustrator and Photoshop.
- Conducted user testing and created high-fidelity interactive wireframes for enterprise portals.
""",
    "Cybersecurity / Network Security": """
Network Security Specialist with 6 years experience in threat hunting, penetration testing, and security infrastructure.
Core Competencies: Firewalls, Penetration Testing, Network Protocols, Linux, Cryptography, Python, IDS/IPS, SIEM, Vulnerability Assessment.
Experience:
- Administered next-gen enterprise firewalls and deployed Snort IDS/IPS threat monitoring rules.
- Performed regular vulnerability assessments, ethical penetration tests, and SIEM log correlation.
- Scripted automated network audit scripts using Python and hardened Linux production environments.
"""
}


# ==============================================================================
# SIDEBAR CONTROLS & MODEL ARCHITECTURE
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/null/briefcase.png", width=70)
    st.markdown("### **SmartHire Controls**")
    st.caption("Classical ML-Powered Matching & Analytics")
    
    st.markdown("---")
    st.markdown("#### **1. Recommendation Filters**")
    top_n = st.slider("Number of Job Matches (Top-N):", min_value=3, max_value=15, value=5, step=1)
    
    all_cats = ["All Categories"] + sorted(list(df_jobs["Category"].unique())) if engine_ready else ["All Categories"]
    filter_category = st.selectbox("Filter Target Category (Optional):", all_cats)
    actual_cat_filter = None if filter_category == "All Categories" else filter_category
    
    min_sim = st.slider("Minimum Match Threshold (%):", min_value=0, max_value=50, value=5, step=5) / 100.0

    st.markdown("---")
    st.markdown("#### **2. Quick Demo Profiles**")
    chosen_sample = st.selectbox("Load Sample CV Template:", list(SAMPLE_RESUMES.keys()))

    st.markdown("---")
    st.markdown("#### **Classical ML Pipeline**")
    st.markdown("""
    - **Vector Space**: TF-IDF (1-2 ngrams, 5,000 terms)
    - **Classifier**: Supervised Multinomial Logistic Regression
    - **Recommender**: Unsupervised Cosine Vector Similarity
    - **Clustering**: K-Means ($k=15$) + Silhouette Validation
    - **Constraints**: 100% Classical ML (Zero LLMs / GenAI)
    """)


# ==============================================================================
# MAIN PAGE HERO HEADER
# ==============================================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        <span>SmartHire</span>
    </div>
    <div class="hero-tagline">
        Resume-to-Job Matching, Role Domain Classification & Targeted Career Readiness Diagnostics
    </div>
    <span class="badge-classical-ml">Classical Machine Learning Architecture &bull; TF-IDF &bull; Cosine Similarity &bull; K-Means Clustering</span>
</div>
""", unsafe_allow_html=True)


if not engine_ready:
    st.error(f"Failed to load machine learning artifacts. Error: {load_err}")
    st.stop()


# ==============================================================================
# RESUME INPUT SECTION (Upload or Sample)
# ==============================================================================
st.markdown("### 📄 **Step 1: Provide Candidate Resume (CV)**")

col_upload, col_sample = st.columns([1.8, 1.2])

uploaded_file = None
with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Resume Document (PDF, DOCX, or TXT):",
        type=["pdf", "docx", "txt"],
        help="Upload a candidate CV to extract text and analyze against the SmartHire machine learning engine."
    )

raw_cv_text = ""
if uploaded_file is not None:
    try:
        raw_cv_text = parse_resume(uploaded_file, uploaded_file.name)
        st.success(f"Successfully extracted text from **{uploaded_file.name}** ({len(raw_cv_text.split())} words)")
    except Exception as e:
        st.error(f"Error parsing uploaded file: {e}")
elif chosen_sample != "Select a preloaded CV or upload your own...":
    raw_cv_text = SAMPLE_RESUMES[chosen_sample].strip()
    with col_sample:
        st.info(f"Using template: **{chosen_sample}**")

# Text Area for manual inspection or edits
with st.expander("📝 View or Edit Candidate Resume Text", expanded=bool(raw_cv_text and len(raw_cv_text) < 1500)):
    raw_cv_text = st.text_area("Resume Content:", value=raw_cv_text, height=180, placeholder="Paste or edit resume text here...")


# ==============================================================================
# ANALYSIS & PREDICTION
# ==============================================================================
if raw_cv_text.strip():
    with st.spinner("Analyzing candidate CV across classical ML models..."):
        # 1. Supervised Category Classification
        pred_category, confidence, prob_dict = predict_category(raw_cv_text, model=classifier, vectorizer=vectorizer)
        
        # 2. Unsupervised Clustering & Skill-Gap Analysis
        skill_gap = cluster_engine.get_skill_gap_report(raw_cv_text)
        
        # 3. Unsupervised Top-N Recommendations
        recommendations = recommender.recommend(
            raw_cv_text,
            top_n=top_n,
            filter_category=actual_cat_filter,
            min_similarity=min_sim,
        )

    # --------------------------------------------------------------------------
    # SUMMARY KPI CARDS
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📊 **Step 2: Candidate ML Assessment Overview**")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Predicted Role Domain</div>
            <div class="metric-value" style="color:#38bdf8; font-size:1.45rem;">{pred_category}</div>
            <div class="metric-caption">Confidence: <strong>{confidence*100:.1f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        readiness = skill_gap["readiness_score"]
        readiness_color = "#10b981" if readiness >= 60 else "#f59e0b" if readiness >= 30 else "#ef4444"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Cluster Readiness Score</div>
            <div class="metric-value" style="color:{readiness_color};">{readiness}%</div>
            <div class="metric-caption">Cluster: <strong>{skill_gap['target_role_domain']}</strong></div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        n_matched = len(skill_gap["matched_skills"])
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Core Skills Matched</div>
            <div class="metric-value" style="color:#34d399;">{n_matched}</div>
            <div class="metric-caption">Out of {len(skill_gap['cluster_core_skills'])} core skills</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        primary_gap = skill_gap["primary_gap"]
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Primary Skill Gap</div>
            <div class="metric-value" style="color:#fbbf24; font-size:1.35rem;">{primary_gap}</div>
            <div class="metric-caption">Single biggest hurdle to close</div>
        </div>
        """, unsafe_allow_html=True)

    # One-line Targeted Career Guidance Callout
    st.markdown(f"""
    <div class="callout-box">
        <div class="callout-title">💡 Actionable Career Guidance Insight</div>
        <div class="callout-text">{skill_gap['single_line_explanation']}</div>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # TABS: DETAILED BREAKDOWN & RECOMMENDATIONS
    # --------------------------------------------------------------------------
    tab_recs, tab_gap, tab_probs = st.tabs([
        f"🎯 Top-{len(recommendations)} Matching Jobs",
        "🔍 Cluster Skill-Gap Breakdown",
        "📈 Domain Probability Distribution",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: TOP-N MATCHING JOBS
    # --------------------------------------------------------------------------
    with tab_recs:
        if not recommendations:
            st.warning("No jobs matched your current filter criteria. Try lowering the minimum similarity threshold.")
        else:
            st.markdown(f"Found **{len(recommendations)} jobs** mathematically closest to the candidate's vector profile:")
            
            for idx, job in enumerate(recommendations, 1):
                # Build skill badges
                badges_html = []
                for sk in job["skills"]:
                    if sk in job["matched_skills"]:
                        badges_html.append(f'<span class="skill-badge skill-matched">✓ {sk}</span>')
                    else:
                        badges_html.append(f'<span class="skill-badge skill-missing">− {sk}</span>')
                badges_str = "".join(badges_html)

                st.markdown(f"""
                <div class="job-card">
                    <div class="job-header">
                        <div>
                            <div class="job-title">#{idx} {job['title']}</div>
                            <div class="job-company">🏢 {job['company']} &bull; 📍 {job['location']} &bull; 🏷️ {job['category']}</div>
                        </div>
                        <div class="score-badge">
                            {job['match_score']}% Match
                        </div>
                    </div>
                    <div style="margin: 0.6rem 0;">
                        <span class="meta-tag">⏳ <strong>Exp:</strong> {job['experience_required']}</span>
                        <span class="meta-tag">💰 <strong>Salary:</strong> ₹{job['salary_lpa']} LPA</span>
                        <span class="meta-tag">🎯 <strong>Readiness:</strong> {job['readiness_score']}%</span>
                    </div>
                    <div style="font-size:0.92rem; color:#cbd5e1; margin: 0.6rem 0; line-height:1.45;">
                        {job['description']}
                    </div>
                    <div style="margin-top: 0.7rem;">
                        <span style="font-size:0.8rem; color:#94a3b8; font-weight:600; margin-right:0.4rem;">Skills:</span>
                        {badges_str}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # TAB 2: SKILL GAP REPORT
    # --------------------------------------------------------------------------
    with tab_gap:
        st.markdown(f"#### Cluster Diagnostics for **{skill_gap['target_role_domain']}**")
        st.markdown("The candidate's TF-IDF vector is assigned to this job family by K-Means clustering. Here is how their profile stacks up against the benchmark:")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### ✅ **Candidate Matched Core Skills**")
            if skill_gap["matched_skills"]:
                for s in skill_gap["matched_skills"]:
                    st.markdown(f"- <span style='color:#34d399; font-weight:600;'>{s}</span>", unsafe_allow_html=True)
            else:
                st.write("No exact core cluster skills detected.")

        with c2:
            st.markdown("##### ⚠️ **Missing Core Skills to Acquire**")
            if skill_gap["missing_skills"]:
                for s in skill_gap["missing_skills"]:
                    st.markdown(f"- <span style='color:#fbbf24; font-weight:600;'>{s}</span>", unsafe_allow_html=True)
            else:
                st.write("Zero missing skills — candidate has full cluster coverage!")

        st.markdown("---")
        st.markdown("##### **Candidate Profile Skill Extraction**")
        st.caption("Skills automatically identified within candidate text:")
        if skill_gap["candidate_skills"]:
            cand_badges = "".join([f'<span class="skill-badge skill-neutral">{s}</span>' for s in skill_gap["candidate_skills"]])
            st.markdown(cand_badges, unsafe_allow_html=True)
        else:
            st.write("No recognized technical skills found in resume.")

    # --------------------------------------------------------------------------
    # TAB 3: PROBABILITY DISTRIBUTION
    # --------------------------------------------------------------------------
    with tab_probs:
        st.markdown("#### **Model A: Multi-Class Domain Probability Spectrum**")
        st.caption("Calibrated posterior probabilities across the 15 job categories:")
        
        prob_series = pd.Series(prob_dict).sort_values(ascending=False).head(8)
        
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.set_facecolor("#1e293b")
        fig.patch.set_facecolor("#1e293b")
        
        y_pos = np.arange(len(prob_series))
        bars = ax.barh(y_pos, prob_series.values * 100, color="#6366f1", edgecolor="#818cf8")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(prob_series.index, color="#e2e8f0", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Probability (%)", color="#94a3b8", fontsize=10)
        ax.set_xlim(0, max(prob_series.values * 100) + 15)
        ax.tick_params(colors="#94a3b8")
        ax.grid(color="#334155", linestyle="--", alpha=0.6)
        
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 1.2, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", color="#f8fafc", fontweight="bold", fontsize=9.5)
            
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

else:
    # --------------------------------------------------------------------------
    # EMPTY STATE / WELCOME DASHBOARD
    # --------------------------------------------------------------------------
    st.info("👆 Please upload a candidate resume document or select a preloaded sample CV in the sidebar to begin matching!")
    
    st.markdown("### 🏢 **Job Market Database Overview**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Indexed Jobs", f"{len(df_jobs)}")
    m2.metric("Job Categories", f"{df_jobs['Category'].nunique()}")
    m3.metric("Locations Covered", f"{df_jobs['Location'].nunique()}")
    m4.metric("Avg Market Salary", f"₹{df_jobs['avg_salary_lpa'].mean():.1f} LPA")

    st.markdown("#### **Available Job Domains**")
    cats = sorted(df_jobs["Category"].unique())
    cols = st.columns(3)
    for i, c in enumerate(cats):
        cols[i % 3].markdown(f"• **{c}** ({len(df_jobs[df_jobs['Category']==c])} postings)")
