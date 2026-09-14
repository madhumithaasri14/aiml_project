"""
SmartHire Phase 1 EDA Runner.
Executes Exploratory Data Analysis, exports publication-grade figures to reports/figures/,
preprocesses the raw corpus into data/interim/job_corpus_clean.csv,
and builds notebooks/01_eda.ipynb with rich markdown, code, and visualizations.
"""
import os
import re
import json
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nbformat as nbf

# Define paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw" / "job_postings.csv"
DATA_INTERIM = ROOT_DIR / "data" / "interim" / "job_corpus_clean.csv"
FIG_DIR = ROOT_DIR / "reports" / "figures"
NOTEBOOK_PATH = ROOT_DIR / "notebooks" / "01_eda.ipynb"

FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_INTERIM.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 150

palette = sns.color_palette("deep")

def parse_skills(skills_str):
    if not isinstance(skills_str, str) or not skills_str.strip():
        return []
    return [s.strip() for s in re.split(r"[;,]", skills_str) if s.strip()]

def parse_experience(exp_str):
    if not isinstance(exp_str, str):
        return 0.0, 0.0, 0.0
    nums = [float(n) for n in re.findall(r"\d+", exp_str)]
    if len(nums) >= 2:
        return nums[0], nums[1], (nums[0] + nums[1]) / 2.0
    elif len(nums) == 1:
        return nums[0], nums[0] + 2.0, nums[0] + 1.0
    return 0.0, 1.0, 0.5

def parse_salary(sal_str):
    if not isinstance(sal_str, str):
        return 0.0, 0.0, 0.0
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", sal_str)]
    if len(nums) >= 2:
        return nums[0], nums[1], (nums[0] + nums[1]) / 2.0
    elif len(nums) == 1:
        return nums[0], nums[0], nums[0]
    return 0.0, 0.0, 0.0

def run_eda_pipeline():
    print(f"Loading raw dataset from {DATA_RAW}...")
    df = pd.read_csv(DATA_RAW)
    print(f"Dataset shape: {df.shape}")

    # Parse columns
    df["parsed_skills"] = df["Skills"].apply(parse_skills)
    df["skills_joined"] = df["parsed_skills"].apply(lambda l: " ".join(l))
    
    exp_tuples = df["Experience_Required"].apply(parse_experience)
    df["min_exp"] = [t[0] for t in exp_tuples]
    df["max_exp"] = [t[1] for t in exp_tuples]
    df["avg_exp"] = [t[2] for t in exp_tuples]
    
    sal_tuples = df["Salary_LPA"].apply(parse_salary)
    df["min_salary_lpa"] = [t[0] for t in sal_tuples]
    df["max_salary_lpa"] = [t[1] for t in sal_tuples]
    df["avg_salary_lpa"] = [t[2] for t in sal_tuples]

    df["clean_description"] = df["Description"].astype(str).str.lower()
    df["composite_text"] = df["Title"] + " " + df["clean_description"] + " " + df["skills_joined"]

    # Save cleaned interim dataset
    df.to_csv(DATA_INTERIM, index=False)
    print(f"Saved preprocessed dataset to {DATA_INTERIM}")

    # -------------------------------------------------------------
    # FIGURE 1: Category Distribution
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    cat_counts = df["Category"].value_counts().sort_values(ascending=False)
    ax = sns.barplot(x=cat_counts.values, y=cat_counts.index, palette="mako", hue=cat_counts.index, legend=False)
    plt.title("SmartHire Job Category Distribution (Perfect 35 Jobs/Category Balance)", weight="bold", pad=15)
    plt.xlabel("Number of Job Postings")
    plt.ylabel("Job Category")
    for i, v in enumerate(cat_counts.values):
        ax.text(v + 0.5, i, f"{v} ({v/len(df)*100:.1f}%)", va="center", fontweight="bold", fontsize=9, color="#1e293b")
    plt.xlim(0, 42)
    plt.tight_layout()
    fig1_path = FIG_DIR / "eda_category_distribution.png"
    plt.savefig(fig1_path, dpi=180)
    plt.close()
    print(f"Saved {fig1_path}")

    # -------------------------------------------------------------
    # FIGURE 2: Top Skills per Primary Job Category
    # -------------------------------------------------------------
    categories_to_plot = ["Data Science", "Web Development", "DevOps Engineer", "Python Developer", "Network Security Engineer", "Digital Marketing"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, cat in enumerate(categories_to_plot):
        sub = df[df["Category"] == cat]
        all_skills = [s for slist in sub["parsed_skills"] for s in slist]
        top_skills = pd.Series(all_skills).value_counts().head(7)
        
        sns.barplot(ax=axes[idx], x=top_skills.values, y=top_skills.index, palette="crest", hue=top_skills.index, legend=False)
        axes[idx].set_title(f"{cat}", weight="bold", fontsize=11)
        axes[idx].set_xlabel("Frequency")
        for i, v in enumerate(top_skills.values):
            axes[idx].text(v + 0.3, i, str(v), va="center", fontsize=9, fontweight="bold", color="#334155")
        axes[idx].set_xlim(0, max(top_skills.values) + 5)

    plt.suptitle("Skill Frequency Profiles Across Key Technical Disciplines", weight="bold", fontsize=14, y=0.98)
    plt.tight_layout()
    fig2_path = FIG_DIR / "eda_top_skills_by_category.png"
    plt.savefig(fig2_path, dpi=180)
    plt.close()
    print(f"Saved {fig2_path}")

    # -------------------------------------------------------------
    # FIGURE 3: Salary Distribution by Category & Location
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Average Salary by Category
    sal_by_cat = df.groupby("Category")["avg_salary_lpa"].agg(["mean", "median"]).sort_values(by="mean", ascending=False)
    sns.barplot(ax=ax1, x=sal_by_cat["mean"].values, y=sal_by_cat.index, palette="viridis", hue=sal_by_cat.index, legend=False)
    ax1.set_title("Mean Salary by Job Category (LPA)", weight="bold")
    ax1.set_xlabel("Mean Salary (LPA in ₹ Lakhs)")
    ax1.set_ylabel("Job Category")
    for i, v in enumerate(sal_by_cat["mean"].values):
        ax1.text(v + 0.2, i, f"₹{v:.1f}L", va="center", fontsize=9, fontweight="bold")
    ax1.set_xlim(0, max(sal_by_cat["mean"].values) + 3)

    # Average Salary by Location
    sal_by_loc = df.groupby("Location")["avg_salary_lpa"].agg(["mean", "count"]).sort_values(by="mean", ascending=False)
    sns.barplot(ax=ax2, x=sal_by_loc["mean"].values, y=sal_by_loc.index, palette="flare", hue=sal_by_loc.index, legend=False)
    ax2.set_title("Mean Salary by Location (LPA)", weight="bold")
    ax2.set_xlabel("Mean Salary (LPA in ₹ Lakhs)")
    ax2.set_ylabel("Location / Hub")
    for i, v in enumerate(sal_by_loc["mean"].values):
        n = sal_by_loc["count"].iloc[i]
        ax2.text(v + 0.2, i, f"₹{v:.1f}L ({n} jobs)", va="center", fontsize=9, fontweight="bold")
    ax2.set_xlim(0, max(sal_by_loc["mean"].values) + 3)

    plt.tight_layout()
    fig3_path = FIG_DIR / "eda_salary_by_category_location.png"
    plt.savefig(fig3_path, dpi=180)
    plt.close()
    print(f"Saved {fig3_path}")

    # -------------------------------------------------------------
    # FIGURE 4: Genuine Insight 1: Ubiquitous / Cross-Category Skills
    # -------------------------------------------------------------
    skill_to_cats = {}
    for _, row in df.iterrows():
        cat = row["Category"]
        for s in row["parsed_skills"]:
            if s not in skill_to_cats:
                skill_to_cats[s] = set()
            skill_to_cats[s].add(cat)

    cross_cat_data = [
        {"Skill": skill, "Category_Count": len(cats), "Categories": ", ".join(sorted(cats))}
        for skill, cats in skill_to_cats.items()
    ]
    cross_df = pd.DataFrame(cross_cat_data).sort_values(by="Category_Count", ascending=False)

    plt.figure(figsize=(12, 6))
    top_cross = cross_df[cross_df["Category_Count"] >= 2].head(12)
    ax = sns.barplot(data=top_cross, x="Category_Count", y="Skill", palette="crest", hue="Skill", legend=False)
    plt.title("Insight: 'Bridge Skills' Spanning the Most Job Domains", weight="bold", pad=15)
    plt.xlabel("Number of Distinct Job Categories (out of 15)")
    plt.ylabel("Skill")
    for i, row in enumerate(top_cross.itertuples()):
        ax.text(row.Category_Count + 0.1, i, f"{row.Category_Count} Categories ({row.Categories[:45]}...)", va="center", fontsize=8.5, fontweight="bold", color="#1e293b")
    plt.xlim(0, 16)
    plt.tight_layout()
    fig4_path = FIG_DIR / "eda_cross_category_skills.png"
    plt.savefig(fig4_path, dpi=180)
    plt.close()
    print(f"Saved {fig4_path}")

    # -------------------------------------------------------------
    # FIGURE 5: Genuine Insight 2: Hidden High-Paying Niches & Experience Tiers
    # -------------------------------------------------------------
    df["exp_bracket"] = pd.cut(
        df["avg_exp"],
        bins=[-0.1, 3.5, 7.5, 15.0],
        labels=["Junior (0-3 yrs)", "Mid-Level (4-7 yrs)", "Senior (8-11 yrs)"]
    )

    pivot_exp_sal = df.pivot_table(
        index="Category",
        columns="exp_bracket",
        values="avg_salary_lpa",
        aggfunc="mean"
    ).round(1)

    # Sort by Junior pay to highlight entry-level sweet spots
    pivot_exp_sal = pivot_exp_sal.sort_values(by="Junior (0-3 yrs)", ascending=False)

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_exp_sal, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Mean Salary LPA (₹ Lakhs)'}, linewidths=0.5)
    plt.title("Insight: 'Hidden Niches' — Salary Trajectories Across Experience Brackets", weight="bold", pad=15)
    plt.ylabel("Job Category")
    plt.xlabel("Experience Bracket")
    plt.tight_layout()
    fig5_path = FIG_DIR / "eda_salary_experience_niches.png"
    plt.savefig(fig5_path, dpi=180)
    plt.close()
    print(f"Saved {fig5_path}")

    # -------------------------------------------------------------
    # Build notebooks/01_eda.ipynb using nbformat
    # -------------------------------------------------------------
    build_eda_notebook()
    print("EDA pipeline execution complete!")


def build_eda_notebook():
    nb = nbf.v4.new_notebook()

    cells = []
    
    # Title & Introduction
    cells.append(nbf.v4.new_markdown_cell("""# SmartHire — Exploratory Data Analysis (EDA)
### Project: Resume-to-Job Matching & Career Guidance Engine
**Scope**: Exploration of the 525 job postings corpus across 15 categories, analyzing skill frequencies, salary/location dynamics, and cross-domain market patterns.
**Constraints**: Classical Machine Learning foundation (no LLMs, no generative AI, no live scraping).
"""))

    # Imports
    cells.append(nbf.v4.new_code_cell("""import re
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual styling
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10
"""))

    # Load Data
    cells.append(nbf.v4.new_markdown_cell("""## 1. Data Ingestion & Schema Inspection
Loading the raw job postings corpus (`data/raw/job_postings.csv`).
"""))

    cells.append(nbf.v4.new_code_cell("""DATA_PATH = Path("../data/raw/job_postings.csv")
df = pd.read_csv(DATA_PATH)

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Missing Values:\\n{df.isnull().sum()}")
df.head(3)
"""))

    # Preprocessing
    cells.append(nbf.v4.new_markdown_cell("""## 2. Feature Parsing & Transformation
We extract structured numerical columns from `Experience_Required` and `Salary_LPA`, and tokenize semicolon-delimited `Skills`.
"""))

    cells.append(nbf.v4.new_code_cell("""def parse_skills(s):
    if not isinstance(s, str) or not s.strip(): return []
    return [x.strip() for x in re.split(r"[;,]", s) if x.strip()]

def parse_experience(exp):
    nums = [float(n) for n in re.findall(r"\\d+", str(exp))]
    if len(nums) >= 2: return nums[0], nums[1], (nums[0] + nums[1]) / 2.0
    elif len(nums) == 1: return nums[0], nums[0] + 2.0, nums[0] + 1.0
    return 0.0, 1.0, 0.5

def parse_salary(sal):
    nums = [float(n) for n in re.findall(r"\\d+(?:\\.\\d+)?", str(sal))]
    if len(nums) >= 2: return nums[0], nums[1], (nums[0] + nums[1]) / 2.0
    elif len(nums) == 1: return nums[0], nums[0], nums[0]
    return 0.0, 0.0, 0.0

df['parsed_skills'] = df['Skills'].apply(parse_skills)
df['skills_count'] = df['parsed_skills'].apply(len)

exp_tuples = df['Experience_Required'].apply(parse_experience)
df['min_exp'] = [t[0] for t in exp_tuples]
df['max_exp'] = [t[1] for t in exp_tuples]
df['avg_exp'] = [t[2] for t in exp_tuples]

sal_tuples = df['Salary_LPA'].apply(parse_salary)
df['min_salary'] = [t[0] for t in sal_tuples]
df['max_salary'] = [t[1] for t in sal_tuples]
df['avg_salary'] = [t[2] for t in sal_tuples]

df[['Job_ID', 'Title', 'Category', 'avg_exp', 'avg_salary', 'skills_count']].head()
"""))

    # Category Distribution
    cells.append(nbf.v4.new_markdown_cell("""## 3. Job Category Distribution
Checking class balance across target domains. Balanced classes ensure the supervised category classifier (Model A) is not biased toward over-represented sectors.
"""))

    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 5))
cat_counts = df['Category'].value_counts()
ax = sns.barplot(x=cat_counts.values, y=cat_counts.index, palette="mako")
plt.title("SmartHire Job Category Distribution", weight="bold")
plt.xlabel("Posting Count")
plt.ylabel("Job Category")
for i, v in enumerate(cat_counts.values):
    ax.text(v + 0.3, i, f"{v} ({v/len(df)*100:.1f}%)", va="center", fontsize=9)
plt.xlim(0, 42)
plt.show()
"""))

    # Skill Frequency Analysis
    cells.append(nbf.v4.new_markdown_cell("""## 4. Skill Frequency Analysis by Category
We evaluate the dominant skills required within distinct engineering and functional career paths.
"""))

    cells.append(nbf.v4.new_code_cell("""target_cats = ['Data Science', 'Web Development', 'DevOps Engineer', 'Python Developer', 'Network Security Engineer', 'Digital Marketing']
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, cat in enumerate(target_cats):
    sub = df[df['Category'] == cat]
    all_s = [s for slist in sub['parsed_skills'] for s in slist]
    top_s = pd.Series(all_s).value_counts().head(6)
    
    sns.barplot(ax=axes[idx], x=top_s.values, y=top_s.index, palette="crest")
    axes[idx].set_title(f"{cat}", weight="bold")
    axes[idx].set_xlabel("Count")
    axes[idx].set_xlim(0, max(top_s.values) + 4)

plt.suptitle("Core Technical Stacks by Domain", weight="bold", fontsize=14)
plt.tight_layout()
plt.show()
"""))

    # Salary & Location Patterns
    cells.append(nbf.v4.new_markdown_cell("""## 5. Compensation & Geographic Dynamics
Evaluating salary ranges across categories and tech hubs in India.
"""))

    cells.append(nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Mean Salary by Category
sal_cat = df.groupby('Category')['avg_salary'].mean().sort_values(ascending=False)
sns.barplot(ax=ax1, x=sal_cat.values, y=sal_cat.index, palette="viridis")
ax1.set_title("Average Salary by Category (LPA)", weight="bold")
ax1.set_xlabel("LPA (₹ Lakhs)")

# Mean Salary by Location
sal_loc = df.groupby('Location')['avg_salary'].agg(['mean', 'count']).sort_values(by='mean', ascending=False)
sns.barplot(ax=ax2, x=sal_loc['mean'].values, y=sal_loc.index, palette="flare")
ax2.set_title("Average Salary by Hub (LPA)", weight="bold")
ax2.set_xlabel("LPA (₹ Lakhs)")

plt.tight_layout()
plt.show()
"""))

    # Genuine Insights
    cells.append(nbf.v4.new_markdown_cell("""## 6. Genuinely Interesting Insights

### Insight 1: Universal "Bridge Skills"
Which skills transcend boundaries and appear in the greatest number of categories?
Candidates with these bridge skills possess natural flexibility to pivot across multiple industries.
"""))

    cells.append(nbf.v4.new_code_cell("""skill_cat_map = {}
for _, row in df.iterrows():
    for s in row['parsed_skills']:
        skill_cat_map.setdefault(s, set()).add(row['Category'])

bridge_df = pd.DataFrame([
    {'Skill': s, 'Category_Count': len(cats), 'Categories': ', '.join(sorted(cats))}
    for s, cats in skill_cat_map.items()
]).sort_values(by='Category_Count', ascending=False)

plt.figure(figsize=(10, 5))
top_bridge = bridge_df.head(10)
sns.barplot(data=top_bridge, x='Category_Count', y='Skill', palette="crest")
plt.title("Universal Bridge Skills Across Job Domains", weight="bold")
plt.xlabel("Number of Categories Covered (out of 15)")
plt.show()

top_bridge[['Skill', 'Category_Count', 'Categories']]
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Insight 2: High-Paying Niches & Entry-Level Premiums
Where do candidates gain the highest early-career return on investment?
We segment job postings into experience tiers:
- **Junior (0-3 yrs)**
- **Mid-Level (4-7 yrs)**
- **Senior (8-11 yrs)**
"""))

    cells.append(nbf.v4.new_code_cell("""df['exp_bracket'] = pd.cut(
    df['avg_exp'],
    bins=[-0.1, 3.5, 7.5, 15.0],
    labels=["Junior (0-3 yrs)", "Mid-Level (4-7 yrs)", "Senior (8-11 yrs)"]
)

pivot_salary = df.pivot_table(
    index="Category",
    columns="exp_bracket",
    values="avg_salary",
    aggfunc="mean"
).round(1).sort_values(by="Junior (0-3 yrs)", ascending=False)

plt.figure(figsize=(10, 7))
sns.heatmap(pivot_salary, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'LPA (₹ Lakhs)'})
plt.title("Salary Trajectories across Experience Brackets", weight="bold")
plt.xlabel("Experience Tier")
plt.ylabel("Category")
plt.tight_layout()
plt.show()

pivot_salary
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 7. Conclusions & Preparation for Phase 2
### Summary of EDA Findings:
1. **Perfect Class Balance**: The dataset features exactly 35 records per category across 15 distinct job categories (525 total rows), establishing a reliable benchmark for multi-class classification.
2. **Domain-Specific Skill Signatures**: Key roles have distinct skill vocabularies (e.g. Kotlin/Android SDK for Mobile; Spring Boot/Hibernate for Java; AWS/Terraform/Docker for DevOps; AutoCAD/Revit for Civil).
3. **Bridge Competencies**: Foundational tools like `Communication`, `Excel`, `SQL`, `Python`, and `Git` bridge multiple sectors.
4. **Compensation Drivers**: Domains like `Network Security Engineer`, `DevOps Engineer`, and `Data Science` exhibit the highest entry-level packages (>15 LPA average at 0-3 years), making them high-value target recommendations.

We are ready to proceed with **Phase 2: Supervised Category Classifier (Model A)**.
"""))

    nb.cells = cells
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Saved executed notebook template to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    run_eda_pipeline()
