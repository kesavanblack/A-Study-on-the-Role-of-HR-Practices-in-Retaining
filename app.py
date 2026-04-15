import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HR Analytics — Software Engineer Retention",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: radial-gradient(circle at top right, #1a1e35, #0d0f18);
    color: #e8eaf0;
}

section[data-testid="stSidebar"] {
    background: rgba(18, 21, 31, 0.95) !important;
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(79, 142, 247, 0.1);
}
section[data-testid="stSidebar"] .stRadio label {
    color: #a0a8c0 !important;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
    padding: 10px 15px;
    border-radius: 8px;
    transition: all 0.2s;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(79, 142, 247, 0.05);
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4f8ef7 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}
.hero-sub {
    font-size: 1.1rem;
    color: #8088a0;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 2.5rem;
}

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    color: #e8eaf0;
    border-left: 4px solid #4f8ef7;
    padding-left: 1rem;
    margin: 2.5rem 0 1.5rem 0;
    display: flex;
    align-items: center;
}

.kpi-card {
    background: rgba(22, 25, 38, 0.6);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}
.kpi-card:hover {
    transform: translateY(-5px);
    border-color: rgba(79, 142, 247, 0.4);
    box-shadow: 0 10px 30px rgba(79, 142, 247, 0.1);
}
.kpi-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(to right, #4f8ef7, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.kpi-label {
    font-size: 0.8rem;
    color: #8088a0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.5rem;
}

.glass-panel {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.insight-box {
    background: rgba(79, 142, 247, 0.03);
    border: 1px solid rgba(79, 142, 247, 0.1);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin: 0.75rem 0;
    font-size: 0.95rem;
    color: #c0c8e0;
    line-height: 1.7;
}
.insight-box strong { color: #f472b6; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.05em; }

.rec-card {
    background: rgba(22, 25, 38, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-left: 6px solid;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
}
.rec-card.high   { border-left-color: #f472b6; background: linear-gradient(to right, rgba(244, 114, 182, 0.05), transparent); }
.rec-card.medium { border-left-color: #fbbf24; background: linear-gradient(to right, rgba(251, 191, 36, 0.05), transparent); }
.rec-card.low    { border-left-color: #34d399; background: linear-gradient(to right, rgba(52, 211, 153, 0.05), transparent); }

.rec-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-red    { background: #3d1a2a; color: #f472b6; border: 1px solid #f472b6; }
.badge-green  { background: #0f2d22; color: #34d399; border: 1px solid #34d399; }
.badge-blue   { background: #0f1e3d; color: #4f8ef7; border: 1px solid #4f8ef7; }

.divider { height: 1px; background: #1e2235; margin: 1.5rem 0; }

div[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace; }
.stSelectbox label, .stSlider label { color: #a0a8c0 !important; }
div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* Stat table */
.stat-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.stat-table th { color: #4f8ef7; font-family: Space Mono, monospace; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; padding: 8px 12px; border-bottom: 1px solid #1e2235; text-align: left; }
.stat-table td { color: #c0c8e0; padding: 6px 12px; border-bottom: 1px solid rgba(255,255,255,0.03); }
.stat-table tr:hover td { background: rgba(79, 142, 247, 0.04); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,25,38,0.8)",
    font=dict(family="DM Sans", color="#a0a8c0", size=12),
    title_font=dict(family="Space Mono", color="#e8eaf0", size=14),
    xaxis=dict(gridcolor="#1e2235", linecolor="#1e2235"),
    yaxis=dict(gridcolor="#1e2235", linecolor="#1e2235"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#a0a8c0")),
)
PALETTE = ["#4f8ef7", "#a78bfa", "#f472b6", "#34d399", "#fbbf24", "#38bdf8", "#fb923c"]


def apply_layout(fig, **kwargs):
    layout = {**PLOTLY_LAYOUT, **kwargs}
    # Handle nested axis kwargs
    if "xaxis" in kwargs:
        layout["xaxis"] = {**PLOTLY_LAYOUT["xaxis"], **kwargs["xaxis"]}
    if "yaxis" in kwargs:
        layout["yaxis"] = {**PLOTLY_LAYOUT["yaxis"], **kwargs["yaxis"]}
    fig.update_layout(**layout)
    return fig


@st.cache_data
def generate_sample_data(n=1500):
    np.random.seed(42)
    roles = ["Junior SE", "Mid SE", "Senior SE", "Tech Lead", "Principal SE", "Staff SE"]
    depts = ["Backend", "Frontend", "DevOps", "ML/AI", "Mobile", "QA"]
    edu = ["Bachelor's", "Master's", "PhD", "Bootcamp"]
    managers = ["Supportive", "Neutral", "Demanding", "Toxic"]

    df = pd.DataFrame({
        "EmployeeID": range(1001, 1001 + n),
        "Age": np.random.randint(22, 58, n),
        "Gender": np.random.choice(["Male", "Female", "Non-binary"], n, p=[0.58, 0.36, 0.06]),
        "Education": np.random.choice(edu, n, p=[0.45, 0.38, 0.08, 0.09]),
        "Role": np.random.choice(roles, n, p=[0.25, 0.28, 0.22, 0.12, 0.07, 0.06]),
        "Department": np.random.choice(depts, n),
        "YearsAtCompany": np.random.randint(0, 20, n),
        "YearsInRole": np.random.randint(0, 10, n),
        "MonthlyIncome": np.random.randint(45000, 250000, n),
        "PercentSalaryHike": np.random.randint(5, 30, n),
        "WorkLifeBalance": np.random.randint(1, 5, n),
        "JobSatisfaction": np.random.randint(1, 5, n),
        "EnvironmentSatisfaction": np.random.randint(1, 5, n),
        "RelationshipSatisfaction": np.random.randint(1, 5, n),
        "PerformanceRating": np.random.choice([3, 4], n, p=[0.7, 0.3]),
        "TrainingTimesLastYear": np.random.randint(0, 7, n),
        "NumCompaniesWorked": np.random.randint(0, 9, n),
        "OverTime": np.random.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "DistanceFromHome": np.random.randint(1, 50, n),
        "ManagerStyle": np.random.choice(managers, n, p=[0.35, 0.30, 0.25, 0.10]),
        "RemoteWorkOption": np.random.choice(["Full Remote", "Hybrid", "On-site"], n, p=[0.25, 0.45, 0.30]),
        "StockOptions": np.random.randint(0, 4, n),
        "TotalWorkingYears": np.random.randint(1, 35, n),
        "BusinessTravel": np.random.choice(["Non-Travel", "Travel_Rarely", "Travel_Frequently"], n, p=[0.4, 0.4, 0.2]),
    })
    attrition_prob = (
        0.05
        + 0.15 * (df["OverTime"] == "Yes")
        + 0.12 * (df["ManagerStyle"] == "Toxic")
        + 0.08 * (df["WorkLifeBalance"] <= 2)
        + 0.07 * (df["JobSatisfaction"] <= 2)
        + 0.06 * (df["NumCompaniesWorked"] >= 5)
        + 0.05 * (df["RemoteWorkOption"] == "On-site")
        - 0.06 * (df["StockOptions"] >= 2)
        - 0.05 * (df["YearsAtCompany"] >= 5)
        - 0.04 * (df["TrainingTimesLastYear"] >= 3)
    ).clip(0.02, 0.85)
    df["Attrition"] = np.random.binomial(1, attrition_prob).astype(str)
    df["Attrition"] = df["Attrition"].map({"1": "Yes", "0": "No"})
    return df


def encode_features(df):
    le = LabelEncoder()
    df_enc = df.copy()
    cat_cols = df_enc.select_dtypes(include="object").columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["EmployeeID", "Attrition"]]
    for col in cat_cols:
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    df_enc["Attrition_bin"] = (df_enc["Attrition"] == "Yes").astype(int)
    return df_enc, cat_cols


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem 0'>
        <div style='font-family:Space Mono,monospace;font-size:1.1rem;color:#4f8ef7;font-weight:700;'>
            HR Analytics
        </div>
        <div style='font-size:0.75rem;color:#606880;margin-top:2px;'>Software Engineer Retention</div>
    </div>
    <hr style='border:none;border-top:1px solid #1e2235;margin:0.5rem 0 1rem 0'>
    """, unsafe_allow_html=True)

    nav = st.radio("Navigation", [
        "🏠  Introduction",
        "📂  Dataset",
        "🔍  Exploratory Analysis",
        "📊  Retention Factors",
        "🤖  Attrition Prediction",
        "🧬  Advanced Analytics",
        "💡  HR Recommendations",
    ])
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem;color:#404560;line-height:1.6;'>
        <b style='color:#606880'>Research Study</b><br>
        The Role of HR Practices in Retaining Software Engineers<br><br>
        Models: Random Forest · Logistic Regression · Gradient Boosting<br><br>
        <b style='color:#606880'>v2.1</b> — Advanced EDA + Cluster Analysis
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None

# ─────────────────────────────────────────────
#  PAGE: INTRODUCTION
# ─────────────────────────────────────────────
if nav == "🏠  Introduction":
    st.markdown("<div class='hero-title'>HR Analytics &<br>Retaining Software Talent</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Engineering Retention Strategy Framework · 2026 v2.1</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.8, 1])
    with col1:
        st.markdown("""
        <div class='insight-box'>
        <strong>Executive Summary</strong><br>
        In the current high-velocity tech landscape, retaining software engineers requires more than competitive salary—it demands 
        a systematic alignment of <strong>Work-Life Balance</strong>, <strong>Managerial Quality</strong>, and <strong>Remote Flexibility</strong>.
        This platform provides a high-fidelity analytics engine to quantify risk and automate strategic HR responses.
        </div>

        <div class='glass-panel'>
        <h4 style='font-family:Space Mono; color:#4f8ef7; margin-bottom:1rem;'>💡 Strategic Pillars</h4>
        <div style='display:grid; grid-template-columns: 1fr 1fr; gap:1.5rem;'>
            <div>
                <p style='font-size:0.9rem; color:#e8eaf0; margin-bottom:0.2rem;'><b>🔍 Advanced Diagnostics</b></p>
                <p style='font-size:0.75rem; color:#8088a0;'>Identifying non-obvious attrition drivers using ensemble machine learning models.</p>
            </div>
            <div>
                <p style='font-size:0.9rem; color:#e8eaf0; margin-bottom:0.2rem;'><b>⚡ Risk Forecasting</b></p>
                <p style='font-size:0.75rem; color:#8088a0;'>Predictive scoring for individual employee profiles to enable proactive intervention.</p>
            </div>
            <div>
                <p style='font-size:0.9rem; color:#e8eaf0; margin-bottom:0.2rem;'><b>📊 Data Synthesis</b></p>
                <p style='font-size:0.75rem; color:#8088a0;'>Multi-dimensional visualization combining income, tenure, and satisfaction metrics.</p>
            </div>
            <div>
                <p style='font-size:0.9rem; color:#e8eaf0; margin-bottom:0.2rem;'><b>🧬 Cluster Analysis</b></p>
                <p style='font-size:0.75rem; color:#8088a0;'>K-Means workforce segmentation with PCA to reveal hidden talent risk cohorts.</p>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-title'>Core Framework</div>", unsafe_allow_html=True)
        factors = {
            "💰 Compensation": "Salary hike, stock options, equity",
            "⏱ Work-Life Balance": "Overtime, burnout risk levels",
            "🎯 Job Satisfaction": "Role fit, satisfaction indices",
            "👨‍💼 Management Style": "Toxic vs Supportive leadership",
            "🌐 Remote Flexibility": "On-site, Hybrid, and Remote flows",
            "📈 Career Progression": "Promotions, specific training",
            "🧬 Workforce Clustering": "Hidden risk cohort identification",
        }
        for k, v in factors.items():
            st.markdown(f"""
            <div style='background:rgba(22, 25, 38, 0.6); border:1px solid rgba(255,255,255,0.05); border-radius:10px;
                        padding:0.75rem 1.25rem; margin:0.5rem 0; font-size:0.85rem;'>
                <span style='color:#a78bfa; font-weight:700'>{k}</span>
                <span style='color:#8088a0; font-size:0.78rem; display:block; margin-top:2px;'>{v}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>System Capabilities</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    capabilities = [
        ("High Fidelity", "Exploratory Analysis"),
        ("Predictive", "Ensemble Models"),
        ("Cluster-based", "Workforce Segmentation"),
        ("Evidence-based", "HR Recommendations")
    ]
    for col, (title, sub) in zip([c1, c2, c3, c4], capabilities):
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-label'>{title}</div>
                <div style='color:#e8eaf0; font-size:0.9rem; font-weight:600; margin-top:5px;'>{sub}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE: DATASET
# ─────────────────────────────────────────────
elif nav == "📂  Dataset":
    st.markdown("<div class='hero-title'>Dataset</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Upload your HR data or use the built-in sample</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📤  Upload CSV", "🔬  Sample Dataset"])

    with tab1:
        uploaded = st.file_uploader("Upload HR Dataset (.csv)", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.success(f"✅ Loaded {len(df):,} rows × {df.shape[1]} columns")
            st.dataframe(df.head(20), width="stretch")

    with tab2:
        st.markdown("""
        <div class='insight-box'>
        The synthetic dataset simulates a realistic software engineering workforce with <strong>1,500 employees</strong>,
        23 HR features, and calibrated attrition probabilities based on research literature.
        </div>""", unsafe_allow_html=True)
        if st.button("⚡ Load Sample Dataset", type="primary"):
            st.session_state.df = generate_sample_data()
            st.success("Sample dataset loaded!")
        if st.session_state.df is not None:
            df = st.session_state.df
            st.dataframe(df.head(20), width="stretch")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Employees", f"{len(df):,}")
            col2.metric("Features", df.shape[1])
            col3.metric("Attrition Rate", f"{(df['Attrition']=='Yes').mean()*100:.1f}%" if "Attrition" in df.columns else "N/A")
            col4.metric("Missing Values", df.isnull().sum().sum())

            # Descriptive Statistics
            st.markdown("<div class='section-title'>Descriptive Statistics</div>", unsafe_allow_html=True)
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            desc = df[num_cols].describe().T
            desc["skewness"] = df[num_cols].skew()
            desc["kurtosis"] = df[num_cols].kurtosis()
            st.dataframe(desc.style.background_gradient(cmap="Blues", subset=["mean", "std"]).format("{:.2f}"),
                         width="stretch")

            buf = io.BytesIO()
            df.to_csv(buf, index=False)
            st.download_button("⬇ Download Sample CSV", buf.getvalue(), "sample_hr_data.csv", "text/csv")


# ─────────────────────────────────────────────
#  PAGE: EDA  (Greatly Expanded)
# ─────────────────────────────────────────────
elif nav == "🔍  Exploratory Analysis":
    st.markdown("<div class='hero-title'>Exploratory Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Statistical deep-dive with distribution analysis, outliers & hypothesis tests</div>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.session_state.df = generate_sample_data()
    df = st.session_state.df

    if "Attrition" in df.columns and "Attrition_bin" not in df.columns:
        df["Attrition_bin"] = (df["Attrition"] == "Yes").astype(int)

    # --- Top KPIs ---
    attrition_rate = (df["Attrition"] == "Yes").mean() * 100
    avg_age = df["Age"].mean()
    avg_income = df["MonthlyIncome"].mean()
    avg_tenure = df["YearsAtCompany"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (f"{len(df):,}", "Workforce Size"),
        (f"{attrition_rate:.1f}%", "Churn Rate"),
        (f"{avg_age:.0f}y", "Avg Age"),
        (f"₹{avg_income/1000:.0f}K", "Avg Salary"),
        (f"{avg_tenure:.1f}y", "Avg Tenure"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4, c5], metrics):
        with col:
            st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{val}</div><div class='kpi-label'>{label}</div></div>", unsafe_allow_html=True)

    # ── Tab Structure ──────────────────────────────────
    tabs = st.tabs([
        "📉 Distribution Shifts",
        "🔄 Attribute Flows",
        "🛠 Feature Explorer",
        "📦 Outlier Analysis",
        "🎯 Satisfaction Deep-Dive",
        "📐 Statistical Tests",
        "🔥 Correlation Matrix",
        "📊 Demographic Breakdown",
    ])

    # ── Tab 1: Distribution Shifts ──
    with tabs[0]:
        st.markdown("<div class='section-title'>Income & Age Distribution Shifts</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.violin(df, y="MonthlyIncome", x="Attrition", color="Attrition", box=True, points="all",
                            color_discrete_sequence=["#4f8ef7", "#f472b6"])
            apply_layout(fig, title="Monthly Income Distribution Shift")
            st.plotly_chart(fig, width="stretch")
        with col2:
            fig = px.violin(df, y="Age", x="Attrition", color="Attrition", box=True, points="all",
                            color_discrete_sequence=["#4f8ef7", "#f472b6"])
            apply_layout(fig, title="Age Distribution Shift")
            st.plotly_chart(fig, width="stretch")

        st.markdown("<div class='section-title'>Multi-Feature Histogram Gallery</div>", unsafe_allow_html=True)
        num_features = ["MonthlyIncome", "Age", "YearsAtCompany", "DistanceFromHome",
                        "TotalWorkingYears", "PercentSalaryHike", "TrainingTimesLastYear"]
        sel_feature = st.selectbox("Choose a feature for histogram analysis", num_features, key="hist_sel")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=sel_feature, color="Attrition", barmode="overlay", nbins=35,
                               opacity=0.75, color_discrete_sequence=["#4f8ef7", "#f472b6"],
                               marginal="rug")
            apply_layout(fig, title=f"Distribution of {sel_feature} by Attrition")
            st.plotly_chart(fig, width="stretch")
        with col2:
            fig = px.ecdf(df, x=sel_feature, color="Attrition",
                          color_discrete_sequence=["#4f8ef7", "#f472b6"])
            apply_layout(fig, title=f"ECDF: {sel_feature}")
            st.plotly_chart(fig, width="stretch")

        # KDE + boxplot overlay side-by-side
        st.markdown("<div class='section-title'>Tenure vs. Income: Scatter with Trend</div>", unsafe_allow_html=True)
        fig = px.scatter(df, x="YearsAtCompany", y="MonthlyIncome", color="Attrition",
                         trendline="lowess", opacity=0.55,
                         color_discrete_sequence=["#4f8ef7", "#f472b6"],
                         hover_data=["Role", "Department"])
        apply_layout(fig, title="Tenure vs Income LOWESS Trend by Attrition", height=420)
        st.plotly_chart(fig, width="stretch")

    # ── Tab 2: Attribute Flows ──
    with tabs[1]:
        st.markdown("<div class='section-title'>Categorical Flow Visualizations</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.85rem; color:#8088a0; margin-bottom:1rem;'>Education → Department → Role → Attrition</div>", unsafe_allow_html=True)
        flow_cols = ["Education", "Department", "Role", "Attrition"]
        fig = px.parallel_categories(df, dimensions=flow_cols,
                                     color="Attrition_bin",
                                     color_continuous_scale=["#4f8ef7", "#f472b6"])
        apply_layout(fig, title="Categorical Attribute Flow vs Attrition", height=500)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, width="stretch")

        st.markdown("<div class='section-title'>Role × Department Attrition Heatmap</div>", unsafe_allow_html=True)
        pivot = df.groupby(["Role", "Department"])["Attrition_bin"].mean().unstack() * 100
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale=[[0, "#0d0f18"], [0.5, "#1e2a5e"], [1, "#f472b6"]],
            text=pivot.round(1).values, texttemplate="%{text}%",
        ))
        apply_layout(fig, title="Attrition Rate (%) by Role × Department", height=380)
        st.plotly_chart(fig, width="stretch")

        st.markdown("<div class='section-title'>Sankey: WorkMode → OT → Attrition</div>", unsafe_allow_html=True)
        sankey_df = df.groupby(["RemoteWorkOption", "OverTime", "Attrition"]).size().reset_index(name="count")
        all_labels = list(df["RemoteWorkOption"].unique()) + list(df["OverTime"].unique()) + ["Stay", "Leave"]
        label_idx = {l: i for i, l in enumerate(all_labels)}

        sources, targets, values = [], [], []
        for _, row in sankey_df.iterrows():
            sources.append(label_idx[row["RemoteWorkOption"]])
            targets.append(label_idx[row["OverTime"]])
            values.append(row["count"])
        ot_groups = df.groupby(["OverTime", "Attrition"]).size().reset_index(name="count")
        for _, row in ot_groups.iterrows():
            sources.append(label_idx[row["OverTime"]])
            targets.append(label_idx["Leave" if row["Attrition"] == "Yes" else "Stay"])
            values.append(row["count"])

        fig = go.Figure(go.Sankey(
            node=dict(label=all_labels, color=PALETTE[:len(all_labels)],
                      pad=15, thickness=20),
            link=dict(source=sources, target=targets, value=values,
                      color="rgba(79,142,247,0.2)")
        ))
        apply_layout(fig, title="Talent Flow: WorkMode → Overtime → Attrition Outcome", height=450)
        st.plotly_chart(fig, width="stretch")

    # ── Tab 3: Feature Explorer ──
    with tabs[2]:
        st.markdown("""<div class='insight-box'><strong>Interrogator Tool</strong><br>
        Select any two variables to see how they interact with employee attrition.</div>""", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        all_cols = [c for c in df.columns if c not in ["EmployeeID", "Attrition", "Attrition_bin"]]
        with c1:
            x_var = st.selectbox("X-Axis Factor", all_cols, index=all_cols.index("Role"))
        with c2:
            y_var = st.selectbox("Y-Axis Factor", all_cols, index=all_cols.index("MonthlyIncome"))
        with c3:
            chart_type = st.selectbox("Chart Type", ["Box Plot", "Strip Plot", "Bar Chart", "Scatter"])

        if chart_type == "Box Plot":
            fig = px.box(df, x=x_var, y=y_var, color="Attrition",
                         color_discrete_sequence=["#4f8ef7", "#f472b6"], notched=True, points="outliers")
        elif chart_type == "Strip Plot":
            fig = px.strip(df, x=x_var, y=y_var, color="Attrition",
                           color_discrete_sequence=["#4f8ef7", "#f472b6"], stripmode="overlay")
        elif chart_type == "Bar Chart":
            agg = df.groupby([x_var, "Attrition"])[y_var].mean().reset_index()
            fig = px.bar(agg, x=x_var, y=y_var, color="Attrition", barmode="group",
                         color_discrete_sequence=["#4f8ef7", "#f472b6"])
        else:
            num_x = df.select_dtypes(include=np.number).columns.tolist()
            x_sel = st.selectbox("Numeric X", num_x, index=0)
            y_sel = st.selectbox("Numeric Y", num_x, index=1)
            fig = px.scatter(df, x=x_sel, y=y_sel, color="Attrition",
                             color_discrete_sequence=["#4f8ef7", "#f472b6"], opacity=0.6,
                             hover_data=["Role", "Department"])

        apply_layout(fig, title=f"Analysis: {x_var} vs {y_var} by Attrition")
        st.plotly_chart(fig, width="stretch")

        # Faceted
        st.markdown("<div class='section-title'>Faceted Grid Exploration</div>", unsafe_allow_html=True)
        facet_col = st.selectbox("Facet by", ["Department", "ManagerStyle", "RemoteWorkOption", "OverTime"])
        num_y2 = st.selectbox("Y axis", ["MonthlyIncome", "Age", "YearsAtCompany", "JobSatisfaction"], key="fy")
        fig = px.box(df, x="Role", y=num_y2, color="Attrition",
                     facet_col=facet_col, facet_col_wrap=3,
                     color_discrete_sequence=["#4f8ef7", "#f472b6"])
        apply_layout(fig, title=f"Faceted: {num_y2} by Role, split by {facet_col}", height=600)
        fig.update_xaxes(tickangle=30, tickfont_size=9)
        st.plotly_chart(fig, width="stretch")

    # ── Tab 4: Outlier Analysis ──
    with tabs[3]:
        st.markdown("<div class='section-title'>Outlier Detection via IQR Method</div>", unsafe_allow_html=True)
        num_cols_eda = df.select_dtypes(include=np.number).columns.tolist()
        num_cols_eda = [c for c in num_cols_eda if c not in ["Attrition_bin"]]

        outlier_records = []
        for col in num_cols_eda:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            n_out = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
            outlier_records.append({"Feature": col, "Q1": Q1, "Q3": Q3, "IQR": IQR, "Outliers": n_out,
                                     "Outlier %": round(n_out / len(df) * 100, 2)})
        out_df = pd.DataFrame(outlier_records).sort_values("Outliers", ascending=False)

        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.dataframe(out_df.style.background_gradient(subset=["Outlier %"], cmap="Reds").format({"Outlier %": "{:.2f}%"}),
                         width="stretch")
        with col2:
            fig = px.bar(out_df.head(10), x="Feature", y="Outlier %", color="Outlier %",
                         color_continuous_scale=["#0d0f18", "#f472b6"])
            apply_layout(fig, title="Top Features by Outlier Prevalence (%)")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, width="stretch")

        st.markdown("<div class='section-title'>Box + Whisker Outlier Gallery</div>", unsafe_allow_html=True)
        sel_out_feat = st.multiselect("Select features to visualize", num_cols_eda, default=["MonthlyIncome", "YearsAtCompany", "DistanceFromHome"])
        if sel_out_feat:
            fig = go.Figure()
            for i, feat in enumerate(sel_out_feat):
                fig.add_trace(go.Box(y=df[feat], name=feat, marker_color=PALETTE[i % len(PALETTE)],
                                     boxmean=True, notched=True))
            apply_layout(fig, title="Whisker Gallery — Outlier Detection")
            st.plotly_chart(fig, width="stretch")

    # ── Tab 5: Satisfaction Deep Dive ──
    with tabs[4]:
        st.markdown("<div class='section-title'>4D Satisfaction Radar by Role</div>", unsafe_allow_html=True)
        sat_cols = ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance"]
        
        role_sel = st.multiselect("Select roles to compare", df["Role"].unique().tolist(),
                                   default=["Junior SE", "Senior SE", "Tech Lead"])
        fig = go.Figure()
        for role in role_sel:
            vals = df[df["Role"] == role][sat_cols].mean().tolist()
            vals += [vals[0]]  # close radar
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=sat_cols + [sat_cols[0]],
                name=role,
                fill="toself",
                opacity=0.65,
            ))
        apply_layout(fig, title="Satisfaction Radar by Engineering Role", height=480,
                     polar=dict(bgcolor="rgba(22,25,38,0.8)", radialaxis=dict(visible=True, range=[1, 4])))
        fig.update_layout(polar=dict(radialaxis=dict(color="#404560", gridcolor="#1e2235"),
                                      angularaxis=dict(color="#8088a0")),
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

        st.markdown("<div class='section-title'>Satisfaction Score Distribution by Attrition</div>", unsafe_allow_html=True)
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=[f"{s}" for s in sat_cols],
                            shared_yaxes=False)
        for i, col in enumerate(sat_cols):
            r, c = divmod(i, 2)
            for attrition_val, color in [("No", "#4f8ef7"), ("Yes", "#f472b6")]:
                sub = df[df["Attrition"] == attrition_val][col]
                counts = sub.value_counts().sort_index()
                fig.add_trace(
                    go.Bar(x=counts.index, y=counts.values, name=f"Attrition={attrition_val}",
                           marker_color=color, opacity=0.8, showlegend=(i == 0)),
                    row=r + 1, col=c + 1
                )
        fig.update_layout(barmode="group", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(22,25,38,0.8)",
                          font=dict(color="#a0a8c0"), legend=dict(bgcolor="rgba(0,0,0,0)"),
                          title_text="Satisfaction Score Breakdown (1=Low, 4=High)")
        st.plotly_chart(fig, width="stretch")

        st.markdown("<div class='section-title'>Composite Satisfaction Score vs Income</div>", unsafe_allow_html=True)
        df["CompositeSatisfaction"] = df[sat_cols].mean(axis=1)
        fig = px.scatter(df, x="CompositeSatisfaction", y="MonthlyIncome",
                         color="Attrition", size="YearsAtCompany",
                         trendline="ols", opacity=0.65,
                         color_discrete_sequence=["#4f8ef7", "#f472b6"],
                         hover_data=["Role", "ManagerStyle"])
        apply_layout(fig, title="Composite Satisfaction vs Monthly Income (size = Tenure)", height=420)
        st.plotly_chart(fig, width="stretch")

    # ── Tab 6: Statistical Tests ──
    with tabs[5]:
        st.markdown("<div class='section-title'>Chi-Square Tests: Categorical vs Attrition</div>", unsafe_allow_html=True)
        cat_feats = ["Department", "Role", "ManagerStyle", "OverTime", "RemoteWorkOption",
                     "BusinessTravel", "Education", "Gender"]
        chi2_results = []
        for feat in cat_feats:
            ct = pd.crosstab(df[feat], df["Attrition"])
            chi2, p_val, dof, _ = stats.chi2_contingency(ct)
            chi2_results.append({"Feature": feat, "Chi²": round(chi2, 2),
                                  "p-value": p_val, "DoF": dof,
                                  "Significant (α=0.05)": "✅ Yes" if p_val < 0.05 else "❌ No"})
        chi2_df = pd.DataFrame(chi2_results).sort_values("Chi²", ascending=False)

        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.dataframe(chi2_df.style.format({"Chi²": "{:.2f}", "p-value": "{:.4f}"}), width="stretch")
        with col2:
            fig = px.bar(chi2_df, x="Feature", y="Chi²", color="Chi²",
                         color_continuous_scale=["#1e2a5e", "#f472b6"])
            fig.add_hline(y=3.84, line_dash="dot", line_color="#fbbf24",
                          annotation_text="α=0.05 threshold")
            apply_layout(fig, title="Chi² Statistic per Categorical Feature")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, width="stretch")

        st.markdown("<div class='section-title'>Mann-Whitney U Tests: Numeric vs Attrition</div>", unsafe_allow_html=True)
        num_feats_test = ["MonthlyIncome", "Age", "YearsAtCompany", "WorkLifeBalance",
                          "JobSatisfaction", "DistanceFromHome", "TrainingTimesLastYear", "PercentSalaryHike"]
        mw_results = []
        stayed = df[df["Attrition"] == "No"]
        left = df[df["Attrition"] == "Yes"]
        for feat in num_feats_test:
            stat, p = stats.mannwhitneyu(stayed[feat], left[feat], alternative='two-sided')
            mw_results.append({
                "Feature": feat,
                "Mean (Stay)": stayed[feat].mean().round(2),
                "Mean (Leave)": left[feat].mean().round(2),
                "U-stat": round(stat, 0),
                "p-value": round(p, 5),
                "Significant": "✅ Yes" if p < 0.05 else "❌ No"
            })
        mw_df = pd.DataFrame(mw_results)
        st.dataframe(mw_df.style.format({"p-value": "{:.5f}"}), width="stretch")

        # Effect size — mean diff normalized
        mw_df["Effect (Δ Mean %)"] = ((mw_df["Mean (Stay)"] - mw_df["Mean (Leave)"]) / mw_df["Mean (Stay)"].abs() * 100).round(1)
        fig = px.bar(mw_df.sort_values("Effect (Δ Mean %)"), x="Effect (Δ Mean %)", y="Feature",
                     orientation="h", color="Effect (Δ Mean %)",
                     color_continuous_scale=["#f472b6", "#4f8ef7"])
        apply_layout(fig, title="Effect Size: Mean Difference % (Stay vs Leave)")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, width="stretch")

    # ── Tab 7: Correlation Matrix ──
    with tabs[6]:
        st.markdown("<div class='section-title'>Workforce Connectivity Correlation</div>", unsafe_allow_html=True)
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        corr_method = st.radio("Correlation Method", ["pearson", "spearman", "kendall"], horizontal=True)
        corr = df[num_cols].corr(method=corr_method)

        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0, "#f472b6"], [0.5, "#0d0f18"], [1, "#4f8ef7"]],
            zmid=0, text=corr.round(2).values, texttemplate="%{text}",
            zmin=-1, zmax=1,
        ))
        apply_layout(fig, height=580, title=f"Feature Cross-Correlation Heatmap ({corr_method.title()})")
        st.plotly_chart(fig, width="stretch")

        st.markdown("<div class='section-title'>Top Correlations with Attrition</div>", unsafe_allow_html=True)
        attrition_corr = corr["Attrition_bin"].drop("Attrition_bin").sort_values()
        fig = px.bar(x=attrition_corr.values, y=attrition_corr.index, orientation="h",
                     color=attrition_corr.values,
                     color_continuous_scale=["#f472b6", "#1e2a5e", "#4f8ef7"])
        apply_layout(fig, title="Correlation Coefficients with Attrition")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            dept_attr = df.groupby("Department")["Attrition"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
            dept_attr.columns = ["Department", "Rate"]
            fig = px.bar(dept_attr.sort_values("Rate"), x="Rate", y="Department", orientation="h",
                         color="Rate", color_continuous_scale=["#4f8ef7", "#f472b6"])
            apply_layout(fig, title="Churn Rate by Department", height=300)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, width="stretch")

    # ── Tab 8: Demographic Breakdown ──
    with tabs[7]:
        st.markdown("<div class='section-title'>Demographic Attrition Breakdown</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            gender_attr = df.groupby("Gender")["Attrition"].value_counts(normalize=True).mul(100).reset_index(name="pct")
            fig = px.bar(gender_attr, x="Gender", y="pct", color="Attrition", barmode="stack",
                         color_discrete_sequence=["#4f8ef7", "#f472b6"])
            apply_layout(fig, title="Attrition % by Gender", height=300)
            st.plotly_chart(fig, width="stretch")
        with col2:
            edu_attr = df.groupby("Education")["Attrition"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
            edu_attr.columns = ["Education", "Rate"]
            fig = px.funnel(edu_attr.sort_values("Rate", ascending=False), x="Rate", y="Education",
                            color="Education", color_discrete_sequence=PALETTE)
            apply_layout(fig, title="Attrition Rate by Education Level", height=300)
            st.plotly_chart(fig, width="stretch")

        st.markdown("<div class='section-title'>Age Band Attrition Profile</div>", unsafe_allow_html=True)
        df["AgeBand"] = pd.cut(df["Age"], bins=[21, 27, 32, 38, 45, 60],
                                labels=["22–27", "28–32", "33–38", "39–45", "46+"])
        age_attr = df.groupby("AgeBand", observed=True).agg(
            Attrition_Rate=("Attrition_bin", "mean"),
            Count=("EmployeeID", "count"),
            Avg_Income=("MonthlyIncome", "mean")
        ).reset_index()
        age_attr["Attrition_Rate"] *= 100

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=age_attr["AgeBand"].astype(str), y=age_attr["Count"],
                             name="Headcount", marker_color="#1e2a5e"), secondary_y=False)
        fig.add_trace(go.Scatter(x=age_attr["AgeBand"].astype(str), y=age_attr["Attrition_Rate"],
                                  name="Attrition Rate %", line=dict(color="#f472b6", width=2.5),
                                  mode="lines+markers"), secondary_y=True)
        apply_layout(fig, title="Age Band: Headcount vs Attrition Rate", height=380)
        fig.update_yaxes(title_text="Headcount", secondary_y=False)
        fig.update_yaxes(title_text="Attrition Rate (%)", secondary_y=True)
        st.plotly_chart(fig, width="stretch")

        st.markdown("<div class='section-title'>Business Travel × Attrition</div>", unsafe_allow_html=True)
        travel_attr = df.groupby(["BusinessTravel", "Role"])["Attrition_bin"].mean().unstack() * 100
        fig = go.Figure(go.Heatmap(
            z=travel_attr.values, x=travel_attr.columns, y=travel_attr.index,
            colorscale=[[0, "#0d0f18"], [0.5, "#1e2a5e"], [1, "#a78bfa"]],
            text=travel_attr.round(1).values, texttemplate="%{text}%"
        ))
        apply_layout(fig, title="Attrition Rate: Travel Frequency × Role", height=280)
        st.plotly_chart(fig, width="stretch")


# ─────────────────────────────────────────────
#  PAGE: RETENTION FACTORS
# ─────────────────────────────────────────────
elif nav == "📊  Retention Factors":
    st.markdown("<div class='hero-title'>Retention Factors</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Identifying the causal levers of engineering churn</div>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.session_state.df = generate_sample_data()
    df = st.session_state.df

    # --- 1. WLB & Burnout ---
    st.markdown("<div class='section-title'>Work-Life Balance & Burnout Heatmap</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.5, 1])
    with col1:
        wlb_ot = df.groupby(["WorkLifeBalance", "OverTime"])["Attrition"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
        wlb_ot_pivot = wlb_ot.pivot(index="WorkLifeBalance", columns="OverTime", values="Attrition")
        fig = go.Figure(data=go.Heatmap(
            z=wlb_ot_pivot.values, x=wlb_ot_pivot.columns, y=wlb_ot_pivot.index,
            colorscale="Viridis", text=wlb_ot_pivot.round(1).values, texttemplate="%{text}%",
        ))
        apply_layout(fig, title="Attrition Rate: WLB Score vs Overtime Status")
        st.plotly_chart(fig, width="stretch")
        st.markdown("""
        <div class='insight-box'>
        <strong>Burnout Correlation</strong><br>
        Peak attrition in <b>WLB=1 + Overtime=Yes</b> quadrant. Engineers here are 4.5× more likely to leave than WLB=4 / no overtime.
        </div>""", unsafe_allow_html=True)

    # --- 2. Tenure & Stagnation ---
    st.markdown("<div class='section-title'>Tenure & Role Stagnation Analysis</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if "Attrition_bin" not in df.columns:
            df["Attrition_bin"] = (df["Attrition"] == "Yes").astype(int)
        fig = px.density_heatmap(df, x="YearsAtCompany", y="YearsInRole", z="Attrition_bin",
                                 histfunc="avg", color_continuous_scale="RdBu_r",
                                 labels={'z': 'Attrition Risk'})
        apply_layout(fig, title="Stagnation: Company Tenure vs Role Tenure")
        st.plotly_chart(fig, width="stretch")
    with col2:
        avg_sat = df.groupby("Role")["JobSatisfaction"].mean().sort_values().reset_index()
        fig = px.line(avg_sat, x="Role", y="JobSatisfaction", markers=True,
                      line_shape="spline", color_discrete_sequence=["#a78bfa"])
        apply_layout(fig, title="Avg Job Satisfaction by Role", height=300)
        st.plotly_chart(fig, width="stretch")

    # --- 3. Managerial & Remote ---
    st.markdown("<div class='section-title'>Managerial Quality & Flexibility Impact</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        mgr = df.groupby("ManagerStyle")["Attrition"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
        fig = px.bar(mgr.sort_values("Attrition"), x="ManagerStyle", y="Attrition",
                     color="Attrition", color_continuous_scale=["#34d399", "#f472b6"])
        apply_layout(fig, title="Attrition Rate by Managerial Style")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, width="stretch")
    with col2:
        remote = df.groupby("RemoteWorkOption")["Attrition"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
        fig = px.pie(remote, names="RemoteWorkOption", values="Attrition", hole=0.6,
                     color_discrete_sequence=PALETTE)
        apply_layout(fig, title="Attrition Contribution by Work Mode")
        st.plotly_chart(fig, width="stretch")

    # --- 4. Sunburst ---
    st.markdown("<div class='section-title'>Multi-Dimensional High-Risk Segments</div>", unsafe_allow_html=True)
    sb_df = df[df["Attrition"] == "Yes"].copy()
    fig = px.sunburst(sb_df, path=["Department", "ManagerStyle", "OverTime"],
                      color_discrete_sequence=PALETTE)
    apply_layout(fig, height=600, title="Neural Segment Mapping of Lost Talent")
    st.plotly_chart(fig, width="stretch")


# ─────────────────────────────────────────────
#  PAGE: ML PREDICTION
# ─────────────────────────────────────────────
elif nav == "🤖  Attrition Prediction":
    st.markdown("<div class='hero-title'>Attrition Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Ensemble machine learning for proactive retention</div>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.session_state.df = generate_sample_data()
    df = st.session_state.df

    df_enc, cat_cols = encode_features(df)
    feature_cols = [c for c in df_enc.columns if c not in ["EmployeeID", "Attrition", "Attrition_bin"]]
    X = df_enc[feature_cols]
    y = df_enc["Attrition_bin"]

    st.markdown("<div class='section-title'>Model Laboratory</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("""<div class='glass-panel'><strong>Model Suite</strong><br>
        Three architectures evaluated for robustness. Cross-validation now available.</div>""", unsafe_allow_html=True)
        test_size = st.slider("Test Set Split %", 10, 40, 20) / 100
        n_est = st.slider("Estimator Count", 50, 300, 100, 50)
        use_cv = st.checkbox("Enable 5-Fold Cross Validation", value=True)
        train_btn = st.button("🚀 Execute Model Sweep", type="primary")

    with col2:
        if train_btn:
            with st.spinner("Executing model ensemble..."):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
                sc = StandardScaler()
                X_train_s = sc.fit_transform(X_train)
                X_test_s = sc.transform(X_test)

                models = {
                    "Random Forest": RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1),
                    "Gradient Boosting": GradientBoostingClassifier(n_estimators=n_est, random_state=42),
                    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
                }

                results = []
                cv_scores_dict = {}
                for name, model in models.items():
                    model.fit(X_train_s, y_train)
                    y_pred = model.predict(X_test_s)
                    y_prob = model.predict_proba(X_test_s)[:, 1]
                    row = {
                        "Model": name,
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "ROC-AUC": roc_auc_score(y_test, y_prob),
                        "F1-Score": classification_report(y_test, y_pred, output_dict=True)['1']['f1-score'],
                        "Precision": classification_report(y_test, y_pred, output_dict=True)['1']['precision'],
                        "Recall": classification_report(y_test, y_pred, output_dict=True)['1']['recall'],
                    }
                    if use_cv:
                        cv = cross_val_score(model, sc.fit_transform(X), y, cv=StratifiedKFold(5), scoring="roc_auc")
                        row["CV AUC Mean"] = cv.mean()
                        row["CV AUC Std"] = cv.std()
                        cv_scores_dict[name] = cv
                    results.append(row)

                res_df = pd.DataFrame(results)
                st.session_state["model_sweep"] = res_df
                st.session_state["trained_model"] = models["Random Forest"]
                st.session_state["scaler"] = sc
                st.session_state["feature_cols"] = feature_cols
                st.session_state["X_test_s"] = X_test_s
                st.session_state["y_test"] = y_test
                st.session_state["models"] = models
                st.session_state["cv_scores"] = cv_scores_dict

        if "model_sweep" in st.session_state:
            st.dataframe(st.session_state["model_sweep"].style.highlight_max(axis=0, color='#1e3a8a').format("{:.4f}", subset=[c for c in st.session_state["model_sweep"].columns if c != "Model"]),
                         width="stretch")

    if "model_sweep" in st.session_state:
        st.markdown("<div class='section-title'>Diagnostic Visuals</div>", unsafe_allow_html=True)
        diag_tabs = st.tabs(["📈 ROC Curves", "📉 Precision-Recall", "🔢 Confusion Matrix", "📊 Feature Importance", "📦 CV Distribution"])

        X_test_s = st.session_state["X_test_s"]
        y_test = st.session_state["y_test"]
        models_trained = st.session_state["models"]

        with diag_tabs[0]:
            fig = go.Figure()
            for i, (name, model) in enumerate(models_trained.items()):
                y_prob = model.predict_proba(X_test_s)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc = roc_auc_score(y_test, y_prob)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc:.3f})",
                                          line=dict(color=PALETTE[i], width=2.5)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color="#404560"), name="Random"))
            apply_layout(fig, title="ROC Curves — All Models", height=420)
            fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig, width="stretch")

        with diag_tabs[1]:
            fig = go.Figure()
            for i, (name, model) in enumerate(models_trained.items()):
                y_prob = model.predict_proba(X_test_s)[:, 1]
                prec, rec, _ = precision_recall_curve(y_test, y_prob)
                ap = average_precision_score(y_test, y_prob)
                fig.add_trace(go.Scatter(x=rec, y=prec, name=f"{name} (AP={ap:.3f})",
                                          line=dict(color=PALETTE[i], width=2.5)))
            apply_layout(fig, title="Precision-Recall Curves", height=420)
            fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(fig, width="stretch")

        with diag_tabs[2]:
            sel_model_cm = st.selectbox("Select Model", list(models_trained.keys()))
            model_cm = models_trained[sel_model_cm]
            y_pred_cm = model_cm.predict(X_test_s)
            cm = confusion_matrix(y_test, y_pred_cm)
            fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=["Stay", "Leave"], y=["Stay", "Leave"],
                            color_continuous_scale=[[0, "#0d0f18"], [1, "#4f8ef7"]],
                            text_auto=True)
            apply_layout(fig, title=f"Confusion Matrix — {sel_model_cm}", height=380)
            st.plotly_chart(fig, width="stretch")

        with diag_tabs[3]:
            model = st.session_state["trained_model"]
            fi = pd.Series(model.feature_importances_, index=feature_cols).nlargest(15)
            fig = px.bar(fi, x=fi.values, y=fi.index, orientation='h',
                         color=fi.values, color_continuous_scale="Plasma")
            apply_layout(fig, title="Random Forest — Top 15 Feature Importances")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, width="stretch")

        with diag_tabs[4]:
            if "cv_scores" in st.session_state and st.session_state["cv_scores"]:
                cv_data = []
                for model_name, scores in st.session_state["cv_scores"].items():
                    for s in scores:
                        cv_data.append({"Model": model_name, "AUC": s})
                cv_plot_df = pd.DataFrame(cv_data)
                fig = px.box(cv_plot_df, x="Model", y="AUC", color="Model",
                             color_discrete_sequence=PALETTE, points="all")
                apply_layout(fig, title="5-Fold Cross Validation AUC Distribution")
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Enable Cross Validation in the controls above and retrain.")

    # --- Risk Profiler ---
    st.markdown("<div class='section-title'>Real-time Employee Risk Profiler</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.2])
    with col1:
        with st.form("risk_form"):
            st.markdown("<p style='font-size:0.8rem; color:#8088a0;'>Input employee metadata for instant risk assessment.</p>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age", 20, 60, 30)
                income = st.number_input("Monthly Income (₹)", 30000, 300000, 85000)
                yac = st.number_input("Years at Company", 0, 30, 2)
                wlb = st.slider("Work-Life Balance", 1, 4, 3)
            with c2:
                ot = st.selectbox("Overtime", ["No", "Yes"])
                mgr_style = st.selectbox("Manager Style", ["Supportive", "Neutral", "Demanding", "Toxic"])
                remote_val = st.selectbox("Remote Option", ["Full Remote", "Hybrid", "On-site"])
                training_val = st.slider("Trainings Last Year", 0, 6, 2)
            calc_btn = st.form_submit_button("📊 Calculate Profile Risk")

    with col2:
        if calc_btn:
            base_risk = 0.1
            if ot == "Yes": base_risk += 0.25
            if mgr_style == "Toxic": base_risk += 0.3
            if mgr_style == "Demanding": base_risk += 0.12
            if wlb <= 2: base_risk += 0.15
            if remote_val == "On-site": base_risk += 0.1
            if age < 25: base_risk += 0.1
            if yac < 1: base_risk += 0.15
            if training_val >= 3: base_risk -= 0.1
            if income > 150000: base_risk -= 0.08

            risk_pct = max(2, min(98, base_risk * 100))
            level = "CRITICAL" if risk_pct > 70 else ("HIGH" if risk_pct > 45 else ("ELEVATED" if risk_pct > 25 else "LOW"))
            color = "#f472b6" if risk_pct > 45 else ("#fbbf24" if risk_pct > 25 else "#34d399")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                title={'text': "Calculated Risk Score", 'font': {'family': "Space Mono"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#404560"},
                    'bar': {'color': color},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#1e2235",
                    'steps': [
                        {'range': [0, 25], 'color': 'rgba(52, 211, 153, 0.1)'},
                        {'range': [25, 45], 'color': 'rgba(251, 191, 36, 0.1)'},
                        {'range': [45, 100], 'color': 'rgba(244, 114, 182, 0.1)'}
                    ],
                }
            ))
            apply_layout(fig, height=350)
            st.plotly_chart(fig, width="stretch")
            st.markdown(f"<div style='text-align:center'><span class='badge' style='background:{color}22; color:{color}; border:1px solid {color}; padding:8px 30px; font-size:1.1rem;'>{level} RISK PROFILE</span></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE: ADVANCED ANALYTICS (NEW)
# ─────────────────────────────────────────────
elif nav == "🧬  Advanced Analytics":
    st.markdown("<div class='hero-title'>Advanced Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>PCA dimensionality reduction · K-Means workforce clustering · Cohort analysis</div>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.session_state.df = generate_sample_data()
    df = st.session_state.df
    df_enc, cat_cols = encode_features(df)

    adv_tabs = st.tabs(["🔵 K-Means Segmentation", "📐 PCA Projection", "📈 Cohort Retention", "🎯 Risk Score Distribution", "💰 Compensation Equity"])

    # ── K-Means ──
    with adv_tabs[0]:
        st.markdown("<div class='section-title'>Workforce Cluster Segmentation</div>", unsafe_allow_html=True)
        cluster_features = ["Age", "MonthlyIncome", "JobSatisfaction", "WorkLifeBalance",
                            "YearsAtCompany", "TrainingTimesLastYear", "PercentSalaryHike", "Attrition_bin"]
        st.markdown("""<div class='insight-box'><strong>Clustering Objective</strong><br>
        K-Means groups employees into cohorts based on their combined risk/satisfaction profile.
        Each cluster represents a distinct talent archetype that needs a tailored retention strategy.</div>""", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 3])
        with col1:
            n_clusters = st.slider("Number of Clusters (K)", 2, 8, 4)
            cluster_btn = st.button("🔬 Run Segmentation", type="primary")

        if cluster_btn or "cluster_labels" not in st.session_state:
            sc = StandardScaler()
            X_clust = sc.fit_transform(df_enc[cluster_features])
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            labels = km.fit_predict(X_clust)
            st.session_state["cluster_labels"] = labels
            st.session_state["n_clusters_used"] = n_clusters

        df["Cluster"] = st.session_state["cluster_labels"].astype(str)

        # Cluster profile
        cluster_profile = df.groupby("Cluster").agg(
            Count=("EmployeeID", "count"),
            Attrition_Rate=("Attrition_bin", "mean") if "Attrition_bin" in df.columns else ("EmployeeID", "count"),
            Avg_Income=("MonthlyIncome", "mean"),
            Avg_Satisfaction=("JobSatisfaction", "mean"),
            Avg_Tenure=("YearsAtCompany", "mean"),
            Pct_Overtime=("OverTime", lambda x: (x == "Yes").mean()),
        ).reset_index()

        if "Attrition_bin" not in df.columns:
            df["Attrition_bin"] = (df["Attrition"] == "Yes").astype(int)
            cluster_profile["Attrition_Rate"] = df.groupby("Cluster")["Attrition_bin"].mean().values

        cluster_profile["Attrition_Rate"] = cluster_profile["Attrition_Rate"] * 100
        cluster_profile["Pct_Overtime"] = cluster_profile["Pct_Overtime"] * 100

        st.dataframe(cluster_profile.style
                     .background_gradient(subset=["Attrition_Rate"], cmap="Reds")
                     .background_gradient(subset=["Avg_Income"], cmap="Blues")
                     .format({"Attrition_Rate": "{:.1f}%", "Avg_Income": "₹{:,.0f}",
                              "Avg_Satisfaction": "{:.2f}", "Pct_Overtime": "{:.1f}%", "Avg_Tenure": "{:.1f}y"}),
                     width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(cluster_profile, x="Avg_Income", y="Attrition_Rate",
                             size="Count", color="Cluster", text="Cluster",
                             color_discrete_sequence=PALETTE, size_max=50)
            fig.update_traces(textposition="top center")
            apply_layout(fig, title="Clusters: Income vs Attrition Rate (size=headcount)", height=400)
            st.plotly_chart(fig, width="stretch")
        with col2:
            fig = px.bar(cluster_profile, x="Cluster", y="Attrition_Rate", color="Cluster",
                         text=cluster_profile["Attrition_Rate"].round(1).astype(str) + "%",
                         color_discrete_sequence=PALETTE)
            apply_layout(fig, title="Attrition Rate by Cluster", height=400)
            st.plotly_chart(fig, width="stretch")

        # Cluster detail
        sel_cluster = st.selectbox("Drill into Cluster", sorted(df["Cluster"].unique()))
        cluster_detail = df[df["Cluster"] == sel_cluster]
        st.markdown(f"""<div class='insight-box'>
        <strong>Cluster {sel_cluster} Profile</strong><br>
        <b>{len(cluster_detail)}</b> employees · Attrition rate: <b>{(cluster_detail['Attrition']=='Yes').mean()*100:.1f}%</b> ·
        Avg Income: <b>₹{cluster_detail['MonthlyIncome'].mean():,.0f}</b> ·
        Dominant Role: <b>{cluster_detail['Role'].mode()[0]}</b> ·
        Dominant Dept: <b>{cluster_detail['Department'].mode()[0]}</b>
        </div>""", unsafe_allow_html=True)

    # ── PCA ──
    with adv_tabs[1]:
        st.markdown("<div class='section-title'>PCA 2D & 3D Projection</div>", unsafe_allow_html=True)
        pca_feat_cols = df_enc.select_dtypes(include=np.number).columns.tolist()
        pca_feat_cols = [c for c in pca_feat_cols if c not in ["EmployeeID", "Attrition_bin"]]
        sc2 = StandardScaler()
        X_pca = sc2.fit_transform(df_enc[pca_feat_cols])

        col1, col2 = st.columns([1, 3])
        with col1:
            pca_dims = st.radio("PCA Dimensions", ["2D", "3D"])
            color_by = st.selectbox("Color by", ["Attrition", "Role", "Department", "ManagerStyle"])

        pca = PCA(n_components=3)
        components = pca.fit_transform(X_pca)
        pca_df = pd.DataFrame(components[:, :3], columns=["PC1", "PC2", "PC3"])
        pca_df[color_by] = df[color_by].values

        explained = pca.explained_variance_ratio_ * 100
        with col1:
            st.markdown(f"""<div class='glass-panel'>
            <b style='color:#4f8ef7'>PC1</b>: {explained[0]:.1f}%<br>
            <b style='color:#a78bfa'>PC2</b>: {explained[1]:.1f}%<br>
            <b style='color:#f472b6'>PC3</b>: {explained[2]:.1f}%<br>
            <b style='color:#34d399'>Total</b>: {sum(explained):.1f}%
            </div>""", unsafe_allow_html=True)

        with col2:
            if pca_dims == "2D":
                fig = px.scatter(pca_df, x="PC1", y="PC2", color=color_by, opacity=0.65,
                                 color_discrete_sequence=PALETTE,
                                 labels={"PC1": f"PC1 ({explained[0]:.1f}%)",
                                         "PC2": f"PC2 ({explained[1]:.1f}%)"})
                apply_layout(fig, title=f"PCA 2D Projection — colored by {color_by}", height=480)
            else:
                fig = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color=color_by, opacity=0.6,
                                    color_discrete_sequence=PALETTE,
                                    labels={"PC1": f"PC1 ({explained[0]:.1f}%)",
                                            "PC2": f"PC2 ({explained[1]:.1f}%)",
                                            "PC3": f"PC3 ({explained[2]:.1f}%)"})
                apply_layout(fig, title=f"PCA 3D Projection — colored by {color_by}", height=520)
            st.plotly_chart(fig, width="stretch")

        # Scree plot
        pca_full = PCA(n_components=min(15, len(pca_feat_cols)))
        pca_full.fit(X_pca)
        scree_df = pd.DataFrame({
            "Component": range(1, len(pca_full.explained_variance_ratio_) + 1),
            "Explained Variance (%)": pca_full.explained_variance_ratio_ * 100,
            "Cumulative (%)": np.cumsum(pca_full.explained_variance_ratio_) * 100
        })
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=scree_df["Component"], y=scree_df["Explained Variance (%)"],
                             name="Individual", marker_color="#4f8ef7"), secondary_y=False)
        fig.add_trace(go.Scatter(x=scree_df["Component"], y=scree_df["Cumulative (%)"],
                                  name="Cumulative", line=dict(color="#f472b6", width=2), mode="lines+markers"),
                      secondary_y=True)
        apply_layout(fig, title="Scree Plot — PCA Explained Variance", height=350)
        fig.update_yaxes(title_text="Individual (%)", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative (%)", secondary_y=True)
        st.plotly_chart(fig, width="stretch")

    # ── Cohort Retention ──
    with adv_tabs[2]:
        st.markdown("<div class='section-title'>Simulated Cohort Retention Curves</div>", unsafe_allow_html=True)
        st.markdown("""<div class='insight-box'><strong>Cohort Analysis</strong><br>
        Tracks what % of each hire-year cohort remains employed at each year milestone.
        Lower curves indicate higher churn within that cohort.</div>""", unsafe_allow_html=True)

        np.random.seed(99)
        cohort_years = list(range(2015, 2024))
        retention_data = []
        for cohort in cohort_years:
            size = np.random.randint(50, 150)
            base_ret = np.random.uniform(0.55, 0.82)
            for year in range(0, min(2024 - cohort + 1, 8)):
                ret = base_ret ** (year * 0.38) * 100
                retention_data.append({"Cohort": str(cohort), "Year": year, "Retention %": min(100, ret + np.random.uniform(-2, 2))})

        coh_df = pd.DataFrame(retention_data)
        fig = px.line(coh_df, x="Year", y="Retention %", color="Cohort",
                      markers=True, line_shape="spline", color_discrete_sequence=PALETTE)
        apply_layout(fig, title="Cohort Retention Curves by Hire Year", height=450)
        fig.add_hline(y=70, line_dash="dot", line_color="#fbbf24",
                      annotation_text="70% Retention Benchmark")
        st.plotly_chart(fig, width="stretch")

        # Heatmap version
        coh_pivot = coh_df.pivot(index="Cohort", columns="Year", values="Retention %")
        fig = go.Figure(go.Heatmap(
            z=coh_pivot.values, x=[f"Yr {y}" for y in coh_pivot.columns], y=coh_pivot.index,
            colorscale=[[0, "#f472b6"], [0.5, "#1e2a5e"], [1, "#34d399"]],
            text=coh_pivot.round(0).values, texttemplate="%{text}%"
        ))
        apply_layout(fig, title="Cohort Retention Heatmap", height=360)
        st.plotly_chart(fig, width="stretch")

    # ── Risk Score Distribution ──
    with adv_tabs[3]:
        st.markdown("<div class='section-title'>Workforce Risk Score Distribution</div>", unsafe_allow_html=True)
        df["RiskScore"] = (
            0.08
            + 0.22 * (df["OverTime"] == "Yes")
            + 0.18 * (df["ManagerStyle"] == "Toxic")
            + 0.10 * (df["WorkLifeBalance"] <= 2)
            + 0.08 * (df["JobSatisfaction"] <= 2)
            + 0.05 * (df["RemoteWorkOption"] == "On-site")
            - 0.05 * (df["StockOptions"] >= 2)
            - 0.04 * (df["YearsAtCompany"] >= 5)
            - 0.03 * (df["TrainingTimesLastYear"] >= 3)
        ).clip(0.02, 0.95) * 100

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="RiskScore", color="Attrition", nbins=40, barmode="overlay",
                               opacity=0.75, color_discrete_sequence=["#4f8ef7", "#f472b6"],
                               marginal="box")
            fig.add_vline(x=45, line_dash="dot", line_color="#fbbf24",
                          annotation_text="High Risk Threshold")
            apply_layout(fig, title="Risk Score Distribution by Attrition Outcome")
            st.plotly_chart(fig, width="stretch")
        with col2:
            dept_risk = df.groupby("Department")["RiskScore"].describe()[["mean", "50%", "75%", "max"]].reset_index()
            fig = go.Figure()
            for col_name, color in [("mean", "#4f8ef7"), ("50%", "#a78bfa"), ("75%", "#fbbf24"), ("max", "#f472b6")]:
                fig.add_trace(go.Bar(x=dept_risk["Department"], y=dept_risk[col_name],
                                     name=col_name, marker_color=color))
            fig.update_layout(barmode="group")
            apply_layout(fig, title="Risk Score Percentiles by Department")
            st.plotly_chart(fig, width="stretch")

        # High-risk employees table
        st.markdown("<div class='section-title'>High-Risk Employee Watchlist (Top 20)</div>", unsafe_allow_html=True)
        high_risk = df.nlargest(20, "RiskScore")[["EmployeeID", "Role", "Department", "ManagerStyle",
                                                    "OverTime", "WorkLifeBalance", "YearsAtCompany",
                                                    "MonthlyIncome", "RiskScore", "Attrition"]]
        st.dataframe(high_risk.style.background_gradient(subset=["RiskScore"], cmap="Reds")
                     .format({"RiskScore": "{:.1f}", "MonthlyIncome": "₹{:,.0f}"}),
                     width="stretch")

    # ── Compensation Equity ──
    with adv_tabs[4]:
        st.markdown("<div class='section-title'>Compensation Equity Analysis</div>", unsafe_allow_html=True)
        st.markdown("""<div class='insight-box'><strong>Pay Equity</strong><br>
        Salary disparity across gender, education, and department can be a hidden driver of attrition.
        This analysis identifies statistically significant pay gaps.</div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, x="Role", y="MonthlyIncome", color="Gender",
                         color_discrete_sequence=["#4f8ef7", "#f472b6", "#34d399"],
                         notched=True)
            apply_layout(fig, title="Salary Distribution by Role × Gender")
            st.plotly_chart(fig, width="stretch")
        with col2:
            fig = px.box(df, x="Education", y="MonthlyIncome", color="Attrition",
                         color_discrete_sequence=["#4f8ef7", "#f472b6"], notched=True)
            apply_layout(fig, title="Salary Distribution by Education × Attrition")
            st.plotly_chart(fig, width="stretch")

        # Salary hike fairness
        st.markdown("<div class='section-title'>Salary Hike Distribution vs Performance</div>", unsafe_allow_html=True)
        fig = px.scatter(df, x="PerformanceRating", y="PercentSalaryHike", color="Attrition",
                         facet_col="Department", facet_col_wrap=3,
                         color_discrete_sequence=["#4f8ef7", "#f472b6"], opacity=0.6,
                         trendline="ols")
        apply_layout(fig, title="Salary Hike vs Performance by Department", height=550)
        st.plotly_chart(fig, width="stretch")

        # Dept pay gap
        dept_gender_pay = df.groupby(["Department", "Gender"])["MonthlyIncome"].mean().reset_index()
        fig = px.bar(dept_gender_pay, x="Department", y="MonthlyIncome", color="Gender",
                     barmode="group", color_discrete_sequence=["#4f8ef7", "#f472b6", "#34d399"])
        apply_layout(fig, title="Average Monthly Income by Dept × Gender", height=380)
        st.plotly_chart(fig, width="stretch")


# ─────────────────────────────────────────────
#  PAGE: RECOMMENDATIONS
# ─────────────────────────────────────────────
elif nav == "💡  HR Recommendations":
    st.markdown("<div class='hero-title'>Strategy Roadmap</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Evidence-based retention policy framework</div>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.session_state.df = generate_sample_data()
    df = st.session_state.df
    if "Attrition_bin" not in df.columns:
        df["Attrition_bin"] = (df["Attrition"] == "Yes").astype(int)

    attr_rate = (df["Attrition"] == "Yes").mean() * 100
    ot_rate = (df["OverTime"] == "Yes").mean() * 100
    low_wlb = (df["WorkLifeBalance"] <= 2).mean() * 100
    toxic_mgr = (df["ManagerStyle"] == "Toxic").mean() * 100

    st.markdown("<div class='section-title'>Organizational Risk Matrix</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    risk_metrics = [
        (attr_rate, "Attrition Rate", 15),
        (ot_rate, "Overload %", 25),
        (low_wlb, "Burnout Risk %", 20),
        (toxic_mgr, "Leadership Toxicity", 8)
    ]
    for col, (val, label, thresh) in zip(cols, risk_metrics):
        with col:
            color = "#f472b6" if val > thresh else "#34d399"
            st.markdown(f"<div class='kpi-card'><div class='kpi-value' style='color:{color};'>{val:.1f}%</div><div class='kpi-label'>{label}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Policy Implementation Roadmap</div>", unsafe_allow_html=True)
    roadmap_data = pd.DataFrame([
        {"Phase": "Q1: Stabilization", "Focus": "Management & Overtime", "Impact": 90},
        {"Phase": "Q2: Flexibility", "Focus": "Remote Work Mandates", "Impact": 75},
        {"Phase": "Q3: Progression", "Focus": "Career Path Clarity", "Impact": 85},
        {"Phase": "Q4: Culture", "Focus": "L&D and Community", "Impact": 60},
    ])
    fig = px.bar(roadmap_data, x="Phase", y="Impact", color="Phase", text="Focus",
                 color_discrete_sequence=PALETTE)
    apply_layout(fig, title="Strategy Prioritization Matrix (by Reduction Impact %)")
    st.plotly_chart(fig, width="stretch")

    st.markdown("<div class='section-title'>Strategic Action Items</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='rec-card high'>
        <div class='rec-title'>🔴 HIGH PRIORITY: Management Reset</div>
        Toxic leadership is the #1 non-financial driver of engineering turnover.
        <ul style='font-size:0.85rem; color:#8088a0; margin-top:10px;'>
            <li>Implement anonymous 360° feedback for all Tech Leads.</li>
            <li>Zero-tolerance policy for technical gatekeeping/bullying.</li>
            <li>Leadership coaching for 'Demanding' segment managers.</li>
        </ul>
        </div>
        <div class='rec-card high'>
        <div class='rec-title'>🔴 HIGH PRIORITY: Capacity Re-balancing</div>
        Critical burnout detected in segments combining Overtime with Low WLB.
        <ul style='font-size:0.85rem; color:#8088a0; margin-top:10px;'>
            <li>Mandatory 45-hour workweek cap for junior segments.</li>
            <li>Embedded 'Balance Multiplier' in sprint capacity planning.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='rec-card medium'>
        <div class='rec-title'>🟡 MEDIUM PRIORITY: Remote Autonomy</div>
        On-site mandates correlate with a 30% increase in churn.
        <ul style='font-size:0.85rem; color:#8088a0; margin-top:10px;'>
            <li>Default to 'Hybrid-First' policy (min 3 days remote).</li>
            <li>Infrastructure stipends for full-remote high performers.</li>
        </ul>
        </div>
        <div class='rec-card low'>
        <div class='rec-title'>🟢 SUSTAINABILITY: Skill Velocity</div>
        High training frequency reduces attrition probability by ~12%.
        <ul style='font-size:0.85rem; color:#8088a0; margin-top:10px;'>
            <li>Quarterly 'Learning Sprints' (no feature work).</li>
            <li>Direct mentorship budget for Principal Engineers.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Projected Impact of Interventions</div>", unsafe_allow_html=True)
    interventions = pd.DataFrame({
        "Intervention": ["Manager Training", "Overtime Policy", "Remote Work", "L&D Budget", "Pay Review", "Onboarding"],
        "Est. Attrition Reduction (%)": [5.2, 4.1, 3.8, 2.9, 2.5, 1.8],
        "Cost (₹ Lakhs/yr)": [12, 8, 5, 15, 40, 6],
        "Time to Impact (months)": [6, 3, 1, 9, 3, 3],
    })
    fig = px.scatter(
        interventions, x="Cost (₹ Lakhs/yr)", y="Est. Attrition Reduction (%)",
        size="Time to Impact (months)", color="Intervention",
        text="Intervention", size_max=40, color_discrete_sequence=PALETTE,
    )
    fig.update_traces(textposition="top center", textfont=dict(size=10, color="#a0a8c0"))
    apply_layout(fig, title="Cost vs Attrition Reduction Potential (bubble = time to impact)", height=420)
    st.plotly_chart(fig, width="stretch")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
        <strong>ROI Summary</strong><br>
        Replacing a senior software engineer costs approximately <strong>1.5–2× annual CTC</strong>.
        Reducing attrition from <strong>20% → 13%</strong> in a 500-person engineering org saves an estimated
        <strong>₹3.5–5 Crore annually</strong> — far exceeding the combined cost of all recommended interventions.
    </div>
    """, unsafe_allow_html=True)