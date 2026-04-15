# A Study on the Role of HR Practices in Retaining Software Engineers

![HR Analytics Dashboard](https://img.shields.io/badge/HR%20Analytics-Software%20Engineer%20Retention-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Version](https://img.shields.io/badge/Version-v2.1--Advanced-blueviolet?style=for-the-badge)

## 📌 Project Overview
This repository contains a comprehensive HR Analytics platform developed to study how various Human Resource practices influence the retention and attrition of software engineers. Given the high cost of replacement for technical talent, this study utilizes machine learning and advanced statistical methods to identify key drivers of turnover and provides evidence-based recommendations for engineering leadership.

## 🚀 Key Features

### 🔍 Exploratory Data Analysis (v2.1 Enhanced)
- **Distribution Analysis:** Violin plots, histograms, and ECDF for income and age shifts.
- **Categorical Flows:** Parallel categories and Sankey diagrams for department/role transitions.
- **Statistical Testing:** Automated Chi-Square (categorical) and Mann-Whitney U (numeric) tests.
- **Outlier Detection:** IQR-based identification of extreme workforce segments.

### 🧬 Advanced Analytics & Segmentation
- **K-Means Clustering:** Workforce segmentation into risk/satisfaction-based cohorts.
- **PCA Dimensionality Reduction:** 2D and 3D projections of the high-dimensional employee feature space.
- **Cohort Analysis:** Retention curve tracking for hire-year cohorts.
- **Pay Equity Analysis:** Statistical comparison of compensation across gender and department.

### 🤖 Predictive Modeling & Diagnostics
- **Ensemble Model Laboratory:** Side-by-side comparison of Random Forest, Gradient Boosting, and Logistic Regression.
- **Performance Diagnostics:** ROC Curves, Precision-Recall Curves, and 5-Fold Cross-Validation.
- **Real-time Risk Profiler:** An interactive "What-If" gauge tool for individual profile risk assessment.

### 💡 Strategy Roadmap
- **Strategy Matrix:** Cost vs. Impact visualization for HR interventions.
- **Policy Framework:** Automated generation of priority-coded retention strategies.

## 🛠️ Technology Stack
- **Framework:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualization:** [Plotly](https://plotly.com/python/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/)
- **Statistical Tests:** [SciPy](https://scipy.org/)

## 📋 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd "A Study on the Role of HR Practices in Retaining Software Engineers"
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Then run:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

## 📊 Dataset Information
The platform supports:
1. **Custom CSV Upload:** Analyze your organization's specific HR data (requires standard columns like `Attrition`, `Age`, `MonthlyIncome`, etc.).
2. **Synthetic Sample Data:** A built-in research-grade dataset of 1,500 software engineers calibrated with realistic attrition probabilities.

## 📄 License
This project is for educational and research purposes.

---

*“Retaining software engineers is not just about the salary; it’s about the environment, the growth, and the balance.”*
