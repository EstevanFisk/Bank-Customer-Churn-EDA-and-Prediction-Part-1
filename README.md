# 🏦 Bank Customer Churn Prediction: Multi-Algorithm Statistical Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Machine_Learning-Scikit--Learn-orange?style=flat&logo=scikit-learn)](https://scikit-learn.org/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue?style=flat&logo=kaggle)](https://www.kaggle.com/)
[![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Statsmodels](https://img.shields.io/badge/Statistics-Statsmodels-green?style=flat)](https://www.statsmodels.org/)

This project focuses on predicting customer churn within a banking environment using personal, behavioral, and financial attributes. By identifying the key drivers of attrition, banks can shift from reactive management to proactive retention strategies. 

The analysis involves a rigorous statistical pipeline, moving from **Exploratory Data Analysis (EDA)** and **Cochran-Mantel-Haenszel (CMH)** testing to advanced **SMOTENC** resampling and **XGBoost/Logistic Regression** modeling.

---

## 📋 Project Outline
1.  **High-Level Data Exploration:** Initial auditing and structural overview.
2.  **Exploratory Data Analysis (EDA):**
    * **Univariate:** Distribution, skewness, and normality testing ($R^2$ Q-Q Analysis).
    * **Bivariate:** Correlation matrices (Spearman) and categorical dependency tests.
    * **Multivariate:** Interaction effects and stratified regression analysis.
3.  **Data Preprocessing & Cleaning:** Advanced pipelines using `imblearn` and `FeatureEngine`.
4.  **Model Implementation:** Multi-model benchmarking and permutation importance.

---

## 📊 Dataset Overview
The dataset features 10,000 records with 18 customer attributes. Key features include:

| Category | Features |
| :--- | :--- |
| **Demographics** | Geography, Gender, Age |
| **Financials** | CreditScore, Balance, EstimatedSalary |
| **Engagement** | Tenure, NumOfProducts, HasCrCard, IsActiveMember |
| **Feedback** | Complain, Satisfaction Score, Point Earned |
| **Target** | **Exited** (0 = Stayed, 1 = Left) |

---

## 🔍 Key Insights from EDA

### The "Complaint" Dominance
One of the most extreme findings in this study was the near-perfect correlation between **Complaints** and **Churn**.
* **Result:** Almost every customer who exited filed a complaint first.
* **Statistical Challenge:** This created **Quasi-complete Separation**, requiring L1/L2 regularization to stabilize model coefficients.

### The Balance Paradox
Counter-intuitively, customers holding a positive balance churned at a higher rate (~24%) than those with a zero balance (~14%).
* **Action:** Created a binary `HasZeroBalance` feature, which proved more predictive than the raw continuous `Balance` variable.

### Interaction Effects
[Image of a interaction plot between NumOfProducts and Exited hue by HasZeroBalance]
* **Product Sweet Spot:** Retention is highest for customers with **two products**. Churn risk skyrockets at three or four products.
* **Moderating Variables:** Age significantly moderates the effect of Active Status on churn. Inactive members churn at nearly 25%, especially in older cohorts (38–51).

---

## 🛠️ Data Preprocessing Pipeline
To handle the high class imbalance (~20% churn) and mixed data types, I constructed a robust `imblearn` pipeline:

1.  **Winsorization:** Capping extreme outliers in numerical columns using IQR.
2.  **Transformation:** Log1p transform for skewed variables (e.g., Age).
3.  **Resampling:** Applied **SMOTENC** (Synthetic Minority Over-sampling Technique for Nominal and Continuous data) to balance the target classes.
4.  **Feature Engineering:** Integrated custom **Interaction Terms**:
    * $Products \times ZeroBalance$
    * $ActiveMember \times Age$
    * $Products \times Age$

---

## 🤖 Model Implementation & Performance
I benchmarked five classification algorithms: **Logistic Regression, SVC, Random Forest, AdaBoost, and XGBoost**.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.9986** | 0.9951 | 0.9982 | **0.9966** |
| **XGBoost** | 0.9986 | 0.9951 | 0.9982 | 0.9966 |

### Final Selection: Logistic Regression
Despite the power of ensemble methods, **Logistic Regression** (with L1 penalty) was chosen for its interpretability and near-perfect performance. The model achieves 99% accuracy on the test set, primarily driven by the `Complain` variable.



---

## 💡 Strategic Recommendations
1.  **The Complaint Task Force:** Since complaints are a nearly 1:1 precursor to churn, the bank should implement an immediate "Red Alert" protocol for any customer filing a formal complaint.
2.  **Product Bundling:** Investigate why customers with 3+ products are leaving. This suggests that the bank's "premium" or "complex" offerings may be driving friction rather than loyalty.
3.  **Targeting "Silent Churners":** Future models should exclude the `Complain` variable to identify subtle behavioral patterns in customers who "ghost" the bank without providing feedback.

---

## 🛠️ Installation & Usage
1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/yourusername/bank-churn-prediction.git](https://github.com/yourusername/bank-churn-prediction.git)
    ```
2.  **Install Dependencies:**
    ```bash
    pip install pandas numpy seaborn scikit-learn imbalanced-learn feature-engine tabulate
    ```
3.  **Run the Notebook:**
    Execute `bank_churn_analysis.ipynb` to reproduce the statistical tests and pipeline.

---

### Connect with Me
* [Portfolio Website](https://estevanfisk.com/)
* [GitHub Profile](https://github.com/EstevanFisk)
* [Kaggle Profile](https://www.kaggle.com/bgfootball04)