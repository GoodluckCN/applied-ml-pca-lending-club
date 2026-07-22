# Applied Machine Learning: Principal Component Analysis (PCA) & Loan Default Prediction

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📌 Project Overview
This repository contains the code and methodology for **Homework 6** of the **Applied Machine Learning course** (taught by Prof. Krzysztof Rybinski). The primary objective of this project is to explore unsupervised dimensionality reduction using **Principal Component Analysis (PCA)** and evaluate its impact on supervised classification performance (Logistic Regression) using a subset of the **LendingClub** loan dataset.

---

## 📂 Project Tasks & Structure
1. **Task 1: PCA Analysis & Biplot Interpretation**
   - Standardized features to zero mean and unit variance.
   - Performed PCA and analyzed the explained variance ratio of principal components.
   - Generated and interpreted feature loadings and biplots to understand underlying data structure.
2. **Task 2: Logistic Regression Performance Comparison**
   - Built and evaluated baseline logistic regression models on original standardized features.
   - Compared model performance (accuracy, ROC-AUC) against models trained on PCA-transformed feature spaces.
3. **Task 3: Optimal Threshold Tuning via ROC Analysis**
   - Utilized Receiver Operating Characteristic (ROC) curves and Youden's J statistic to determine the optimal classification threshold for the PCA-logistic regression model.

---

## 🗂 Dataset Description
The analysis uses a subset of the LendingClub loan dataset (`LendingClub_small.csv`), containing structural financial attributes and loan performance metrics:
- **`loan_status`**: Target variable (Default / Good loan indicator).
- **`loan_amnt`**: The listed amount of the loan applied for.
- **`int_rate`**: Interest rate on the loan.
- **`emp_length`**: Employment length in years.
- **`annual_inc`**: Self-reported annual income.
- **`pub_rec`**: Number of derogatory public records.
- **`home_*`**: Homeownership indicator dummy variables (mortgage, rent, own, other).

*(Note: Raw data files are excluded from this repository due to file size constraints.)*

---
# 📊 Key Findings Summary: LendingClub PCA & Classification Project
1. Data Exploration & Variance Distribution
- Dataset Scale: Analyzed 1,340,973 loan records across 12 financial and demographic features.
- Target Distribution: The loan default rate (loan_status = 1) stands at 22.15%, while non-defaults (0) represent 77.85% of the dataset, indicating a moderately imbalanced classification problem.
- Dimensionality Reduction (PCA):
-- The first 4 principal components capture approximately 67.8% of the total variance in the standardized feature space.
-- Capturing ~90% of the cumulative variance requires 6 principal components, effectively reducing feature dimensionality while retaining the core data structure.

2.  Feature Contributions & Biplot Insights
- Interest Rate & Loan Amount: Strong positive loadings on early components reflect how larger loan amounts scale closely with risk-adjusted interest rates (int_rate mean: 13.32%, max: 30.99%).
- Homeownership & Financial Stability: Homeownership indicators (home_mor averaging 49.4%, home_rent at 39.8%) heavily influence orthogonal axes in the PCA biplot, separating borrowers by collateral/housing status.
- Public Records: Derogatory public records (pub_rec) and bankruptcies (pub_rec_bankruptcies) form distinct high-variance orthogonal vectors, capturing outlier risk profiles separate from standard income metrics.

3. Model Performance & Threshold Tuning
- Logistic Regression Baseline vs. PCA: Training logistic regression on the reduced PCA feature space preserves computational efficiency and stabilizes convergence given the large sample size, though a slight trade-off in raw ROC-AUC occurs compared to using the full feature set.
- Optimal Decision Threshold: Utilizing ROC curve analysis and Youden's J statistic helps overcome class imbalance (22.15% positive class), allowing for a calibrated probability threshold that optimizes the true positive rate while controlling false positives.

## 🛠 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/applied-ml-pca-lending-club.git](https://github.com/your-username/applied-ml-pca-lending-club.git)
   cd applied-ml-pca-lending-club

2. python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

3. pip install -r requirements.txt

# 🚀 Usage
Run the analysis script or open the Jupyter Notebook to replicate data preprocessing, PCA projections, model training, and threshold optimization steps.
