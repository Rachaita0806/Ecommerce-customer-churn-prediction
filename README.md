# E-Commerce Customer Intelligence & Churn Prediction System

## Project Overview

This project is a complete, end-to-end **E-Commerce Customer Intelligence & Churn Prediction System** built as an interactive Streamlit web application. It analyses customer behaviour from a 50,000-record e-commerce dataset, identifies churn patterns, segments customers into meaningful groups, trains multiple machine learning classifiers, and provides an interactive interface for churn probability prediction and data-driven business recommendations.

---

## Objective

The system helps an e-commerce business:

1. Understand customer demographics and behaviour.
2. Analyse purchasing and engagement patterns.
3. Identify customers who are more likely to churn.
4. Understand the factors associated with churn.
5. Segment customers into meaningful groups using K-Means clustering.
6. Build machine learning models for binary churn prediction.
7. Display churn probability and risk level for individual customers.
8. Generate data-driven business insights and actionable recommendations.
9. Present all findings through an interactive Streamlit dashboard.

The primary machine learning target variable is **`Churned`** (binary: 0 = Retained, 1 = Churned).

---

## Technology Stack

| Component | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Backend** | Python (integrated into the same `.py` application file) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly |
| **Machine Learning** | Scikit-learn |
| **Model Persistence** | Joblib |
| **Dataset** | Kaggle E-Commerce Customer Behavior Dataset |

- **Frontend:** Streamlit
- **Backend:** Python
- **ML Models:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib
- **Dataset:** Kaggle E-Commerce Customer Behavior Dataset

---

## Dataset

**Dataset Name:** E-Commerce Customer Behavior Dataset  
**Source:** [https://www.kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset](https://www.kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset)

The dataset contains approximately **50,000 customer records** and **25 columns**:

| # | Column | Description |
|---|---|---|
| 1 | Age | Customer age |
| 2 | Gender | Customer gender (Male/Female/Other) |
| 3 | Country | Customer country |
| 4 | City | Customer city |
| 5 | Membership_Years | Duration of platform membership in years |
| 6 | Login_Frequency | Average number of logins per month |
| 7 | Session_Duration_Avg | Average session duration in minutes |
| 8 | Pages_Per_Session | Average number of pages viewed per session |
| 9 | Cart_Abandonment_Rate | Percentage of cart sessions abandoned |
| 10 | Wishlist_Items | Number of items saved to wishlist |
| 11 | Total_Purchases | Total number of purchases made |
| 12 | Average_Order_Value | Average monetary value per order ($) |
| 13 | Days_Since_Last_Purchase | Days elapsed since the customer's last purchase |
| 14 | Discount_Usage_Rate | Percentage of purchases using a discount |
| 15 | Returns_Rate | Percentage of purchases returned |
| 16 | Email_Open_Rate | Percentage of marketing emails opened |
| 17 | Customer_Service_Calls | Number of customer service calls made |
| 18 | Product_Reviews_Written | Number of product reviews written |
| 19 | Social_Media_Engagement_Score | Composite social media engagement score |
| 20 | Mobile_App_Usage | Mobile app usage percentage |
| 21 | Payment_Method_Diversity | Number of distinct payment methods used |
| 22 | Lifetime_Value | Customer lifetime value ($) |
| 23 | Credit_Balance | Customer credit balance ($) |
| 24 | Churned | **Target variable**: 1 = Churned, 0 = Retained |
| 25 | Signup_Quarter | Quarter in which the customer signed up (Q1–Q4) |

---

## Project Structure

```
E-Commerce-Customer-Intelligence-Churn-Prediction-System/
│
├── RachaitaBhattacharjee_E-Commerce-Customer-Intelligence-&-Churn-Prediction-System.py
│   └── Complete application: frontend + backend + analytics + ML  ← SINGLE FILE
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
├── RachaitaBhattacharjee_E-Commerce-Customer-Intelligence-&-Churn-Prediction-System_ProjectReport.docx
│   └── Formal academic project report
│
└── ecommerce_customer_churn_dataset.csv
    └── Kaggle dataset (place in the same directory as the script)
```

> **Important:** The entire application is intentionally implemented in **ONE Python file**.  
> There are **no** separate `frontend.py`, `backend.py`, `model.py`, `analytics.py`, or any other Python source files.  
> All logic — frontend rendering, data loading, cleaning, feature engineering, analytics, ML training, prediction, and business insights — resides in the single `.py` application file.

---

## Frontend

Streamlit provides the complete interactive interface including:

- **Executive Dashboard** — KPI cards, churn distribution, country-level analysis, engagement overview
- **Customer Analytics** — Demographics, membership, engagement, purchase behaviour, customer value
- **Churn Analysis** — Factor-by-factor churn association analysis, correlation heatmap
- **Customer Segmentation** — K-Means clustering with elbow/silhouette selection, PCA projection
- **Machine Learning** — Model training, evaluation metrics, ROC curves, confusion matrices, feature importance
- **Churn Prediction** — Interactive individual-customer prediction form with probability gauge
- **Business Insights** — Data-driven observations, interpretations, and recommendations
- **Data Quality** — Missing values, duplicates, data types summary
- **Sidebar Filters** — Country, Gender, Signup Quarter, Churn Status, Customer Value Category, Membership Category

---

## Backend

Python logic inside the **same `.py` file** handles:

- Data uploaded from local CSV file
- Column validation against expected schema
- Data cleaning (duplicates, outlier capping, type conversion)
- Feature engineering (Purchase_Value, Engagement_Score, Customer_Value_Category, Membership_Category, Recency_Category)
- Descriptive analytics and KPI calculation
- K-Means customer segmentation
- Data leakage assessment
- Feature selection (justified by correlation analysis and leakage review)
- ML preprocessing via `sklearn.pipeline.Pipeline` + `ColumnTransformer`
- Model training (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- Model evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
- Churn probability prediction with configurable threshold
- Business insight generation from actual dataset statistics
- Model persistence via Joblib

---

## Machine Learning

### Classification Models

| Model | Notes |
|---|---|
| **Logistic Regression** | Balanced class weights, L2 regularisation |
| **Decision Tree** | Max depth 8, balanced class weights |
| **Random Forest** | 150 estimators, max depth 10, balanced class weights |
| **Gradient Boosting** | 100 estimators, learning rate 0.1, max depth 4 |

### Preprocessing Pipeline

- **Numerical features:** Median imputation → Standard scaling
- **Categorical features:** Most-frequent imputation → One-hot encoding
- Train/test split: 80/20 with stratification on `Churned`
- Preprocessing fitted **only** on training data to prevent data leakage

### Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Overall fraction of correct predictions |
| **Precision** | Of all predicted churners, how many actually churned |
| **Recall** | Of all actual churners, how many were correctly identified |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Area under the ROC curve; threshold-independent discrimination |
| **Confusion Matrix** | TN, FP, FN, TP breakdown |

> In churn prediction, **Recall** and **ROC-AUC** are prioritised over Accuracy alone, as missing a churner (false negative) typically has higher business cost than a false positive.

---

## Customer Segmentation

K-Means clustering is applied to 13 selected behavioural and value features:

`Membership_Years`, `Login_Frequency`, `Session_Duration_Avg`, `Pages_Per_Session`, `Total_Purchases`, `Average_Order_Value`, `Days_Since_Last_Purchase`, `Discount_Usage_Rate`, `Returns_Rate`, `Email_Open_Rate`, `Social_Media_Engagement_Score`, `Mobile_App_Usage`, `Lifetime_Value`

- Missing values are imputed with the column median before clustering.
- Features are standardised (zero mean, unit variance) before clustering.
- `Churned` is **NOT** used as a clustering input; it is displayed post-hoc to validate segment quality.
- Optimal K is determined using the **Elbow Method** (inertia) and **Silhouette Score**.
- Segment names are assigned programmatically based on actual cluster characteristics (engagement, LTV, churn rate, recency).
- A PCA-based 2D scatter plot visualises cluster separation.

---

## Churn Prediction

- The prediction page allows users to input values for the final selected ML features.
- The selected model returns a **churn probability** (0–100%).
- A configurable **decision threshold** determines the binary prediction class.
- Risk categories:
  - 🟢 **Low Risk** — probability < 35%
  - 🟡 **Medium Risk** — 35% ≤ probability < 65%
  - 🔴 **High Risk** — probability ≥ 65%
- An interactive gauge chart visualises the probability.
- A disclaimer is displayed clarifying that probabilities are model estimates, not guarantees.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run "RachaitaBhattacharjee_E-Commerce-Customer-Intelligence-&-Churn-Prediction-System.py"
```

---

## Dataset Usage

The application uses the **E-Commerce Customer Behavior Dataset** provided by Kaggle.

The dataset should be downloaded as a CSV file and placed in the **same directory as the Python application**.

Expected file:

```text
ecommerce_customer_churn_dataset.csv

## Important Notes

- `Churned` is the ML target variable and is **never** included as an input feature during training.
- A **data leakage assessment** is performed before model training; `Lifetime_Value` is excluded from ML features as a precaution against potential forward-looking aggregation.
- **Feature selection** is performed based on correlation analysis, missing-value rates, and leakage review — not all columns are blindly used.
- All model evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC) are **calculated from the actual dataset** using a held-out 20% test set.
- All business insights are derived from **actual dataset statistics**, not fabricated values.
- No hardcoded machine-specific file paths are used.
- No separate frontend or backend Python files exist.

---

## Author

**Rachaita Bhattacharjee**  
Dataset: [E-Commerce Customer Behavior Dataset — Kaggle](https://www.kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset)
