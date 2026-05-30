#  Diabetes Prediction System Using Machine Learning

##  Problem Statement

Diabetes is one of the most common chronic diseases worldwide and early diagnosis is crucial for preventing serious health complications. Traditional diagnosis methods may not always support timely risk assessment. This project aims to predict whether a person is diabetic or non-diabetic using machine learning techniques based on medical and lifestyle-related features.

---

##  Project Objectives

- Analyze health-related factors associated with diabetes.
- Identify important predictors of diabetes.
- Predict diabetes status using machine learning models.
- Compare multiple classification algorithms.
- Explore hidden patterns using clustering techniques.
- Deploy a real-time diabetes prediction application.

---

##  Model Deployment & Live Demo

The trained model is deployed using Streamlit for real-time predictions.
Live demo link : https://diabetesmlproject-nfoyyhwzjhmwxreidtyrac.streamlit.app/

###  Run Locally

```bash
git clone <repo-url>
pip install -r requirements.txt
streamlit run app/app.py
```

---

##  Dataset Overview

The project uses the Diabetes Prediction Dataset containing medical and lifestyle-related information.

###  Key Features

- Age
- BMI
- HbA1c Level
- Blood Glucose Level
- Hypertension
- Heart Disease
- Smoking History

###  Target Variable

- Diabetes (0 = Non-Diabetic, 1 = Diabetic)

---

##  Capabilities

- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Engineering
- Feature Selection
- SMOTE for Class Balancing
- Machine Learning Classification
- Regression Analysis
- Clustering Analysis
- Hyperparameter Tuning
- Model Deployment

---

## Exploratory Data Analysis (EDA)

EDA was performed to understand data characteristics and identify important patterns.

- Missing value analysis
- Duplicate record detection
- Distribution analysis
- Correlation analysis
- Outlier detection using IQR
- Diabetes risk pattern identification

---

##  Data Preprocessing

The following preprocessing steps were performed:

- Handling missing values
- Outlier treatment using IQR capping
- Feature scaling using StandardScaler
- Feature engineering
- Feature selection
- Train-test split
- SMOTE for class balancing

---

##  Feature Engineering

Custom features were created to improve predictive performance:

- Risk Score
- Blood Sugar Risk
- Age-Hypertension Interaction
- BMI-Glucose Interaction

---

##  Machine Learning Models Implemented

###  Classification Models

- Logistic Regression
- Decision Tree
- Random Forest
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

###  Regression Models

- Multiple Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regression
- Random Forest Regression

###  Clustering Models

- K-Means Clustering
- DBSCAN

---

##  Model Evaluation

Classification models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

Regression models were evaluated using:

- R² Score
- RMSE
- MAE

Clustering models were evaluated using:

- Silhouette Score

---

##  Best Models

| Task | Best Model |
|--------|------------|
| Classification | Random Forest |
| Regression | Polynomial Regression |
| Clustering | DBSCAN |

---

##  Key Insights

- HbA1c Level and Blood Glucose Level were the strongest predictors.
- Feature engineering significantly improved classification performance.
- SMOTE helped balance class distribution and improve recall.
- Random Forest achieved the best overall classification performance.
- DBSCAN produced the highest clustering quality based on Silhouette Score.

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Joblib
- Streamlit
- Jupyter Notebook
- VS Code
- Git
- GitHub

---

##  Future Scope

- Use larger real-world datasets
- Improve prediction accuracy
- Add deep learning models
- Enhance Streamlit UI
- Deploy on cloud platforms
- Extend for multi-disease prediction

---

Machine Learning Project – Diabetes Prediction System
