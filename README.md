# E-Commerce Purchasing Intention ML Deployment

## a. Problem Statement
The objective of this assignment is to develop, evaluate, and deploy multiple machine learning classification algorithms to predict whether an online shopper's website session will culminate in a purchase (`Revenue`). The end-to-end workflow covers the full machine learning engineering lifecycle: data ingestion, preprocessing, model training, performance evaluation, and interactive web application deployment on Streamlit Community Cloud.

## b. Dataset Description
The dataset used for this project is the **Online Shoppers Purchasing Intention Dataset** (ID: 468) from the UCI Machine Learning Repository. It contains 12,330 online user sessions across various online retail environments, comprising 18 feature attributes (10 numerical and 8 categorical) measuring user interactions such as administrative page durations, bounce rates, exit rates, and special day closeness. The target variable is the boolean indicator `Revenue` (True/False), indicating whether the session resulted in a completed transaction. This dataset comfortably satisfies the assignment constraints of a minimum of 12 features and 500 instances.

## c. Github Repository Link
[Insert Your Full GitHub Repository URL Here]

## d. Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8873 | 0.8944 | 0.8799 | 0.8873 | 0.8694 | 0.5010 |
| **Decision Tree** | 0.8617 | 0.7309 | 0.8593 | 0.8617 | 0.8605 | 0.4704 |
| **kNN** | 0.8650 | 0.7566 | 0.8458 | 0.8650 | 0.8431 | 0.3823 |
| **Naive Bayes** | 0.8187 | 0.8387 | 0.8483 | 0.8187 | 0.8302 | 0.4205 |
| **Random Forest (Ensemble)** | 0.8970 | 0.9120 | 0.8890 | 0.8970 | 0.8889 | 0.5689 |

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed strongly as a linear baseline, achieving high accuracy (88.73%) and AUC (0.8944), demonstrating effective linear separability after standard feature encoding. |
| **Decision Tree** | Achieved reasonable accuracy (86.17%) but exhibited a lower AUC score (0.7309), indicating high sensitivity to decision thresholds and slight over-fitting on individual split branches. |
| **kNN** | Yielded solid overall accuracy (86.50%), though it recorded the lowest MCC score (0.3823), reflecting difficulty in balancing true positives against true negatives due to session class imbalance. |
| **Naive Bayes** | Recorded the lowest accuracy (81.87%) among all models, primarily due to the strong conditional dependencies between features like `BounceRates` and `ExitRates` violating independence assumptions. |
| **Random Forest (Ensemble)** | Emerged as the top performer across all six evaluation metrics (89.70% Accuracy, 0.9120 AUC, 0.5689 MCC), effectively capturing complex non-linear relationships between session duration metrics and conversion probability. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** |