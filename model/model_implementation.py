import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder

os.makedirs('model', exist_ok=True)

# 1. Load data directly from UCI via URL
print("Downloading Online Shoppers dataset from UCI...")
url = "https://archive.ics.uci.edu/static/public/468/data.csv"
df = pd.read_csv(url)

# Drop any accidental missing values to be safe
df.dropna(inplace=True)

# 2. Separate features and target (Target column is 'Revenue')
X = df.drop('Revenue', axis=1)
y = df['Revenue']

# 3. Preprocessing
# Convert text features (like Month, VisitorType) into numbers
X = pd.get_dummies(X, drop_first=True)

# Encode the target (False/True to 0/1)
le = LabelEncoder()
y = le.fit_transform(y)

# 4. Train-Test Split (Using a unique seed of 2026)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2026)

# Save a 500-instance subset of test data for the Streamlit app
test_data = X_test.copy()
test_data['Revenue'] = y_test
test_data.sample(500, random_state=2026).to_csv('test_data.csv', index=False)
print("Saved test_data.csv to the root directory.\n")

# 5. Initialize Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000), 
    "Decision Tree": DecisionTreeClassifier(random_state=2026),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=2026)
}

# 6. Train, Evaluate, and Save Models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    print(f"--- {name} ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"MCC: {matthews_corrcoef(y_test, y_pred):.4f}\n")
    
    # Save the trained model 
    filename = name.replace(" ", "_").lower() + '.pkl'
    joblib.dump(model, os.path.join('model', filename))