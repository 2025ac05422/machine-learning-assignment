import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="E-Commerce Purchase Predictor", layout="wide")

st.title("🛒 Online Shopper Purchasing Intention Predictor")
st.markdown("Upload visitor session data to evaluate how well our machine learning models predict if a user will generate revenue.")

st.sidebar.header("Model Configuration")
uploaded_file = st.sidebar.file_uploader("1. Upload test data (CSV)", type=["csv"])

model_choice = st.sidebar.selectbox("2. Select Classifier Algorithm", 
                                ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Session Data Preview", df.head())
    
    if 'Revenue' in df.columns:
        X_test = df.drop('Revenue', axis=1)
        y_test = df['Revenue']
    else:
        st.error("Missing 'Revenue' target column in the uploaded dataset.")
        st.stop()
        
    model_filename = model_choice.replace(" ", "_").lower() + '.pkl'
    model_path = os.path.join('model', model_filename)
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
        st.divider()
        st.subheader(f"Performance Metrics: {model_choice}")
        
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        m2.metric("AUC", f"{roc_auc_score(y_test, y_prob):.4f}")
        m3.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        m4.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.4f}")
        m5.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
        m6.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax, 
                        xticklabels=['No Purchase', 'Purchase'], yticklabels=['No Purchase', 'Purchase'])
            st.pyplot(fig)
            
        with col2:
            st.write("**Classification Report**")
            st.text(classification_report(y_test, y_pred, zero_division=0))
            
    else:
        st.warning(f"Model file {model_filename} not found. Please train models first.")
else:
    st.info("Please upload your `test_data.csv` via the sidebar to begin testing.")