

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load trained model and scaler
model = xgb.XGBClassifier()
model.load_model("xgb_model.json")   # portable between GPU/CPU
scaler = joblib.load("scaler.pkl")

# 2. Streamlit page config
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection App")

# 3. Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose an option:", ["Predict Single Transaction", "Batch CSV Prediction", "Model Evaluation"])

# --- Predict Single Transaction ---
if page == "Predict Single Transaction":
    st.header("Predict Single Transaction")
    cols = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']
    input_data = []

    for col in cols:
        val = st.number_input(f'Enter {col}:', value=0.0)
        input_data.append(val)

    if st.button("Predict"):
        # Scale Time & Amount (0th and last column)
        input_data[0] = scaler.transform(np.array(input_data[0]).reshape(-1,1))[0][0]
        input_data[-1] = scaler.transform(np.array(input_data[-1]).reshape(-1,1))[0][0]

        pred = model.predict(np.array(input_data).reshape(1,-1))
        st.success("🚨 Fraudulent Transaction!" if pred[0]==1 else "✅ Legitimate Transaction")

# --- Batch CSV Upload ---
elif page == "Batch CSV Prediction":
    st.header("Batch Prediction from CSV")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(df.head())

        # Scale Time & Amount
        df['Time'] = scaler.transform(df['Time'].values.reshape(-1,1))
        df['Amount'] = scaler.transform(df['Amount'].values.reshape(-1,1))

        predictions = model.predict(df)
        df['Prediction'] = ["Fraud" if p==1 else "Legit" for p in predictions]

        st.write("Predictions:")
        st.dataframe(df)

        csv = df.to_csv(index=False)
        st.download_button("Download Predictions CSV", data=csv, file_name="predictions.csv")

# --- Model Evaluation ---
elif page == "Model Evaluation":
    st.header("Model Evaluation")
    st.write("Upload dataset with ground truth labels (must contain 'Class' column)")

    data_file = st.file_uploader("Upload full dataset CSV for evaluation", type=["csv"])
    
    if data_file:
        data = pd.read_csv(data_file)
        st.write("Dataset Preview:")
        st.dataframe(data.head())

        X = data.drop('Class', axis=1)
        y = data['Class']

        # Scale Time & Amount
        X['Time'] = scaler.transform(X['Time'].values.reshape(-1,1))
        X['Amount'] = scaler.transform(X['Amount'].values.reshape(-1,1))

        y_pred = model.predict(X)

        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        st.pyplot(fig)

        # Feature Importance
        st.subheader("Feature Importance")
        feat_importances = model.feature_importances_
        feat_series = pd.Series(feat_importances, index=X.columns).sort_values(ascending=False).head(15)
        fig2, ax2 = plt.subplots()
        sns.barplot(x=feat_series, y=feat_series.index, palette="viridis")
        ax2.set_xlabel("Importance")
        ax2.set_ylabel("Feature")
        st.pyplot(fig2)

        # Metrics
        st.subheader("Classification Report")
        report = classification_report(y, y_pred, output_dict=True)
        st.json(report)
