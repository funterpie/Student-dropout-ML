import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# Load trained model
model = joblib.load("models/dropout_model.pkl")

st.set_page_config(page_title="Student Dropout Risk System", layout="wide")

st.title("🎓 Student Dropout Early Warning System By Muhammad Taha Sattar")
st.write("Upload a student CSV to predict dropout risk and visualize risk metrics.")

# Upload CSV
uploaded_file = st.file_uploader("Upload Student CSV here", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # Drop target column if exists
    if "Class" in df.columns:
        X = df.drop("Class", axis=1)
    else:
        X = df.copy()

    # Encode categorical columns
    for col in X.select_dtypes(include="object"):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    try:
        # Ensure same column order as training
        if hasattr(model, "feature_names_in_"):
            X = X[model.feature_names_in_]

        # Predict probabilities
        risk_scores = model.predict_proba(X)[:,1]

        df["risk_score"] = risk_scores

        # Risk labeling
        def label(score):
            if score >= 0.7:
                return "High"
            elif score >= 0.3:
                return "Medium"
            else:
                return "Low"

        df["risk_label"] = df["risk_score"].apply(label)
        df["predicted_dropout"] = model.predict(X)

        # Metrics cards
        high = (df["risk_label"] == "High").sum()
        medium = (df["risk_label"] == "Medium").sum()
        low = (df["risk_label"] == "Low").sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("High Risk Students", high)
        col2.metric("Medium Risk Students", medium)
        col3.metric("Low Risk Students", low)

        # Top high-risk students table
        st.subheader("Top 20 High-Risk Students")
        top_students = df.sort_values("risk_score", ascending=False).head(20)
        st.dataframe(top_students)

        # Risk Score Histogram
        st.subheader("Risk Score Distribution")
        fig = px.histogram(df, x="risk_score", nbins=20, title="Distribution of Student Risk Scores")
        st.plotly_chart(fig)

        # Feature importance (if model has it)
        if hasattr(model, "feature_importances_"):
            st.subheader("Feature Importance")
            importance = model.feature_importances_
            importance_df = pd.DataFrame({
                "feature": X.columns,
                "importance": importance
            }).sort_values("importance", ascending=False)
            st.bar_chart(importance_df.set_index("feature"))

        # Download CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions CSV",
            csv,
            "dropout_predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error("Prediction failed. Check dataset format.")
        st.code(str(e))
