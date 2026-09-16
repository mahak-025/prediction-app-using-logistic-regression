import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Logistic Regression Predictor", layout="centered")
st.title("🔮 Logistic Regression - Prediction App")

# ---------------- LOAD MODEL BUNDLE ----------------
@st.cache_resource
def load_model():
    with open("model_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"]

try:
    classifier, sc = load_model()
    st.success("✅ Model loaded successfully")
except FileNotFoundError:
    st.error("⚠️ model_bundle.pkl nahi mili. Same folder mein rakho jahan app.py hai.")
    st.stop()

st.markdown("---")

# ---------------- SINGLE PREDICTION (Manual Input) ----------------
st.header("Single Prediction")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
with col2:
    salary = st.number_input("Estimated Salary", min_value=0, value=50000)

if st.button("Predict"):
    input_data = np.array([[age, salary]])
    input_scaled = sc.transform(input_data)
    prediction = classifier.predict(input_scaled)[0]
    proba = classifier.predict_proba(input_scaled)[0]

    if prediction == 1:
        st.success(f"✅ Prediction: **Purchased (1)** | Confidence: {proba[1]*100:.2f}%")
    else:
        st.info(f"❌ Prediction: **Not Purchased (0)** | Confidence: {proba[0]*100:.2f}%")

st.markdown("---")

# ---------------- BULK PREDICTION (CSV Upload) ----------------
st.header("Bulk Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV file (columns: Age, EstimatedSalary)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    try:
        features = data[["Age", "EstimatedSalary"]].values
        features_scaled = sc.transform(features)

        data["Prediction"] = classifier.predict(features_scaled)

        st.subheader("Prediction Results")
        st.dataframe(data)

        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
    except KeyError:
        st.error("⚠️ CSV mein 'Age' aur 'EstimatedSalary' columns nahi mile. Column names check karo.")