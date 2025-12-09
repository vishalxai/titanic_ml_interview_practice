

# streamlit_app/app.py
"""
Streamlit app for Titanic survival probability prediction.
This file is intended to live inside the titanic_set/streamlit_app directory.
It inserts the repository root onto sys.path so the pickled pipeline (which
references `src`) can be unpickled successfully when running in this folder.
"""
from pathlib import Path
import sys
import joblib
import pandas as pd
import streamlit as st

# --- make project package importable when running this script ---
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# --- model / artifact locations ---
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# load artifacts (pipeline includes preprocessing)
pipeline = joblib.load(MODELS_DIR / "pipeline.joblib")
model = joblib.load(MODELS_DIR / "rf.joblib")

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("Titanic Survival Prediction App 🚢")

st.write("Enter passenger data on the left and press Predict to get survival probability.")

# ---------------------- Sidebar inputs ----------------------
st.sidebar.header("Passenger features")
Pclass = st.sidebar.selectbox("Pclass", options=[1, 2, 3], index=2)
Sex = st.sidebar.selectbox("Sex", options=["male", "female"], index=0)
Age = st.sidebar.number_input("Age", min_value=0.0, max_value=120.0, value=30.0, step=0.5)
SibSp = st.sidebar.number_input("Siblings/Spouses aboard (SibSp)", min_value=0, max_value=10, value=0)
Parch = st.sidebar.number_input("Parents/Children aboard (Parch)", min_value=0, max_value=10, value=0)
Fare = st.sidebar.number_input("Fare", min_value=0.0, max_value=10000.0, value=32.2, step=0.1)

Embarked = st.sidebar.selectbox("Embarked", options=["S", "C", "Q", "Missing"], index=0)

# Name field so pipeline can extract Title; if you don't have real name use a placeholder
Name = st.sidebar.text_input("Name (first Last)", value="Mr. John Doe")
Ticket = st.sidebar.text_input("Ticket (e.g. A/5 21171)", value="" )
Cabin = st.sidebar.text_input("Cabin (e.g. C85)", value="")

# ---------------------- Prepare input DataFrame ----------------------
# create a single-row dataframe that matches what create_features expects
input_dict = {
    "PassengerId": [0],
    "Pclass": [Pclass],
    "Name": [Name],
    "Sex": [Sex],
    "Age": [Age],
    "SibSp": [SibSp],
    "Parch": [Parch],
    "Ticket": [Ticket],
    "Fare": [Fare],
    "Cabin": [Cabin],
    "Embarked": [Embarked if Embarked != "Missing" else None],
}
input_df = pd.DataFrame(input_dict)

st.subheader("Preview input")
st.table(input_df.T)

# ---------------------- Prediction ----------------------
if st.button("Predict Survival Probability"):
    # pipeline expects the raw DataFrame (it runs create_features internally)
    X_trans = pipeline.transform(input_df)
    proba = model.predict_proba(X_trans)[:, 1][0]
    st.metric(label="Predicted survival probability", value=f"{proba:.3f}")
    st.write("Probability > 0.5 indicates model predicts survival.")

# ---------------------- Bulk prediction (test.csv) ----------------------
st.markdown("---")
if st.button("Generate predictions for data/raw/test.csv (save to reports/predictions.csv)"):
    test_path = Path(REPO) / "data" / "raw" / "test.csv"
    if not test_path.exists():
        st.error(f"Test file not found: {test_path}")
    else:
        df_test = pd.read_csv(test_path)
        X_test = df_test.copy()  # pipeline will create features
        X_trans = pipeline.transform(X_test)
        preds = model.predict_proba(X_trans)[:, 1]
        out = pd.DataFrame({"PassengerId": df_test["PassengerId"], "survived_proba": preds})
        out_path = Path(REPO) / "reports" / "predictions_streamlit.csv"
        out.to_csv(out_path, index=False)
        st.success(f"Saved predictions → {out_path}")
        st.dataframe(out.head(10))

st.caption("App uses the saved pipeline and RandomForest model from models/*.joblib")