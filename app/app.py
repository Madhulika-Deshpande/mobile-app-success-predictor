"""Mobile App Success Predictor - simple Streamlit UI.

Run from the repo root:  streamlit run app/app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
DATA_PATH = ROOT / "data" / "googleplaystore_final_cleaned.csv"

# Soft background colour for each result
TIER_COLOURS = {"Low": "#EBDDDD", "Medium": "#F1E9CF", "High": "#D8E7DA"}
TIER_MEANING = {
    "Low": "fewer than 10,000 installs",
    "Medium": "10,000 to 999,999 installs",
    "High": "1,000,000 or more installs",
}


# ---------- 1. Load the model and the dropdown options ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_options():
    """Read the cleaned data only to fill the dropdown lists."""
    df = pd.read_csv(DATA_PATH)
    return {
        "category": sorted(df["Category"].dropna().unique()),
        # the model uses only the first genre, e.g. "Art & Design;Creativity" -> "Art & Design"
        "genres": sorted(df["Genres"].dropna().str.split(";").str[0].unique()),
        "content_rating": sorted(df["Content Rating"].dropna().unique()),
        "android": sorted(df["android_ver_min"].dropna().unique()),
    }


# ---------- 2. Page layout ----------
st.set_page_config(page_title="App Success Predictor", page_icon="📱")
st.title("Mobile App Success Predictor")
st.write("Describe your app before launch. The model predicts how many "
         "installs it is likely to reach: Low, Medium or High.")

if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("Model or data file not found. Check that models/best_model.joblib "
             "and data/googleplaystore_final_cleaned.csv exist in the repo.")
    st.stop()

model = load_model()
options = load_options()

# ---- Inputs ----
col1, col2 = st.columns(2)
with col1:
    category = st.selectbox("Category", options["category"],
                            format_func=lambda x: x.replace("_", " ").title())
    genres = st.selectbox("Main genre", options["genres"])
    content_rating = st.selectbox("Content rating", options["content_rating"])
    android = st.selectbox("Minimum Android version", options["android"])

with col2:
    app_type = st.radio("Pricing", ["Free", "Paid"], horizontal=True)
    price = st.number_input("Price (USD)", min_value=0.0, max_value=400.0,
                            value=0.99, step=0.5, disabled=(app_type == "Free"))
    size_varies = st.checkbox("Size varies with device")
    size_mb = st.number_input("App size (MB)", min_value=0.0, max_value=100.0,
                              value=15.0, step=1.0, disabled=size_varies)

# ---------- 3. Validation ----------
errors = []
if app_type == "Paid" and price <= 0:
    errors.append("A paid app needs a price above 0.")
if not size_varies and size_mb <= 0:
    errors.append("Enter the app size, or tick 'Size varies with device'.")

# ---------- 4. Predict ----------
if st.button("Predict success tier", type="primary"):
    if errors:
        for e in errors:
            st.warning(e)
    else:
        # One row, using the same column names the model was trained on
        row = pd.DataFrame([{
            "category": category,
            "genres": genres,
            "size_mb": np.nan if size_varies else size_mb,   # empty -> model fills it
            "size_varies": int(size_varies),
            "type": app_type,
            "price": 0.0 if app_type == "Free" else price,
            "content_rating": content_rating,
            "android_ver_min": android,
        }])
        # If the model was trained with extra columns (e.g. days_since_update),
        # add them as empty so the model does not complain. They are not used.
        for col in model.feature_names_in_:
            if col not in row.columns:
                row[col] = np.nan

        tier = model.predict(row)[0]
        st.markdown(
            f"<div style='background:{TIER_COLOURS[tier]}; padding:18px; "
            f"border-radius:10px; color:#2B3A42;'>"
            f"<h3 style='margin:0'>Predicted tier: {tier}</h3>"
            f"<p style='margin:4px 0 0 0'>{TIER_MEANING[tier]}</p></div>",
            unsafe_allow_html=True)

        # Show how confident the model is (if the model supports it)
        if hasattr(model, "predict_proba"):
            probs = pd.Series(model.predict_proba(row)[0], index=model.classes_)
            st.write("")
            st.caption("Model confidence")
            st.bar_chart(probs[["Low", "Medium", "High"]])

st.caption("Tiers are approximate because Google Play shows installs as ranges. "
           "This is a pattern-based estimate, not a guarantee.")
