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
COMPARISON_PATH = ROOT / "models" / "model_comparison.csv"

# Look of each result: emoji, card colour, bar colour, meaning
TIERS = {
    "Low":    {"emoji": "🌱", "card": "#FFE3E3", "bar": "#F4A6A6", "text": "fewer than 10,000 installs"},
    "Medium": {"emoji": "⭐", "card": "#FFF3C4", "bar": "#F2C94C", "text": "10,000 to 999,999 installs"},
    "High":   {"emoji": "🚀", "card": "#D6F5E3", "bar": "#6FCF97", "text": "1,000,000 or more installs"},
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


def start_at(values, wanted):
    """Position of the value we want to show first (0 if it is not in the list)."""
    return values.index(wanted) if wanted in values else 0


def confidence_bar(label, value, colour):
    """One small coloured bar, drawn with HTML."""
    return (f"<div style='margin:8px 0 2px 0'>{label}: <b>{value:.0%}</b></div>"
            f"<div style='background:#EEE9F7; border-radius:8px; height:12px'>"
            f"<div style='width:{value * 100:.0f}%; background:{colour}; "
            f"height:12px; border-radius:8px'></div></div>")


def score_bar(label, score, colour):
    """A bar for one model's score (0 to 1), with the number shown."""
    return (f"<div style='margin:8px 0 2px 0'>{label}: <b>{score:.3f}</b></div>"
            f"<div style='background:#EEE9F7; border-radius:8px; height:12px'>"
            f"<div style='width:{score * 100:.0f}%; background:{colour}; "
            f"height:12px; border-radius:8px'></div></div>")


# ---------- 2. Page header ----------
st.set_page_config(page_title="App Success Predictor", page_icon="📱")

st.markdown(
    "<div style='background:linear-gradient(135deg,#E8E0FF,#FFE3EC,#DFF5EA); "
    "padding:24px; border-radius:16px; margin-bottom:16px; color:#3A3350'>"
    "<h1 style='margin:0'>📱 Mobile App Success Predictor</h1>"
    "<p style='margin:8px 0 0 0'>Tell us about your app before launch, and the model "
    "predicts how many installs it is likely to reach.</p></div>",
    unsafe_allow_html=True)

if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("Model or data file not found. Check that models/best_model.joblib "
             "and data/googleplaystore_final_cleaned.csv exist in the repo.")
    st.stop()

model = load_model()
options = load_options()

# ---------- 3. Inputs ----------
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("🎮 Category", options["category"],
                                index=start_at(options["category"], "GAME"),
                                format_func=lambda x: x.replace("_", " ").title())
        genres = st.selectbox("🎨 Main genre", options["genres"],
                              index=start_at(options["genres"], "Casual"))
        content_rating = st.selectbox("👪 Content rating", options["content_rating"],
                                      index=start_at(options["content_rating"], "Everyone"))
        android = st.selectbox("🤖 Minimum Android version", options["android"],
                               index=start_at(options["android"], 4.0),
                               format_func=lambda v: f"Android {v}+")

    with col2:
        app_type = st.radio("💰 Pricing", ["Free", "Paid"], horizontal=True)
        price = st.number_input("Price (USD)", min_value=0.0, max_value=400.0,
                                value=0.99, step=0.5, disabled=(app_type == "Free"))
        size_varies = st.checkbox("Size varies with device")
        size_mb = st.number_input("📦 App size (MB)", min_value=0.0, max_value=100.0,
                                  value=15.0, step=1.0, disabled=size_varies)

# ---------- 4. Validation ----------
errors = []
if app_type == "Paid" and price <= 0:
    errors.append("A paid app needs a price above 0.")
if not size_varies and size_mb <= 0:
    errors.append("Enter the app size, or tick 'Size varies with device'.")

# ---------- 5. Predict ----------
if st.button("✨ Predict success tier", type="primary"):
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
        look = TIERS[tier]
        st.markdown(
            f"<div style='background:{look['card']}; padding:20px; "
            f"border-radius:14px; color:#3A3350; margin-top:12px'>"
            f"<h2 style='margin:0'>{look['emoji']} Predicted tier: {tier}</h2>"
            f"<p style='margin:6px 0 0 0'>{look['text']}</p></div>",
            unsafe_allow_html=True)

        # Show how confident the model is (if the model supports it)
        if hasattr(model, "predict_proba"):
            probs = pd.Series(model.predict_proba(row)[0], index=model.classes_)
            st.write("")
            st.caption("Model confidence")
            for t in ["Low", "Medium", "High"]:
                st.markdown(confidence_bar(t, probs[t], TIERS[t]["bar"]),
                            unsafe_allow_html=True)

# ---------- 6. Model comparison (sidebar) ----------
with st.sidebar:
    st.markdown("### 📊 How the 6 models compare")
    if COMPARISON_PATH.exists():
        comp = pd.read_csv(COMPARISON_PATH).sort_values("Macro-F1", ascending=False)
        best = comp.iloc[0]["Model"]
        st.write(f"**{best}** scored highest, so it is the model used for the predictions. "
                 "The other models were trained and tested only for comparison.")
        st.caption("Macro-F1 on the test set (higher is better)")
        for _, r in comp.iterrows():
            colour = "#8E7CC3" if r["Model"] == best else "#CFC8E6"
            st.markdown(score_bar(r["Model"], r["Macro-F1"], colour), unsafe_allow_html=True)
    else:
        st.info("Add models/model_comparison.csv to the repo to see this comparison.")
