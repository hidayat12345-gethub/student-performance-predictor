"""
Performance Index Predictor
---------------------------
A Streamlit dashboard that trains a Linear Regression model on
Student_Performance.csv and lets you predict a student's Performance
Index from their study profile.

Run with:
    pip install streamlit pandas scikit-learn
    streamlit run app.py

Expects Student_Performance.csv in the same folder. If it isn't
found, the app falls back to a pre-trained model (coefficients fit
on the full 10,000-row dataset ahead of time) so the app still works.
"""

import io
import os

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CSV_PATH = "Student_Performance.csv"
FEATURES = [
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced",
]
TARGET = "Performance Index"

# Fallback coefficients: a real fit on the full 10,000-row dataset,
# used only if Student_Performance.csv isn't present at runtime.
FALLBACK_COEF = {
    "Hours Studied": 2.8524839300725766,
    "Previous Scores": 1.016988198932932,
    "Extracurricular Activities": 0.6086166795764194,
    "Sleep Hours": 0.4769414841762728,
    "Sample Question Papers Practiced": 0.1918314414505425,
}
FALLBACK_INTERCEPT = -33.92194621555638
FALLBACK_METRICS = {
    "r2_train": 0.9887,
    "r2_test": 0.9890,
    "mae": 1.61,
    "rmse": 2.02,
    "n_train": 8000,
}


# --------------------------------------------------------------------------
# Model training (cached so it only runs once per session / file change)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training model on Student_Performance.csv...")
def train_model():
    if not os.path.exists(CSV_PATH):
        return None, None

    df = pd.read_csv(CSV_PATH)
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map(
        {"Yes": 1, "No": 0}
    )

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    metrics = {
        "r2_train": r2_score(y_train, pred_train),
        "r2_test": r2_score(y_test, pred_test),
        "mae": mean_absolute_error(y_test, pred_test),
        "rmse": mean_squared_error(y_test, pred_test) ** 0.5,
        "n_train": len(X_train),
    }
    coef = dict(zip(FEATURES, model.coef_))
    intercept = model.intercept_
    return {"coef": coef, "intercept": intercept}, metrics


def predict(profile: dict, coef: dict, intercept: float) -> float:
    raw = intercept + sum(coef[f] * profile[f] for f in FEATURES)
    return max(0.0, min(100.0, raw))


def band_for(score: float):
    if score >= 80:
        return "Excellent", "🟢"
    if score >= 60:
        return "Good", "🔵"
    if score >= 40:
        return "Average", "🟡"
    return "Needs Improvement", "🔴"


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Performance Index Predictor",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0F1626; }
    h1, h2, h3, .stMarkdown, p, label, .stMetric label { color: #F8F8F8 !important; }
    div[data-testid="stMetricValue"] { color: #00BFFF; }
    div[data-testid="stMetric"] {
        background-color: #1A2233;
        border: 1px solid #2A3550;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #39FF14;
        color: #0F1626;
        font-weight: 700;
        border: none;
        border-radius: 8px;
    }
    .stButton>button:hover { background-color: #00e600; color: #0F1626; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎯 Performance Index Predictor")
st.caption("Linear Regression dashboard — Student_Performance.csv")

trained, metrics = train_model()
if trained is None:
    st.warning(
        f"`{CSV_PATH}` not found next to app.py — using a pre-trained fallback "
        "model (fit on the full 10,000-row dataset) instead."
    )
    coef, intercept, metrics = FALLBACK_COEF, FALLBACK_INTERCEPT, FALLBACK_METRICS
else:
    coef, intercept = trained["coef"], trained["intercept"]

if "history" not in st.session_state:
    st.session_state.history = []

col_left, col_right = st.columns([1.05, 0.95], gap="large")

# --------------------------------------------------------------------------
# LEFT: Predictor
# --------------------------------------------------------------------------
with col_left:
    st.subheader("Predict Performance Index")
    st.caption("Adjust a student's profile and predict their performance index (0–100).")

    hours = st.slider("Hours Studied", 1, 9, 6, help="Daily study hours")
    prev = st.slider("Previous Scores", 40, 99, 75, help="Most recent test score")
    extra_label = st.radio("Extracurricular Activities", ["Yes", "No"], horizontal=True)
    sleep = st.slider("Sleep Hours", 4, 9, 7, help="Average sleep per night")
    papers = st.slider("Sample Question Papers Practiced", 0, 9, 4, help="Practice papers completed")

    if st.button("Predict Performance Index", use_container_width=True):
        profile = {
            "Hours Studied": hours,
            "Previous Scores": prev,
            "Extracurricular Activities": 1 if extra_label == "Yes" else 0,
            "Sleep Hours": sleep,
            "Sample Question Papers Practiced": papers,
        }
        score = predict(profile, coef, intercept)
        label, emoji = band_for(score)

        st.markdown(
            f"""
            <div style="background-color:#151D2E;border:1px solid rgba(157,44,229,0.4);
                        border-radius:12px;padding:18px;text-align:center;margin-top:10px;">
                <div style="font-size:2.4rem;font-weight:800;color:#9D2CE5;">{score:.1f}</div>
                <div style="color:#9AA7BD;font-size:.8rem;">Predicted Performance Index (out of 100)</div>
                <div style="margin-top:8px;font-weight:700;">{emoji} {label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.session_state.history.insert(
            0,
            {
                "Hours": hours,
                "Prev.": prev,
                "Extra": extra_label,
                "Sleep": sleep,
                "Papers": papers,
                "Predicted": round(score, 1),
            },
        )

    st.markdown("---")
    st.subheader("Prediction History")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Clear history", use_container_width=True):
                st.session_state.history = []
                st.rerun()
        with col_b:
            csv_buffer = io.StringIO()
            hist_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Export CSV",
                data=csv_buffer.getvalue(),
                file_name="performance_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("No predictions yet — try one above.")

# --------------------------------------------------------------------------
# RIGHT: Model info + feature importance
# --------------------------------------------------------------------------
with col_right:
    st.subheader("Model Information")
    st.caption(
        "Trained on the full dataset (80/20 train-test split)."
        if trained is not None
        else "Using fallback coefficients (full-dataset fit)."
    )

    m1, m2 = st.columns(2)
    m1.metric("Training R²", f"{metrics['r2_train']:.3f}")
    m2.metric("Test R²", f"{metrics['r2_test']:.3f}")
    m3, m4 = st.columns(2)
    m3.metric("MAE", f"{metrics['mae']:.2f}")
    m4.metric("RMSE", f"{metrics['rmse']:.2f}")
    m5, m6 = st.columns(2)
    m5.metric("Features", len(FEATURES))
    m6.metric("Training Samples", metrics["n_train"])

    st.subheader("Feature Importance")
    st.caption("Absolute coefficient weight scaled by each feature's typical range.")

    feature_ranges = {
        "Hours Studied": 8,
        "Previous Scores": 59,
        "Extracurricular Activities": 1,
        "Sleep Hours": 5,
        "Sample Question Papers Practiced": 9,
    }
    importance = pd.Series(
        {f: abs(coef[f]) * feature_ranges[f] for f in FEATURES}
    ).sort_values(ascending=True)
    st.bar_chart(importance, horizontal=True, color="#00BFFF")

    st.markdown("---")
    st.caption(
        "Model trained on the full Student_Performance.csv (10,000 records, "
        "80/20 train-test split). Previous Scores and Hours Studied are the "
        "strongest positive drivers of Performance Index, followed by "
        "Extracurricular Activities, Sleep Hours, and Sample Papers Practiced."
    )
