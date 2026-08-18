"""Interactive model evaluation app for ML Assignment 2."""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model" / "artifacts"
TEST_DATA_PATH = ROOT / "test_data.csv"
TARGET_COLUMN = "target"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

st.set_page_config(
    page_title="Model Lens | ML Assignment 2",
    page_icon="ML",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f4f6f2; }
    [data-testid="stSidebar"] { background: #173f3a; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] small { color: #f4f6f2; }
    /* Widget values sit on a light surface, so they need dark text. */
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #173f3a; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div { background: #ffffff; }
    [data-baseweb="popover"] [role="option"] { color: #173f3a; }
    /* The uploaded-file row sits on the dark sidebar instead. */
    [data-testid="stSidebar"] [data-testid="stFileUploaderFileName"],
    [data-testid="stSidebar"] [data-testid="stFileUploaderDeleteBtn"] svg { color: #f4f6f2; }
    h1, h2, h3 { color: #173f3a; }
    .hero { padding: 1.4rem 0 0.7rem; border-bottom: 1px solid #b8c8c0; }
    .eyebrow { color: #c45a3a; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .metric-card { background: white; border-left: 5px solid #c45a3a; padding: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_model(model_name):
    with (MODEL_DIR / MODEL_FILES[model_name]).open("rb") as artifact:
        return pickle.load(artifact)


def metric_values(model, data):
    features = data.drop(columns=[TARGET_COLUMN], errors="ignore")
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    values = {
        "Accuracy": accuracy_score(data[TARGET_COLUMN], predictions),
        "AUC": roc_auc_score(data[TARGET_COLUMN], probabilities),
        "Precision": precision_score(data[TARGET_COLUMN], predictions, zero_division=0),
        "Recall": recall_score(data[TARGET_COLUMN], predictions, zero_division=0),
        "F1": f1_score(data[TARGET_COLUMN], predictions, zero_division=0),
        "MCC": matthews_corrcoef(data[TARGET_COLUMN], predictions),
    }
    return values, predictions


st.markdown('<div class="hero"><div class="eyebrow">Machine Learning | Assignment 2</div>', unsafe_allow_html=True)
st.title("Model Lens")
st.write("Compare five classification models on the Breast Cancer Wisconsin dataset.")
st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Experiment controls")
    uploaded_file = st.file_uploader("Upload test CSV", type="csv")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES))
    st.caption("The CSV must contain the 30 dataset features. Include target to calculate evaluation metrics.")

if uploaded_file is None:
    data = pd.read_csv(TEST_DATA_PATH)
    source_label = "Bundled held-out test_data.csv"
else:
    data = pd.read_csv(uploaded_file)
    source_label = uploaded_file.name

model = load_model(model_name)
expected_features = list(model.feature_names_in_)
missing_features = [column for column in expected_features if column not in data.columns]
extra_features = [column for column in data.columns if column not in expected_features and column != TARGET_COLUMN]

st.info(f"Evaluating **{source_label}** | {len(data):,} rows")
if missing_features:
    st.error(f"The CSV is missing {len(missing_features)} required feature column(s): {', '.join(missing_features)}")
    st.stop()
if extra_features:
    st.warning(f"Ignoring unrecognised columns: {', '.join(extra_features)}")

features = data[expected_features]
predictions = model.predict(features)
probabilities = model.predict_proba(features)[:, 1]

if TARGET_COLUMN in data.columns:
    values = {
        "Accuracy": accuracy_score(data[TARGET_COLUMN], predictions),
        "AUC": roc_auc_score(data[TARGET_COLUMN], probabilities),
        "Precision": precision_score(data[TARGET_COLUMN], predictions, zero_division=0),
        "Recall": recall_score(data[TARGET_COLUMN], predictions, zero_division=0),
        "F1": f1_score(data[TARGET_COLUMN], predictions, zero_division=0),
        "MCC": matthews_corrcoef(data[TARGET_COLUMN], predictions),
    }
    st.subheader(f"{model_name} evaluation")
    metric_columns = st.columns(6)
    for column, (label, value) in zip(metric_columns, values.items()):
        column.metric(label, f"{value:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        matrix = confusion_matrix(data[TARGET_COLUMN], predictions)
        figure, axis = plt.subplots(figsize=(4.8, 3.8))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="crest", cbar=False, ax=axis)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Actual label")
        st.pyplot(figure, clear_figure=True)
    with right:
        st.subheader("Classification report")
        report = classification_report(
            data[TARGET_COLUMN], predictions, target_names=["malignant", "benign"], output_dict=True
        )
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)
else:
    st.warning("No target column was found. Predictions are available, but evaluation metrics require the true target values.")

st.subheader("Predictions preview")
preview = data[expected_features].copy()
preview["prediction"] = predictions
preview["benign_probability"] = probabilities.round(4)
st.dataframe(preview.head(10), use_container_width=True)

st.subheader("Held-out benchmark")
benchmark_path = MODEL_DIR / "metrics.csv"
if benchmark_path.exists():
    benchmark = pd.read_csv(benchmark_path, index_col=0)
    st.dataframe(benchmark.round(4), use_container_width=True)
