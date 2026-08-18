"""Train five classifiers and create the held-out CSV used by the app."""

from pathlib import Path
import json
import pickle

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[0]
MODEL_DIR = ROOT / "model" / "artifacts"
TEST_DATA_PATH = ROOT / "test_data.csv"
RANDOM_STATE = 42
TARGET_COLUMN = "target"


def build_models():
    """Return the five models required by the assignment."""
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE
        ),
        "KNN": Pipeline([
            ("scale", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def calculate_metrics(model, features, target):
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": roc_auc_score(target, probabilities),
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1": f1_score(target, predictions, zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
    }


def main():
    dataset = load_breast_cancer(as_frame=True)
    data = dataset.frame.rename(columns={"target": TARGET_COLUMN})
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        stratify=data[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    train_features = train_data.drop(columns=TARGET_COLUMN)
    train_target = train_data[TARGET_COLUMN]
    test_features = test_data.drop(columns=TARGET_COLUMN)
    test_target = test_data[TARGET_COLUMN]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trained_models = {}
    results = {}
    for name, model in build_models().items():
        model.fit(train_features, train_target)
        trained_models[name] = model
        results[name] = calculate_metrics(model, test_features, test_target)
        with (MODEL_DIR / f"{name.lower().replace(' ', '_')}.pkl").open("wb") as artifact:
            pickle.dump(model, artifact)

    test_data.to_csv(TEST_DATA_PATH, index=False)
    metadata = {
        "target_column": TARGET_COLUMN,
        "feature_columns": list(train_features.columns),
        "class_names": {"0": "malignant", "1": "benign"},
        "random_state": RANDOM_STATE,
        "train_rows": len(train_data),
        "test_rows": len(test_data),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    pd.DataFrame(results).T.to_csv(MODEL_DIR / "metrics.csv", index=True)

    print(f"Trained {len(trained_models)} models on {len(train_data)} rows.")
    print(f"Created {TEST_DATA_PATH.name} with {len(test_data)} rows.")
    print(pd.DataFrame(results).T.round(4).to_string())


if __name__ == "__main__":
    main()
