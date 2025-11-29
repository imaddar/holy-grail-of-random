import time
from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier

from decision_tree import DecisionTree


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def load_and_preprocess(path: str) -> DatasetSplit:
    df = pd.read_csv(path).drop(columns=["customer_id"])
    target_col = "loan_status"
    y = df[target_col].to_numpy()

    categorical = [c for c in df.select_dtypes("object").columns if c != target_col]
    numerical = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != target_col
    ]

    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df[categorical])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical),
        index=df.index,
    )

    df_features = df.drop(columns=categorical + [target_col])
    df_features = pd.concat([df_features, encoded_df], axis=1)

    scaler = StandardScaler()
    df_features[numerical] = scaler.fit_transform(df_features[numerical])

    X = df_features.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    feature_names = df_features.columns.tolist()
    return DatasetSplit(X_train, X_test, y_train, y_test, feature_names)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def benchmark_model(
    name: str, model: Any, data: DatasetSplit
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    t0 = time.perf_counter()
    model.fit(data.X_train, data.y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    preds = model.predict(data.X_test)
    infer_time = time.perf_counter() - t1

    metrics = compute_metrics(data.y_test, preds)
    metrics["train_time_sec"] = train_time
    metrics["infer_time_sec"] = infer_time
    metrics["model"] = name

    return metrics, preds, data.y_test


def print_confusion(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n{name} Confusion Matrix\n{cm}")


def main() -> None:
    data = load_and_preprocess("./data/loan_approval.csv")

    custom_tree = DecisionTree(min_samples_split=2000, max_depth=12)
    sklearn_tree = DecisionTreeClassifier(
        max_depth=12, min_samples_split=2000, random_state=42
    )

    results = []

    custom_metrics, custom_pred, y_true = benchmark_model(
        "CustomDecisionTree", custom_tree, data
    )
    results.append(custom_metrics)

    sklearn_metrics, sklearn_pred, _ = benchmark_model(
        "SklearnDecisionTree", sklearn_tree, data
    )
    results.append(sklearn_metrics)

    df_results = pd.DataFrame(results).set_index("model")
    print("\n=== Benchmark Results (test set) ===")
    print(df_results.to_string(float_format=lambda x: f"{x:.4f}"))

    print_confusion("Custom", y_true, custom_pred)
    print_confusion("Sklearn", y_true, sklearn_pred)


if __name__ == "__main__":
    main()
