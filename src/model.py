"""Cross-validated churn modeling, evaluation, and model artifact persistence."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from xgboost import XGBClassifier


DEFAULT_PARAMETER_GRID: dict[str, list[Any]] = {
    "n_estimators": [150, 300],
    "learning_rate": [0.02, 0.05, 0.1],
    "num_leaves": [7, 15, 31],
    "max_depth": [-1, 5],
    "min_child_samples": [10, 30],
    "reg_lambda": [0.0, 2.0],
}


@dataclass
class TrainingResult:
    """Models and diagnostics produced without using the held-out test split."""

    feature_columns: list[str]
    cross_validation_results: pd.DataFrame
    baseline_model: LGBMClassifier
    tuned_model: LGBMClassifier
    selected_model: LGBMClassifier
    selected_model_name: str
    baseline_cv_pr_auc: float
    tuned_cv_pr_auc: float
    best_parameters: dict[str, Any]
    X_test: pd.DataFrame
    y_test: pd.Series


def train_and_tune_model(
    frame: pd.DataFrame,
    *,
    parameter_grid: dict[str, list[Any]] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    folds: int = 5,
    n_jobs: int = -1,
) -> TrainingResult:
    """Compare baseline classifiers and tune LightGBM using training folds only."""
    required_columns = {"UserID", "churned"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Prepared data is missing required columns: {sorted(missing)}")

    excluded = {
        "UserID",
        "churned",
        "days_since_last_bet",
        "days_since_last_deposit",
    }
    feature_columns = [column for column in frame.columns if column not in excluded]
    if not feature_columns:
        raise ValueError("No predictor columns remain after excluding identifiers and leakage fields.")

    X = frame[feature_columns]
    y = frame["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    class_counts = y_train.value_counts()
    if len(class_counts) != 2 or (class_counts == 0).any():
        raise ValueError("Training data must contain both churn outcome classes.")
    positive_class_weight = class_counts[0] / class_counts[1]

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=random_state,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            scale_pos_weight=positive_class_weight,
            random_state=random_state,
            verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            scale_pos_weight=positive_class_weight,
            random_state=random_state,
            eval_metric="logloss",
        ),
    }
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    scoring = ["roc_auc", "average_precision", "f1", "recall", "precision"]
    cross_validation_results = []
    for name, estimator in models.items():
        scores = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        )
        cross_validation_results.append(
            {
                "Model": name,
                "CV ROC-AUC": float(np.mean(scores["test_roc_auc"])),
                "CV PR-AUC": float(np.mean(scores["test_average_precision"])),
                "CV F1-Score": float(np.mean(scores["test_f1"])),
                "CV Recall": float(np.mean(scores["test_recall"])),
                "CV Precision": float(np.mean(scores["test_precision"])),
            }
        )
    cv_frame = pd.DataFrame(cross_validation_results)

    baseline_model = models["LightGBM"]
    baseline_model.fit(X_train, y_train)
    baseline_cv_pr_auc = float(
        cv_frame.loc[cv_frame["Model"] == "LightGBM", "CV PR-AUC"].iloc[0]
    )

    search = GridSearchCV(
        estimator=LGBMClassifier(
            scale_pos_weight=positive_class_weight,
            random_state=random_state,
            verbose=-1,
            n_jobs=1,
        ),
        param_grid=parameter_grid or DEFAULT_PARAMETER_GRID,
        scoring="average_precision",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    tuned_model = search.best_estimator_
    tuned_cv_pr_auc = float(search.best_score_)

    if tuned_cv_pr_auc >= baseline_cv_pr_auc:
        selected_model = tuned_model
        selected_model_name = "Tuned LightGBM"
    else:
        selected_model = baseline_model
        selected_model_name = "Baseline LightGBM"

    return TrainingResult(
        feature_columns=feature_columns,
        cross_validation_results=cv_frame,
        baseline_model=baseline_model,
        tuned_model=tuned_model,
        selected_model=selected_model,
        selected_model_name=selected_model_name,
        baseline_cv_pr_auc=baseline_cv_pr_auc,
        tuned_cv_pr_auc=tuned_cv_pr_auc,
        best_parameters=dict(search.best_params_),
        X_test=X_test,
        y_test=y_test,
    )


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate one already-selected estimator on a held-out test split."""
    if not 0 <= threshold <= 1:
        raise ValueError("Classification threshold must be between 0 and 1.")
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["Retained (0)", "Churned (1)"],
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
    }


def save_model_payload(
    output_path: str | Path,
    model: Any,
    feature_columns: list[str],
    optimal_threshold: float,
) -> Path:
    """Persist the estimator and its inference metadata as a joblib payload."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": feature_columns,
            "optimal_threshold": float(optimal_threshold),
        },
        destination,
    )
    return destination
