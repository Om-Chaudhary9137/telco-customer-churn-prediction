"""
train.py
Executes Step 7 (Baseline Models), Step 8 (5-Fold Stratified CV Model Selection),
and Step 9 (Optuna Hyperparameter Tuning) strictly on the training set,
then exports the final end-to-end pipeline to disk.
"""

import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import optuna

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.config import (
    TRAIN_PARQUET_FILE,
    MODEL_PIPELINE_PATH,
    RANDOM_STATE,
    TARGET_COL,
)
from src.features import create_full_preprocessing_pipeline

# Suppress noisy warnings during CV and Optuna trials
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_training_data():
    """Loads the leakage-free training Parquet file and separates X and y."""
    df = pd.read_parquet(TRAIN_PARQUET_FILE)
    X_train = df.drop(columns=[TARGET_COL])
    y_train = df[TARGET_COL]
    return X_train, y_train


def benchmark_models(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """
    Step 7 & Step 8: Evaluates a Dummy baseline, Logistic Regression baseline,
    and multiple tree-based algorithms using 5-Fold Stratified Cross-Validation.
    """
    logger.info("Starting Step 7 & 8: Baseline & Model Selection via 5-Fold Stratified CV...")

    # Calculate class imbalance ratio for gradient boosting models
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    models = {
        "0_Dummy_Baseline": DummyClassifier(strategy="most_frequent"),
        "1_Logistic_Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "2_Random_Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "3_XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "4_LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring_metrics = {
        "pr_auc": "average_precision",  # Primary metric for imbalanced churn
        "roc_auc": "roc_auc",
        "recall": "recall",
        "f1": "f1",
    }

    results = []
    for name, clf in models.items():
        # Build a fresh end-to-end pipeline so preprocessing fits inside each CV fold!
        preprocessor = create_full_preprocessing_pipeline(X_train)
        full_pipeline = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", clf),
            ]
        )

        cv_scores = cross_validate(
            full_pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring_metrics,
            n_jobs=-1,
        )

        results.append(
            {
                "Model": name,
                "PR_AUC (Mean)": round(cv_scores["test_pr_auc"].mean(), 4),
                "PR_AUC (Std)": round(cv_scores["test_pr_auc"].std(), 4),
                "ROC_AUC (Mean)": round(cv_scores["test_roc_auc"].mean(), 4),
                "Recall (Mean)": round(cv_scores["test_recall"].mean(), 4),
                "F1 (Mean)": round(cv_scores["test_f1"].mean(), 4),
            }
        )
        logger.info(
            f"{name} -> PR-AUC: {results[-1]['PR_AUC (Mean)']:.4f} | "
            f"ROC-AUC: {results[-1]['ROC_AUC (Mean)']:.4f} | "
            f"Recall: {results[-1]['Recall (Mean)']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(by="PR_AUC (Mean)", ascending=False)
    print("\n================ CROSS-VALIDATION BENCHMARK RESULTS ================")
    print(results_df.to_string(index=False))
    print("====================================================================\n")
    return results_df


def tune_lightgbm_with_optuna(
    X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 30
) -> dict:
    """
    Step 9: Uses Optuna (Bayesian Optimization) to tune LightGBM hyperparameters
    by maximizing 5-Fold Stratified CV PR-AUC.
    """
    logger.info(f"Starting Step 9: Hyperparameter Tuning with Optuna ({n_trials} trials)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "verbose": -1,
            "n_jobs": -1,
        }

        preprocessor = create_full_preprocessing_pipeline(X_train)
        pipeline = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", LGBMClassifier(**params)),
            ]
        )

        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1
        )
        return scores["test_score"].mean()

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Best CV PR-AUC after Optuna tuning: {study.best_value:.4f}")
    logger.info(f"Best Hyperparameters: {study.best_params}")
    return study.best_params


def train_and_save_final_model() -> None:
    """
    Runs the full training workflow and saves the winning tuned pipeline to disk.
    """
    X_train, y_train = load_training_data()

    # Run Step 7 & 8 benchmarking
    benchmark_models(X_train, y_train)

    # Run Step 9 Optuna tuning for LightGBM
    best_params = tune_lightgbm_with_optuna(X_train, y_train, n_trials=30)

    # Fit the final pipeline on the ENTIRE training dataset using best hyperparameters
    logger.info("Fitting final tuned LightGBM pipeline on 100% of training data...")
    final_preprocessor = create_full_preprocessing_pipeline(X_train)
    final_model = LGBMClassifier(
        **best_params,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1,
    )

    final_pipeline = Pipeline(
        steps=[
            ("prep", final_preprocessor),
            ("model", final_model),
        ]
    )
    final_pipeline.fit(X_train, y_train)

    # Export the complete pipeline (Feature Engineering + Scaling/OHE + Model)
    joblib.dump(final_pipeline, MODEL_PIPELINE_PATH)
    logger.info(f"✅ Final production pipeline saved successfully to: {MODEL_PIPELINE_PATH}")


if __name__ == "__main__":
    train_and_save_final_model()
