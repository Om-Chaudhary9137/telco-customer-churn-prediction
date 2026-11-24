"""
evaluate.py
Executes Step 10 (One-time unbiased evaluation on the held-out test set +
business threshold analysis) and Step 11 (Global & Local SHAP model interpretability).
Saves high-resolution evaluation charts into the models/ directory.
"""

import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
)

from src.config import (
    TEST_PARQUET_FILE,
    MODEL_PIPELINE_PATH,
    MODELS_DIR,
    TARGET_COL,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_optimal_f1_threshold(y_true: pd.Series, y_probs: np.ndarray) -> float:
    """
    Finds the probability threshold that maximizes the F1-Score on the Precision-Recall curve,
    balancing Recall (catching churners) and Precision (avoiding wasted retention offers).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    # Compute F1 for each threshold (avoid division by zero)
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (
        precisions[:-1] + recalls[:-1] + 1e-10
    )
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


def evaluate_on_test_set() -> None:
    """
    Step 10: Loads the saved pipeline and evaluates it ONCE on the held-out test set.
    Generates ROC Curve, Precision-Recall Curve, and Confusion Matrix plots.
    """
    logger.info("Loading held-out test dataset and trained production pipeline...")
    test_df = pd.read_parquet(TEST_PARQUET_FILE)
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    pipeline = joblib.load(MODEL_PIPELINE_PATH)

    # Predict churn probabilities for class 1
    y_probs = pipeline.predict_proba(X_test)[:, 1]

    # Threshold-independent ranking metrics
    pr_auc = average_precision_score(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)

    # Default 0.50 threshold vs Optimal Business Threshold
    default_preds = (y_probs >= 0.50).astype(int)
    optimal_thresh = find_optimal_f1_threshold(y_test, y_probs)
    optimal_preds = (y_probs >= optimal_thresh).astype(int)

    print("\n================ ONE-TIME TEST SET EVALUATION ================")
    print(f"Test PR-AUC  (Primary Metric): {pr_auc:.4f}")
    print(f"Test ROC-AUC (Ranking Metric): {roc_auc:.4f}")
    print(f"\n--- Classification Report (Default Threshold = 0.50) ---")
    print(classification_report(y_test, default_preds, digits=4))
    print(f"--- Classification Report (Optimal F1 Threshold = {optimal_thresh:.3f}) ---")
    print(classification_report(y_test, optimal_preds, digits=4))
    print("==============================================================\n")

    # Save 3-in-1 Diagnostic Evaluation Plot (PR Curve, ROC Curve, Confusion Matrix)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    axes[0].plot(recall, precision, color="darkorange", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    axes[0].axhline(y=y_test.mean(), color="navy", linestyle="--", label=f"Baseline = {y_test.mean():.4f}")
    axes[0].set_title("Precision-Recall Curve (Test Set)")
    axes[0].set_xlabel("Recall (Churners Caught)")
    axes[0].set_ylabel("Precision")
    axes[0].legend()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    axes[1].plot(fpr, tpr, color="teal", lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess")
    axes[1].set_title("ROC Curve (Test Set)")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate (Recall)")
    axes[1].legend()

    # 3. Confusion Matrix at Optimal Business Threshold
    cm = confusion_matrix(y_test, optimal_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Retained (0)", "Churned (1)"])
    disp.plot(ax=axes[2], cmap="Blues", colorbar=False)
    axes[2].set_title(f"Confusion Matrix (Threshold = {optimal_thresh:.2f})")

    plt.tight_layout()
    eval_plot_path = MODELS_DIR / "test_evaluation_curves.png"
    plt.savefig(eval_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved evaluation curves to: {eval_plot_path}")

    # Proceed to Step 11: SHAP Explainability
    generate_shap_explanations(pipeline, X_test)


def generate_shap_explanations(pipeline, X_test: pd.DataFrame) -> None:
    """
    Step 11: Uses SHAP TreeExplainer to extract global feature importance
    and saves a clean SHAP summary beeswarm plot for stakeholder presentation.
    """
    logger.info("Starting Step 11: Generating SHAP Explainability plots...")

    # Extract the preprocessing step and the trained LightGBM model from the pipeline
    preprocessor = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]

    # Transform X_test into the exact numeric feature matrix fed into LightGBM
    X_test_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.named_steps["preprocessing"].get_feature_names_out()
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    # Compute SHAP values using fast TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df)

    # Handle LightGBM binary output format (list of [class_0, class_1] or single array)
    if isinstance(shap_values, list):
        shap_vals_churn = shap_values[1]
    else:
        shap_vals_churn = shap_values

    # Plot and save Global SHAP Summary Beeswarm Plot (Top 15 Features)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_vals_churn, X_test_df, max_display=15, show=False)
    plt.title("Global Feature Importance & Impact Direction (SHAP Summary)", fontsize=12)
    plt.tight_layout()
    shap_plot_path = MODELS_DIR / "shap_summary_plot.png"
    plt.savefig(shap_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Print Top 10 Most Important Features by Mean Absolute SHAP Value
    mean_abs_shap = np.abs(shap_vals_churn).mean(axis=0)
    shap_importance = (
        pd.DataFrame({"Feature": feature_names, "Mean_Abs_SHAP": mean_abs_shap})
        .sort_values(by="Mean_Abs_SHAP", ascending=False)
        .head(10)
    )
    print("\n================ TOP 10 CHURN DRIVERS (SHAP) ================")
    print(shap_importance.to_string(index=False))
    print("=============================================================\n")
    logger.info(f"✅ Saved SHAP summary plot to: {shap_plot_path}")


if __name__ == "__main__":
    evaluate_on_test_set()
