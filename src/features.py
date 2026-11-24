"""
Defines custom scikit-learn transformers for domain-driven feature engineering
and builds a leakage-free ColumnTransformer preprocessing pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer that creates domain-specific Telecom features
    and safely encodes high-cardinality location features without data leakage.
    """

    def __init__(self):
        # Learned statistics from the training set during .fit()
        self.city_freq_map_ = {}
        self.addon_cols_ = [
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
            "streaming_tv",
            "streaming_movies",
        ]

    def fit(self, X: pd.DataFrame, y=None):
        """
        Learns frequency encoding maps strictly from the training data.
        """
        X_copy = X.copy()
        if "city" in X_copy.columns:
            # Calculate normalized frequency of each city on TRAIN data only
            self.city_freq_map_ = X_copy["city"].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies deterministic domain feature engineering and learned encodings.
        """
        df = X.copy()

        # 1. Financial Ratio Features (Bill Shock Detection)
        # Adding 1 to tenure_months prevents division by zero for brand-new customers
        df["avg_monthly_spend"] = df["total_charges"] / (df["tenure_months"] + 1.0)
        df["charge_diff_from_avg"] = df["monthly_charges"] - df["avg_monthly_spend"]

        # 2. Service Stickiness / Ecosystem Lock-in Score
        # Count how many add-on services are set to 'Yes'
        existing_addons = [col for col in self.addon_cols_ if col in df.columns]
        df["total_addon_services"] = (df[existing_addons] == "Yes").sum(axis=1)

        # 3. High-Risk Behavioral Flags
        if "contract" in df.columns:
            df["is_month_to_month"] = (df["contract"] == "Month-to-month").astype(int)
        if "payment_method" in df.columns:
            df["has_electronic_check"] = (df["payment_method"] == "Electronic check").astype(int)

        # 4. Customer Lifecycle Stage (Tenure Buckets)
        df["tenure_group"] = pd.cut(
            df["tenure_months"],
            bins=[-1, 12, 24, 48, np.inf],
            labels=["0_12m", "13_24m", "25_48m", "49m_plus"]
        ).astype(str)

        # 5. High-Cardinality Geographic Handling
        # Map city frequency learned during .fit(); unseen cities default to 0.0
        if "city" in df.columns:
            df["city_frequency"] = df["city"].map(self.city_freq_map_).fillna(0.0)

        # Drop raw high-cardinality text columns (we already have latitude, longitude & city_frequency)
        drop_high_card = [col for col in ["city", "zip_code"] if col in df.columns]
        df = df.drop(columns=drop_high_card)

        return df


def build_preprocessor(X_sample: pd.DataFrame) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that scales numeric features and one-hot encodes
    categorical features cleanly.
    """
    # Run a temporary pass of TelcoFeatureEngineer to identify resulting column types
    engineer = TelcoFeatureEngineer()
    X_engineered = engineer.fit_transform(X_sample)

    numeric_cols = X_engineered.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_engineered.select_dtypes(exclude=[np.number]).columns.tolist()

    # Numeric Pipeline: Median imputation (safety net for production) + Standard Scaling
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical Pipeline: Most-frequent imputation + One-Hot Encoding
    # handle_unknown='ignore' prevents crashes if an unseen category appears in production
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,  # Keeps feature names clean for SHAP & inspection
    )

    return preprocessor


def create_full_preprocessing_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """
    Combines custom feature engineering and column preprocessing into a single Scikit-Learn Pipeline.
    """
    preprocessor = build_preprocessor(X_train)
    full_pipeline = Pipeline(
        steps=[
            ("feature_engineering", TelcoFeatureEngineer()),
            ("preprocessing", preprocessor),
        ]
    )
    return full_pipeline


if __name__ == "__main__":
    # Quick sanity check using the saved train.parquet file
    from src.config import TRAIN_PARQUET_FILE, TARGET_COL

    train_df = pd.read_parquet(TRAIN_PARQUET_FILE)
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    pipeline = create_full_preprocessing_pipeline(X_train)
    X_transformed = pipeline.fit_transform(X_train, y_train)
    feature_names = pipeline.named_steps["preprocessing"].get_feature_names_out()

    print(f"Original feature count:    {X_train.shape[1]}")
    print(f"Transformed feature count: {X_transformed.shape[1]}")
    print(f"Sample transformed features: {feature_names[:10].tolist()}")
