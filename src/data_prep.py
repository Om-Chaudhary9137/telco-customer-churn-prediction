"""
data_prep.py
Handles data ingestion, schema standardization (snake_case), pre-split data hygiene,
and leakage-free stratified train-test splitting saved in Parquet format.
"""

import re
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_FILE,
    TRAIN_PARQUET_FILE,
    TEST_PARQUET_FILE,
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COL,
)

# Configure logging for clear pipeline monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def to_snake_case(col_name: str) -> str:
    """Converts a column name into standardized snake_case."""
    col_name = col_name.strip().lower()
    col_name = re.sub(r"[^a-z0-9]+", "_", col_name)
    return col_name.strip("_")


def load_and_standardize_data(file_path=RAW_DATA_FILE) -> pd.DataFrame:
    """Loads raw Excel data and converts all column names to snake_case."""
    logger.info(f"Loading raw dataset from: {file_path}")
    df = pd.read_excel(file_path)
    df.columns = [to_snake_case(col) for col in df.columns]
    logger.info(f"Data loaded and standardized. Initial shape: {df.shape}")
    return df


def clean_pre_split_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs safe, row-level cleaning that does NOT cause data leakage:
    1. Removes exact duplicate rows.
    2. Drops target-leakage columns, unique IDs, and zero-variance columns.
    3. Fixes data types (converts 'total_charges' from string to float).
    4. Standardizes target column to integer (0 or 1).
    """
    df = df.copy()

    # 1. Remove exact duplicate rows if any exist
    initial_rows = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_rows:
        logger.info(f"Dropped {initial_rows - len(df)} duplicate rows.")

    # 2. Define columns to drop (Leakage + Zero-Variance + Redundant IDs/Text)
    cols_to_drop = [
        "customerid",    # Unique identifier (causes overfitting)
        "count",         # Constant value of 1 (zero variance)
        "country",       # Constant value 'United States' (zero variance)
        "state",         # Constant value 'California' (zero variance)
        "lat_long",      # Redundant string duplicate of latitude & longitude
        "churn_label",   # Exact text duplicate of target ('Yes'/'No')
        "churn_score",   # Post-churn internal score (DATA LEAKAGE)
        "churn_reason",  # Recorded only after customer churns (DATA LEAKAGE)
    ]
    existing_drop_cols = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drop_cols)
    logger.info(f"Dropped {len(existing_drop_cols)} leakage/useless columns: {existing_drop_cols}")

    # 3. Fix 'total_charges' data type
    # Blank spaces (" ") are converted to NaN via errors='coerce'
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    
    # Domain rule: Customers with 0 tenure have not been billed yet, so total_charges = 0.0
    missing_charges_count = df["total_charges"].isnull().sum()
    if missing_charges_count > 0:
        logger.info(f"Filling {missing_charges_count} blank 'total_charges' (tenure=0) with 0.0")
        df["total_charges"] = df["total_charges"].fillna(0.0)

    # 4. Ensure target column is clean integer (0 or 1)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    logger.info(f"Pre-split cleaning complete. Clean shape: {df.shape}")
    return df


def split_and_save_parquet(df: pd.DataFrame) -> None:
    """
    Splits data into stratified train and test sets and saves them as Parquet files.
    Stratification ensures both train and test sets have the exact same churn ratio.
    """
    logger.info(f"Splitting data into Train ({1-TEST_SIZE:.0%}) and Test ({TEST_SIZE:.0%})...")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL],  # Crucial for imbalanced classification!
    )

    # Reset index for clean Parquet storage
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Verify churn proportions match
    train_churn_rate = train_df[TARGET_COL].mean() * 100
    test_churn_rate = test_df[TARGET_COL].mean() * 100
    logger.info(f"Train shape: {train_df.shape} | Churn rate: {train_churn_rate:.2f}%")
    logger.info(f"Test shape:  {test_df.shape} | Churn rate: {test_churn_rate:.2f}%")

    # Save to Parquet using pyarrow engine
    train_df.to_parquet(TRAIN_PARQUET_FILE, index=False, engine="pyarrow")
    test_df.to_parquet(TEST_PARQUET_FILE, index=False, engine="pyarrow")
    logger.info(f"Saved train data to: {TRAIN_PARQUET_FILE}")
    logger.info(f"Saved test data to:  {TEST_PARQUET_FILE}")


def run_data_prep() -> None:
    """Orchestrates the entire data preparation pipeline."""
    raw_df = load_and_standardize_data()
    clean_df = clean_pre_split_data(raw_df)
    split_and_save_parquet(clean_df)


if __name__ == "__main__":
    run_data_prep()
