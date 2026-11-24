"""
config.py
Centralized configuration file for the Telco Customer Churn Prediction project.
Stores file paths, random seeds, and target column definitions to avoid hardcoding.
"""

from pathlib import Path

# 1. Define the root directory dynamically
# Path(__file__) gets the path of config.py -> .parent is 'src/' -> .parent.parent is project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# 2. Define Data Directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 3. Define File Paths
# Note: Kaggle IBM Telco dataset often comes as an .xlsx file
RAW_DATA_FILE = RAW_DATA_DIR / "Telco_customer_churn.xlsx"
TRAIN_PARQUET_FILE = PROCESSED_DATA_DIR / "train.parquet"
TEST_PARQUET_FILE = PROCESSED_DATA_DIR / "test.parquet"

# 4. Define Model and Artifacts Directories
MODELS_DIR = ROOT_DIR / "models"
MODEL_PIPELINE_PATH = MODELS_DIR / "churn_pipeline.joblib"

# 5. Define Modeling Constants
RANDOM_STATE = 42          # Ensures reproducibility across all splits and models
TEST_SIZE = 0.20           # 80% Train, 20% Test split
TARGET_COL = "churn_value" # Standardized snake_case name for the binary target (0 or 1)

# Create directories automatically if they do not exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

