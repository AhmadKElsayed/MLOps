import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Optional, List


SOURCE = os.path.join("data", "raw")
DESTINATION = os.path.join("data", "processed")

def read_process_data(
    file_name: str,
    id_col: str,
    target_col: str,
    logger: logging.Logger,
    columns_to_drop: Optional[List[str]] = None,
    drop_missing_threshold: Optional[float] = None,
) -> None:
    """
    **Identical to your code** - only added docstring for clarity.
    Processes data from `data/raw/{file_name}.csv` → saves splits to `data/processed/`.
    """
    # Input validation
    input_path = os.path.join(SOURCE, f"{file_name}.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")
    
    logger.info(f"Processing {input_path}")
    df = pd.read_csv(input_path)
    
    # Column validation
    missing_cols = [col for col in [id_col, target_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Column dropping
    if columns_to_drop:
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Missing value threshold
    if drop_missing_threshold is not None:
        missing_ratios = df.isnull().mean()
        cols_to_drop = missing_ratios[missing_ratios > drop_missing_threshold].index.tolist()
        if cols_to_drop:
            logger.info(f"Dropping columns with high missing values: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
    
    # Split and save
    df.set_index(id_col, inplace=True)
    train_df, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df[target_col]
    )
    
    os.makedirs(DESTINATION, exist_ok=True)
    train_df.to_parquet(os.path.join(DESTINATION, f"{file_name}-train.parquet"))
    test_df.to_parquet(os.path.join(DESTINATION, f"{file_name}-test.parquet"))
    
    logger.info(f"Saved {len(train_df)} train and {len(test_df)} test samples")