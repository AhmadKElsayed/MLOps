import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Optional, List
from omegaconf import DictConfig

def read_process_data(
    cfg: DictConfig,
    logger: logging.Logger,
    columns_to_drop: Optional[List[str]] = None,
    drop_missing_threshold: Optional[float] = None,
) -> None:
    """Process data using Hydra config"""
    input_path = os.path.join(cfg.data.raw_data_path, f"{cfg.data.file_name}.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")
    
    logger.info(f"Processing {input_path}")
    df = pd.read_csv(input_path)
    
    missing_cols = [col for col in [cfg.data.id_column, cfg.data.target_column] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if columns_to_drop:
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    if drop_missing_threshold is not None:
        missing_ratios = df.isnull().mean()
        cols_to_drop = missing_ratios[missing_ratios > drop_missing_threshold].index.tolist()
        if cols_to_drop:
            logger.info(f"Dropping columns with high missing values: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
    
    df.set_index(cfg.data.id_column, inplace=True)
    train_df, test_df = train_test_split(
        df, 
        test_size=cfg.data.test_size, 
        random_state=cfg.data.random_state, 
        stratify=df[cfg.data.target_column]
    )
    
    os.makedirs(cfg.data.processed_data_path, exist_ok=True)
    train_df.to_parquet(os.path.join(cfg.data.processed_data_path, f"{cfg.data.file_name}-train.parquet"))
    test_df.to_parquet(os.path.join(cfg.data.processed_data_path, f"{cfg.data.file_name}-test.parquet"))
    
    logger.info(f"Saved {len(train_df)} train and {len(test_df)} test samples")