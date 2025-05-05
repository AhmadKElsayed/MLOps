import json
import os
import pickle
import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from time import time
from omegaconf import DictConfig

def evaluate(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: DictConfig,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Evaluation function using Hydra config"""
    # Load model and encoder
    model_path = os.path.join(cfg.model.model_path, cfg.model.model_name, f"{cfg.model.model_name}_model.pkl")
    encoder_path = os.path.join(cfg.model.model_path, cfg.model.model_name, "label_encoder.pkl")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    
    # Encode y_test if not already numeric
    if not pd.api.types.is_numeric_dtype(y_test):
        y_test_encoded = encoder.transform(y_test)
    else:
        y_test_encoded = y_test

    start_time = time()
    y_pred = model.predict(X_test)
    pred_time = time() - start_time

    clf_report = classification_report(
        y_test_encoded,
        y_pred,
        target_names=[str(cls) for cls in encoder.classes_],
        output_dict=True
    )

    formatted_report = {}
    for k, v in clf_report.items():
        if isinstance(v, dict):
            formatted_report[str(k)] = {str(k2): float(v2) for k2, v2 in v.items()}
        else:
            formatted_report[str(k)] = float(v)
    
    metrics = {
        "model_name": cfg.model.model_name,
        "accuracy": float(accuracy_score(y_test_encoded, y_pred)),
        "precision": float(precision_score(y_test_encoded, y_pred, average='weighted')),
        "recall": float(recall_score(y_test_encoded, y_pred, average='weighted')),
        "f1": float(f1_score(y_test_encoded, y_pred, average='weighted')),
        "prediction_time_sec": float(pred_time),
        "classification_report": formatted_report
    }
    
    os.makedirs(os.path.join(cfg.evaluate.reports_path, cfg.model.model_name), exist_ok=True)
    report_path = os.path.join(cfg.evaluate.reports_path, cfg.model.model_name, "evaluation_report.json")
    
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Evaluation complete. Report saved to {report_path}")
    return metrics
