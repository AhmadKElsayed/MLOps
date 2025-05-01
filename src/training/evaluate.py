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

MODEL_PATH = "models"
REPORT_PATH = "reports"

def evaluate(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Updated to properly handle classification report structure
    while maintaining template paths and functionality
    """
    # Load model and encoder
    model_path = os.path.join(MODEL_PATH, model_name, f"{model_name}_model.pkl")
    encoder_path = os.path.join(MODEL_PATH, "Titanic_Classifier", "label_encoder.pkl")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    
    # Handle target encoding
    if not pd.api.types.is_numeric_dtype(y_test):
        y_test_encoded = y_test.map(lambda x: encoder.get(x, -1))  # -1 for unknown labels
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
    
    # Final metrics
    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test_encoded, y_pred)),
        "precision": float(precision_score(y_test_encoded, y_pred, average='weighted')),
        "recall": float(recall_score(y_test_encoded, y_pred, average='weighted')),
        "f1": float(f1_score(y_test_encoded, y_pred, average='weighted')),
        "prediction_time_sec": float(pred_time),
        "classification_report": formatted_report
    }
    
    # Save report (preserving template path structure)
    os.makedirs(os.path.join(REPORT_PATH, model_name), exist_ok=True)
    report_path = os.path.join(REPORT_PATH, model_name, "evaluation_report.json")
    
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Evaluation complete. Report saved to {report_path}")
    return metrics