import json
import os
import pickle
import logging
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from time import time

def evaluate_models(cfg: dict, logger: logging.Logger):
    """Evaluate all trained models using saved preprocessors and test data"""

    # Load test data
    test_path = os.path.join(cfg['data']['processed_data_path'], f"{cfg['data']['file_name']}-test.parquet")
    test_df = pd.read_parquet(test_path)

    # Load preprocessors (feature preprocessor & target encoder)
    preprocessors_path = os.path.join(cfg['model']['model_path'], cfg['model']['model_name'], "preprocessors.pkl")
    with open(preprocessors_path, "rb") as f:
        preprocessors = pickle.load(f)

    feature_preprocessor = preprocessors['feature_preprocessor']
    target_encoder = preprocessors['target_encoder']

    # Transform test features and encode labels
    X_test = feature_preprocessor.transform(test_df.drop(cfg['data']['target_column'], axis=1))
    y_test = test_df[cfg['data']['target_column']]
    y_test_encoded = target_encoder.transform(y_test)

    # Find all model directories starting with the model_name prefix (excluding base model folder)
    model_dirs = [
        d for d in os.listdir(cfg['model']['model_path'])
        if d.startswith(cfg['model']['model_name'] + "_") and os.path.isdir(os.path.join(cfg['model']['model_path'], d))
    ]

    for model_dir in model_dirs:
        model_path = os.path.join(cfg['model']['model_path'], model_dir, f"{model_dir}_model.pkl")

        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            continue

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        start_time = time()
        y_pred = model.predict(X_test)
        pred_time = time() - start_time

        clf_report = classification_report(
            y_test_encoded,
            y_pred,
            target_names=[str(cls) for cls in target_encoder.classes_],
            output_dict=True
        )

        formatted_report = {
            str(k): {str(k2): float(v2) for k2, v2 in v.items()} if isinstance(v, dict) else float(v)
            for k, v in clf_report.items()
        }

        metrics = {
            "model_name": model_dir,
            "accuracy": float(accuracy_score(y_test_encoded, y_pred)),
            "precision": float(precision_score(y_test_encoded, y_pred, average='weighted')),
            "recall": float(recall_score(y_test_encoded, y_pred, average='weighted')),
            "f1": float(f1_score(y_test_encoded, y_pred, average='weighted')),
            "prediction_time_sec": float(pred_time),
            "classification_report": formatted_report
        }

        # Save report
        report_dir = os.path.join(cfg['evaluate']['reports_path'], model_dir)
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "evaluation_report.json")

        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Evaluation complete for {model_dir}. Report saved to {report_path}")

if __name__ == "__main__":
    import dvc.api
    from logger import ExecutorLogger

    cfg = dvc.api.params_show("../../params.yaml")
    logger = ExecutorLogger("evaluation")

    logger.info("Starting evaluation for all models")
    evaluate_models(cfg, logger)
