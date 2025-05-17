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
    """Evaluate all trained pipelines using test data"""

    # Load test data
    test_path = os.path.join(cfg['data']['processed_data_path'], f"{cfg['data']['file_name']}-test.parquet")
    test_df = pd.read_parquet(test_path)
    
    # Separate features and target
    X_test = test_df.drop(cfg['data']['target_column'], axis=1)
    y_test = test_df[cfg['data']['target_column']]
    
    # Load target encoder to convert predictions back to original labels if needed
    with open(os.path.join(cfg['model']['model_path'], cfg['model']['model_name'], "label_encoder.pkl"), "rb") as f:
        target_encoder = pickle.load(f)
    
    # Encode test labels for metric calculation
    y_test_encoded = target_encoder.transform(y_test)

    # Find all model directories starting with the model_name prefix (excluding base model folder)
    model_dirs = [
        d for d in os.listdir(cfg['model']['model_path'])
        if d.startswith(cfg['model']['model_name'] + "_") and os.path.isdir(os.path.join(cfg['model']['model_path'], d))
    ]

    for model_dir in model_dirs:
        # Use the pipeline instead of the model
        pipeline_path = os.path.join(cfg['model']['model_path'], model_dir, f"{model_dir}_pipeline.pkl")

        if not os.path.exists(pipeline_path):
            logger.warning(f"Pipeline file not found: {pipeline_path}")
            continue

        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)

        logger.info(f"Evaluating pipeline: {model_dir}")
        
        # Measure prediction time
        start_time = time()
        # Use the pipeline directly on the raw test data - it handles preprocessing
        y_pred = pipeline.predict(X_test)
        pred_time = time() - start_time

        # Generate metrics
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
    setup_dagshub(cfg = cfg)
    logger.info("Starting evaluation for all pipelines")
    evaluate_models(cfg, logger)