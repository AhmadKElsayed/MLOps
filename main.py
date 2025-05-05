import hydra
from omegaconf import DictConfig
from training.logger import ExecutorLogger
from src.training.evaluate import evaluate
from src.training.process_data import read_process_data
from src.training.train import encode_target_col, train_all_models

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training pipeline using Hydra config
    """
    logger = ExecutorLogger("training")
    logger.info("Starting training pipeline")
    
    # 1. Data processing
    logger.info("Processing raw data")
    read_process_data(
        cfg=cfg.pipeline,
        logger=logger
    )
    
    # 2. Target encoding and train/test split
    logger.info("Encoding target variable")
    X_train, y_train, X_test, y_test = encode_target_col(
        cfg=cfg.pipeline,
        logger=logger
    )
    
    # 3. Train models
    logger.info("Training models")
    train_all_models(
        X=X_train,
        y=y_train,
        cfg=cfg.pipeline,
        logger=logger
    )
    
    # 4. Evaluate models
    logger.info("Evaluating models")
    for model_type in ["xgboost", "random_forest"]:
        evaluate(
            X_test=X_test,
            y_test=y_test,
            cfg=cfg.pipeline,
            logger=logger
        )
    
    logger.info("Training pipeline completed successfully")

if __name__ == "__main__":
    main()