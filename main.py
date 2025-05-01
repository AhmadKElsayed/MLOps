from src.logger import ExecutorLogger
from src.training.evaluate import evaluate
from src.training.process_data import read_process_data
from src.training.train import encode_target_col, train_all_models

def main(logger: ExecutorLogger) -> None:
    """
    Main training pipeline that:
    1. Processes raw data
    2. Encodes target variables
    3. Trains multiple models
    4. Evaluates all models
    """
    # Configuration
    DATASET_NAME = "Titanic"
    ID_COL = "PassengerId"
    TARGET_COL = "Survived"
    BASE_MODEL_NAME = "Titanic_Classifier"
    
    logger.info("Starting training pipeline")
    
    # 1. Data processing
    logger.info("Processing raw data")
    read_process_data(
        file_name=DATASET_NAME,
        id_col=ID_COL,
        target_col=TARGET_COL,
        logger=logger
    )
    
    # 2. Target encoding and train/test split
    logger.info("Encoding target variable")
    X_train, y_train, X_test, y_test = encode_target_col(
        file_name=DATASET_NAME,
        target_col=TARGET_COL,
        model_name=BASE_MODEL_NAME,
        logger=logger
    )
    
    # 3. Train models
    logger.info("Training models")
    train_all_models(
        X=X_train,
        y=y_train,
        base_model_name=BASE_MODEL_NAME,
        logger=logger
    )
    
    # 4. Evaluate models
    logger.info("Evaluating models")
    for model_type in ["xgboost", "random_forest"]:
        evaluate(
            X_test=X_test,
            y_test=y_test,
            model_name=f"{BASE_MODEL_NAME}_{model_type}",
            logger=logger
        )
    
    logger.info("Training pipeline completed successfully")
    
if __name__ == "__main__":
    logger = ExecutorLogger("training")
    main(logger)