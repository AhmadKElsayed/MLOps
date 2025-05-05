import os
import pickle
import logging
from functools import partial
from typing import Any, Dict

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope

import dvc.api
from logger import ExecutorLogger

logging.getLogger('hyperopt.tpe').setLevel(logging.WARNING)

# Hyperopt search spaces
SPACES = {
    "xgboost": {
        "n_estimators": scope.int(hp.quniform("n_estimators", 50, 200, 1)),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 10, 1)),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
        "random_state": 42,
    },
    "random_forest": {
        "n_estimators": scope.int(hp.quniform("n_estimators", 50, 200, 1)),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 20, 1)),
        "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 10, 1)),
        "random_state": 42,
    }
}

def encode_target_col(cfg: dict, logger: logging.Logger):
    """Handle target encoding using plain config dict"""
    train_df = pd.read_parquet(os.path.join(cfg['data']['processed_data_path'], f"{cfg['data']['file_name']}-train.parquet"))
    test_df = pd.read_parquet(os.path.join(cfg['data']['processed_data_path'], f"{cfg['data']['file_name']}-test.parquet"))

    categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns
    numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.drop(cfg['data']['target_column'], errors='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols.drop(cfg['data']['target_column'], errors='ignore')),
            ('num', SimpleImputer(strategy='median'), numeric_cols)
        ]
    )

    target_encoder = LabelEncoder()

    if train_df[cfg['data']['target_column']].isna().any():
        logger.warning(f"Found {train_df[cfg['data']['target_column']].isna().sum()} missing values in target")
        train_df = train_df.dropna(subset=[cfg['data']['target_column']])

    X_train = preprocessor.fit_transform(train_df.drop(cfg['data']['target_column'], axis=1))
    X_test = preprocessor.transform(test_df.drop(cfg['data']['target_column'], axis=1))

    all_categories = pd.concat([train_df[cfg['data']['target_column']], test_df[cfg['data']['target_column']]]).unique()
    target_encoder.fit(all_categories)
    y_train = target_encoder.transform(train_df[cfg['data']['target_column']])
    y_test = target_encoder.transform(test_df[cfg['data']['target_column']])

    os.makedirs(os.path.join(cfg['model']['model_path'], cfg['model']['model_name']), exist_ok=True)
    with open(os.path.join(cfg['model']['model_path'], cfg['model']['model_name'], "preprocessors.pkl"), "wb") as f:
        pickle.dump({
            'feature_preprocessor': preprocessor,
            'target_encoder': target_encoder
        }, f)

    return X_train, pd.Series(y_train), X_test, pd.Series(y_test)

def objective(model_class, params: Dict[str, Any], X, y, n_folds: int = 5) -> Dict[str, Any]:
    """Objective function for hyperopt"""
    try:
        model = model_class(**params)
        scores = cross_validate(
            model,
            X,
            y,
            cv=n_folds,
            scoring="accuracy",
            error_score='raise'
        )
        accuracy = np.mean(scores["test_score"])
        return {
            "loss": 1 - accuracy,
            "accuracy": accuracy,
            "params": params,
            "status": STATUS_OK
        }
    except Exception as e:
        return {
            "loss": 1.0,
            "accuracy": 0.0,
            "status": STATUS_OK,
            "error": str(e)
        }

def train_model(X, y, cfg: dict, model_class, space: dict, logger: logging.Logger) -> None:
    """Train a single model using plain config dict"""
    trials = Trials()

    best = fmin(
        fn=partial(objective, model_class, X=X, y=y),
        space=space,
        algo=tpe.suggest,
        max_evals=cfg['model']['optimization_params']['max_evals'],
        trials=trials
    )

    int_params = ['n_estimators', 'max_depth', 'min_samples_split']
    for param in int_params:
        if param in best:
            best[param] = int(best[param])

    final_model = model_class(**best)
    final_model.fit(X, y)

    os.makedirs(os.path.join(cfg['model']['model_path'], cfg['model']['model_name']), exist_ok=True)
    pickle.dump(final_model, open(os.path.join(cfg['model']['model_path'], cfg['model']['model_name'], f"{cfg['model']['model_name']}_model.pkl"), "wb"))

def train_all_models(X, y, cfg: dict, logger: logging.Logger) -> None:
    """Train all models using plain config dict"""
    train_model(X, y, cfg, XGBClassifier, SPACES["xgboost"], logger)
    train_model(X, y, cfg, RandomForestClassifier, SPACES["random_forest"], logger)

if __name__ == "__main__":
    cfg = dvc.api.params_show("../../params.yaml")
    logger = ExecutorLogger("training")

    logger.info("Encoding target column and features")
    X_train, y_train, X_test, y_test = encode_target_col(cfg, logger)

    logger.info("Starting model training")
    train_all_models(X_train, y_train, cfg, logger)
