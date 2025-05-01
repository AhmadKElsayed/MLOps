from functools import partial
import os
import pickle
import logging
from typing import Any, Dict
import pandas as pd
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

SOURCE = os.path.join("data", "processed")
MODEL_PATH = "models"

N_FOLDS = 5
MAX_EVALS = 10

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

def encode_target_col(file_name: str, target_col: str, model_name: str, logger):
    """Handle missing values and categorical features"""
    train_df = pd.read_parquet(os.path.join(SOURCE, f"{file_name}-train.parquet"))
    test_df = pd.read_parquet(os.path.join(SOURCE, f"{file_name}-test.parquet"))
    
    # Identify columns
    categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns
    numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.drop(target_col, errors='ignore')
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols.drop(target_col, errors='ignore')),
            ('num', SimpleImputer(strategy='median'), numeric_cols)
        ])
    
    target_encoder = LabelEncoder()
    
    # Handle missing values in target
    if train_df[target_col].isna().any():
        logger.warning(f"Found {train_df[target_col].isna().sum()} missing values in target")
        train_df = train_df.dropna(subset=[target_col])
    
    # Fit and transform
    X_train = preprocessor.fit_transform(train_df.drop(target_col, axis=1))
    X_test = preprocessor.transform(test_df.drop(target_col, axis=1))
    
    # Encode target
    all_categories = pd.concat([train_df[target_col], test_df[target_col]]).unique()
    target_encoder.fit(all_categories)
    y_train = target_encoder.transform(train_df[target_col])
    y_test = target_encoder.transform(test_df[target_col])
    
    # Save preprocessors
    os.makedirs(os.path.join(MODEL_PATH, model_name), exist_ok=True)
    with open(os.path.join(MODEL_PATH, model_name, "preprocessors.pkl"), "wb") as f:
        pickle.dump({
            'feature_preprocessor': preprocessor,
            'target_encoder': target_encoder
        }, f)
    
    return X_train, pd.Series(y_train), X_test, pd.Series(y_test)

def objective(model_class, params: Dict[str, Any], X, y, n_folds: int = N_FOLDS) -> Dict[str, Any]:
    """Modified to return positive loss values"""
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

def train_model(X, y, model_name: str, model_class, space: Dict, logger) -> None:
    """Train with missing value handling"""
    trials = Trials()
    
    best = fmin(
        fn=partial(objective, model_class, X=X, y=y),
        space=space,
        algo=tpe.suggest,
        max_evals=MAX_EVALS,
        trials=trials
    )
    
    # Convert parameters to proper types
    int_params = ['n_estimators', 'max_depth', 'min_samples_split']
    for param in int_params:
        if param in best:
            best[param] = int(best[param])
    
    # Train final model
    final_model = model_class(**best)
    final_model.fit(X, y)
    
    # Save model
    os.makedirs(os.path.join(MODEL_PATH, model_name), exist_ok=True)
    pickle.dump(final_model, open(os.path.join(MODEL_PATH, model_name, f"{model_name}_model.pkl"), "wb"))

def train_all_models(X, y, base_model_name: str, logger) -> None:
    """Maintain template functionality"""
    train_model(X, y, f"{base_model_name}_xgboost", XGBClassifier, SPACES["xgboost"], logger)
    train_model(X, y, f"{base_model_name}_random_forest", RandomForestClassifier, SPACES["random_forest"], logger)