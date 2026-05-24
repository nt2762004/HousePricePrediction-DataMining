"""Model helpers for training and evaluation."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb



def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))



def eval_on_logscale(pred_log, y_log_true):
    rmse_log = np.sqrt(mean_squared_error(y_log_true, pred_log))
    pred = np.expm1(pred_log)
    true = np.expm1(y_log_true)
    return {
        "rmse_log": rmse_log,
        "rmse": rmse(true, pred),
        "r2_log": r2_score(y_log_true, pred_log),
    }



def build_preprocessor(num_cols, cat_cols):
    num_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_tf, num_cols),
            ("cat", cat_tf, cat_cols),
        ],
        verbose_feature_names_out=False,
    )



def train_ridge_model(X_train, y_train, X_valid, y_valid, preprocessor, alpha_grid=None, random_state=42):
    if alpha_grid is None:
        alpha_grid = [0.0005, 0.001, 0.005, 0.01, 0.05]

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", Ridge(random_state=random_state, max_iter=10000)),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {"model__alpha": alpha_grid},
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    pred_log = search.predict(X_valid)
    metrics = eval_on_logscale(pred_log, y_valid)
    return {
        "name": "Ridge",
        "estimator": search.best_estimator_,
        "metrics": metrics,
        "pred_log": pred_log,
    }



def train_random_forest_model(X_train, y_train, X_valid, y_valid, preprocessor, random_state=42):
    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=700,
                    random_state=random_state,
                    n_jobs=-1,
                    max_features=0.8,
                    min_samples_leaf=1,
                    min_samples_split=2,
                    bootstrap=True,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    pred_log = pipeline.predict(X_valid)
    metrics = eval_on_logscale(pred_log, y_valid)
    return {
        "name": "RandomForest",
        "estimator": pipeline,
        "metrics": metrics,
        "pred_log": pred_log,
    }



def train_xgboost_model(X_train, y_train, X_valid, y_valid, preprocessor, random_state=42, n_estimators=2000):
    preprocessor_fit = preprocessor.fit(X_train)
    X_train_p = preprocessor_fit.transform(X_train)
    X_valid_p = preprocessor_fit.transform(X_valid)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="rmse",
    )
    model.fit(
        X_train_p,
        y_train,
        eval_set=[(X_train_p, y_train), (X_valid_p, y_valid)],
        verbose=50,
    )
    pred_log = model.predict(X_valid_p)
    metrics = eval_on_logscale(pred_log, y_valid)
    evals_result = model.evals_result()
    return {
        "name": "XGBoost",
        "estimator": model,
        "preprocessor": preprocessor_fit,
        "train_matrix": X_train_p,
        "valid_matrix": X_valid_p,
        "metrics": metrics,
        "pred_log": pred_log,
        "evals_result": evals_result,
    }



def build_feature_names(preprocessor, num_cols, cat_cols):
    cat_transformer = preprocessor.named_transformers_["cat"]
    if hasattr(cat_transformer, "named_steps"):
        cat_names = cat_transformer.named_steps["onehot"].get_feature_names_out(cat_cols)
    else:
        cat_names = cat_transformer.get_feature_names_out(cat_cols)
    return list(num_cols) + list(cat_names)



def get_permutation_importance(model, X_valid, y_valid, feature_names, random_state=42, n_repeats=5):
    result = permutation_importance(
        model,
        X_valid,
        y_valid,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    import pandas as pd

    return pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)
