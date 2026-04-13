
#--- XGBoost surrogate — one XGBRegressor trained per output target,
#    with early stopping on the validation set.---#
 
#--- Training one model per target (rather than MultiOutputRegressor)
#    allows per-target early stopping and produces better results. ---#


from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
 
import xgboost as xgb
from sklearn.metrics import r2_score
 
from models.base import (ModelResult, TARGET_NAMES,
                               print_metrics, compute_metrics, SEED)
 
DEFAULT_PATHS = {t: f"models/xgb_{t}.pkl" for t in TARGET_NAMES}
 
def _make_predict_fn(fitted_models: list):

    def  predict_fn(X_scaled: np.ndarray) -> np.ndarray:
        
        return np.column_stack([m.predict(X_scaled) for m in fitted_models])
    
    return predict_fn

def train(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
          n_estimators: int = 1000, learning_rate: float = 0.05, max_depth: int = 6,
          subsample: float = 0.8, colsample_bytree: float = 0.8, min_child_weight: int = 3,
          reg_alpha: float = 0.1, reg_lambda: float = 1.0, early_stopping_rounds:int = 40,
          save_paths: dict = None,
) -> ModelResult:
    
    if save_paths is None:
        save_paths = DEFAULT_PATHS

    print("\n Model: XGBoost (per-target, early stopping) ")

    fitted_models = []

    for i, target in enumerate(TARGET_NAMES):
        print(f"\n  Target: {target}")

        model = xgb.XGBRegressor(
            n_estimators = n_estimators,
            learning_rate = learning_rate,
            max_depth = max_depth,
            subsample = subsample,
            colsample_bytree = colsample_bytree,
            min_child_weight = min_child_weight,
            reg_alpha = reg_alpha,
            reg_lambda  = reg_lambda,
            random_state = SEED,
            tree_method = "hist",
            early_stopping_rounds = early_stopping_rounds,
            eval_metric = "rmse",
            verbosity = 0,
        )

        model.fit(
            X_tr, y_tr[:, i],
            eval_set = [(X_val, y_val[:, i])],
            verbose = False,
        )

        best_iter = model.best_iteration
        val_r2 = r2_score(y_val[:, i], model.predict(X_val))
        print(f" Best iteration : {best_iter}")
        print(f" Val R^2 : {val_r2:.6f}")
 
        path = save_paths[target]
        joblib.dump(model, path)
        print(f" Saved : {path}")
 
        fitted_models.append(model)

    y_pred_val = np.column_stack([m.predict(X_val) for m in fitted_models])
    metrics = compute_metrics(y_val, y_pred_val)
    print_metrics(metrics, "XGBoost")

    return ModelResult(
        name = "XGBoost",
        model = fitted_models, 
        predict_fn = _make_predict_fn(fitted_models),
        save_path = str(list(save_paths.values())),
        meta = {
            "n_targets": len(TARGET_NAMES),
            "best_iters":  [m.best_iteration for m in fitted_models],
            "hyperparams": {
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "subsample": subsample,
            },
        },
    )

def load(paths: dict = None) -> ModelResult:
    if paths is None:
        paths = DEFAULT_PATHS

    fitted_models = [joblib.load(paths[t]) for t in TARGET_NAMES]

    return ModelResult(
        name = "XGBoost",
        model = fitted_models,
        predict_fn = _make_predict_fn(fitted_models),
        save_path  = str(list(paths.values())),
    )

def get_feature_importances(
    result: ModelResult,
    feat_names: list[str],
) -> pd.DataFrame:
    
    importances = {}

    for i, tname in enumerate(TARGET_NAMES):

        importances[tname] = result.model[i].feature_importances_

    return pd.DataFrame(importances, index = feat_names)


