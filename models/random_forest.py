
#--- Random Forest surrogate using scikit-learn's RandomForestRegressor
#    wrapped in MultiOutputRegressor, with RandomizedSearchCV tuning ---#
 

from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
 
from sklearn.ensemble    import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
 
from models.base import (ModelResult, TARGET_NAMES, print_metrics, compute_metrics, SEED)
 
DEFAULT_PATH = "models/rf_model.pkl"



def _make_predict_fn(fitted_model):

    def predict_fn(X_scaled: np.ndarray) -> np.ndarray:

        return fitted_model.predict(X_scaled)
    
    return predict_fn

def train(X_tr: np.ndarray, y_tr:  np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    n_iter: int = 20, cv: int = 3,
    save_path: str = DEFAULT_PATH,) -> ModelResult:

    print("\n Model: Random Forest Regressor")

    param_dist = {
        "estimator__n_estimators": [200, 300, 400, 500],
        "estimator__max_depth": [None, 10, 20, 30],
        "estimator__min_samples_leaf": [1, 2, 4],
        "estimator__min_samples_split":[2, 5, 10],
        "estimator__max_features": ["sqrt", 0.5, 0.7, 1.0],
        "estimator__bootstrap": [True, False],
    }

    base_rf = RandomForestRegressor(random_state = SEED, n_jobs = -1)
    mo_rf = MultiOutputRegressor(base_rf, n_jobs = 1)

    search = RandomizedSearchCV(
        mo_rf, param_dist,
        n_iter = n_iter,
        cv = cv,
        scoring = "r2",
        random_state = SEED,
        n_jobs = 1,
        verbose = 1,
        refit = True,
    )
    search.fit(X_tr, y_tr)
    best = search.best_estimator_

    print(f"\n  Best hyperparameters found:")

    for k, v in search.best_params_.items():
        print(f" {k}: {v}")

    metrics = compute_metrics(y_val, best.predict(X_val))
    print_metrics(metrics, "Random Forest")

    joblib.dump(best, save_path)
    print(f" Saved: {save_path}")

    return ModelResult(
        name = "Random Forest",
        model = best,
        predict_fn = _make_predict_fn(best),
        save_path  = save_path,
        meta = {"best_params": search.best_params_},
    )

def load(path: str = DEFAULT_PATH) -> ModelResult:

    fitted = joblib.load(path)
    
    return ModelResult(
        name = "Random Forest",
        model = fitted,
        predict_fn = _make_predict_fn(fitted),
        save_path = path,
    )

def get_feature_importances(result: ModelResult, 
                            feat_names: list[str],) -> pd.DataFrame:
    importances = {}

    for i, tname in enumerate(TARGET_NAMES):
        sub = result.model.estimators_[i]
        importances[tname] = sub.feature_importances_

    return pd.DataFrame(importances, index = feat_names)

    



    
 
