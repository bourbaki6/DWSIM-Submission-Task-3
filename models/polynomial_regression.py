
#--- Polynomial Regression surrogate with Ridge regularisation
#    Deg 2 and  deg 3 polynomial selects teh better performing one
#    on the val set, and returns a ModelResult ---#

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
 

import joblib

from models.base import ModelResult, TARGET_NAMES, print_metrics, compute_metrics

DEFAULT_PATH = "models/poly_model.pkl"


def _build_pipeline(degree: int, alpha: float = 1.0) -> MultiOutputRegressor:

    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree = degree, include_bias = False)),
        ("ridge", Ridge(alpha = alpha)),
    ])
    return MultiOutputRegressor(pipe, n_jobs = -1)

def _mean_r2(model, X: np.ndarray, y: np.ndarray) -> float:

    y_pred = model.predict(X)
    return float(np.mean([
        r2_score(y[:, i], y_pred[:, i])
        for i in range(y.shape[1])
    ]))

def _make_predict_fn(fitted_model): 

    def predict_fn(X_scaled: np.ndarray) -> np.ndarray:
        
        return fitted_model.predict(X_scaled)
    
    return predict_fn

def train(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    degrees: list[int] = (2, 3),
    alphas: list[float] = (0.1, 1.0, 10.0),
    save_path: str = DEFAULT_PATH,
) -> ModelResult:
    
    print("\n  Model: Polynomial Regression (Ridge) ")

    best_r2    = -np.inf
    best_model = None
    best_cfg   = {}

    for degree in degrees:

        for alpha in alphas:
            model = _build_pipeline(degree, alpha)
            model.fit(X_tr, y_tr)
            val_r2 = _mean_r2(model, X_val, y_val)
            print(f"  degree={degree}  alpha={alpha:6.1f}  val R²={val_r2:.5f}")

            if val_r2 > best_r2:
                best_r2 = val_r2
                best_model = model 
                best_cfg = {"degree": degree, "alpha": alpha}

    print(f"\n Selected -> degree = {best_cfg['degree']},"
          f"alpha = {best_cfg['alpha']}, val R^2 = {best_r2:.5f}")
    
    metrics = compute_metrics(y_val, best_model.predict(X_val))
    print_metrics(metrics, "Polynomial Regression")

    joblib.dump(best_model, save_path)
    print(f"Saved: {save_path}")

    return  ModelResult(
        name = "Polynomial",
        model = best_model,
        predict_fn = _make_predict_fn(best_model),
        save_path = save_path,
        meta = best_cfg,
    )

def load(path: str = DEFAULT_PATH) -> ModelResult:
    fitted = joblib.load(path)

    return ModelResult(
        name = "Polynomial",
        model = fitted,
        predict_fn = _make_predict_fn(fitted),
        save_path = path,
    )