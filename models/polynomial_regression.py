
import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
import xgboost as xgb
 
warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)
 
SEED = 42
np.random.seed(SEED)
 
TARGET_NAMES = ["xD", "xB", "QC_kW", "QR_kW"]
 
 

def load_arrays():

    X_tr = np.load("data/X_train_s.npy")
    X_val = np.load("data/X_val_s.npy")
    X_te = np.load("data/X_test_s.npy")
    y_tr = np.load("data/y_train.npy")    
    y_val = np.load("data/y_val.npy")
    y_te = np.load("data/y_test.npy")
    feat_names = pd.read_csv("data/feature_names.csv").iloc[:, 0].tolist()
    return X_tr, X_val, X_te, y_tr, y_val, y_te, feat_names
 
 
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    target_names: list) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        rows.append(
            {
            "target": name,
            "MAE": mean_absolute_error(yt, yp),
            "RMSE": np.sqrt(mean_squared_error(yt, yp)),
            "R2": r2_score(yt, yp),
            }
        )
    return pd.DataFrame(rows).set_index("target")


def train_polynomial(X_tr, y_tr, X_val, y_val):
    
    print("Model 1: Polynomial Regression (degree=2, Ridge)")
   
 
    best_val_r2 = -np.inf
    best_model  = None
    best_degree = 2
 
    for degree in [2, 3]:

        pipe = Pipeline([
            ("poly",  PolynomialFeatures(degree = degree, include_bias = False)),
            ("ridge", Ridge(alpha = 1.0)),
        ])
        model = MultiOutputRegressor(pipe, n_jobs = -1)
        model.fit(X_tr, y_tr)
        y_pred_val = model.predict(X_val)
        r2_mean = np.mean([r2_score(y_val[:, i], y_pred_val[:, i])
                           for i in range(y_val.shape[1])])
        print(f"Degree {degree}: val mean R^2 = {r2_mean:.4f}")
        
        if r2_mean > best_val_r2:
            best_val_r2 = r2_mean
            best_model  = model
            best_degree = degree
 
    print(f"Selected degree: {best_degree}, val R² = {best_val_r2:.4f}")
    joblib.dump(best_model, "models/poly_model.pkl")
    
    
    print("\n Saved: models/poly_model.pkl")
    
    return best_model