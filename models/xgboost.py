

import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
 

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
import xgboost as xgb
 
warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)
 
SEED = 42
np.random.seed(SEED)
 
TARGET_NAMES = ["xD", "xB", "QC_kW", "QR_kW"]
 
 
def load_arrays():
    X_tr  = np.load("data/X_train_s.npy")
    X_val = np.load("data/X_val_s.npy")
    X_te  = np.load("data/X_test_s.npy")
    y_tr  = np.load("data/y_train.npy")    
    y_val = np.load("data/y_val.npy")
    y_te  = np.load("data/y_test.npy")
    feat_names = pd.read_csv("data/feature_names.csv").iloc[:, 0].tolist()
    return X_tr, X_val, X_te, y_tr, y_val, y_te, feat_names
 
 

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    target_names: list) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        rows.append({
            "target": name,
            "MAE":    mean_absolute_error(yt, yp),
            "RMSE":   np.sqrt(mean_squared_error(yt, yp)),
            "R2":     r2_score(yt, yp),
        })
    return pd.DataFrame(rows).set_index("target")


class XGBoost:

    def train_xgboost(X_tr, y_tr, X_val, y_val):
        
        print("Model 3: XGBoost (per-target, with early stopping)")
        
        xgb_models = []
        for i, target in enumerate(TARGET_NAMES):
            
            print(f"\n  Training XGBoost for: {target}")
            model = xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=SEED,
                tree_method="hist",
                early_stopping_rounds=30,
                eval_metric="rmse",
                verbosity=0,
                )
            
            model.fit(
                X_tr, y_tr[:, i], eval_set=[(X_val, y_val[:, i])], verbose = False,
                )
            
            best_iter = model.best_iteration
            val_pred  = model.predict(X_val)
            r2_val    = r2_score(y_val[:, i], val_pred)
            
            print(f" Best iteration: {best_iter}, val R ^ 2: {r2_val:.5f}")
            
            xgb_models.append(model)
            joblib.dump(model, f"models/xgb_{target}.pkl")
 
            print("\n  Saved: models/xgb_<target>.pkl for each output")
            
            return xgb_models

    