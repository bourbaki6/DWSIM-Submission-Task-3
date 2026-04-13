


from __future__ import annotations
import os
import time
import numpy as np
import pandas as pd
 
from models.base import ModelResult, TARGET_NAMES, compute_metrics
from models import (polynomial_regression, random_forest,
    xgboost_model, 
    ann,
)
 
 
class ModelRegistry:

    def __init__(self):
        self.results: dict[str, ModelResult] = {}

    def train_all(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        y_tr_s: np.ndarray  = None,   
        y_val_s: np.ndarray  = None,  
        scaler_y = None,   
        skip: list[str] = None,   
        **kwargs,
    ) -> "ModelRegistry":
        
        skip = skip or []
        os.makedirs("models", exist_ok=True)
        total_start = time.time()

        if "Polynomial" not in skip:
            t0 = time.time()
            result = polynomial_regression.train(X_tr, y_tr, X_val, y_val)
            self.results["Polynomial"] = result
            print(f"  [Polynomial] done in {time.time()-t0:.1f}s\n")

        
        if "Random Forest" not in skip:
            t0 = time.time()
            result = random_forest.train(X_tr, y_tr, X_val, y_val)
            self.results["Random Forest"] = result
            print(f"  [Random Forest] done in {time.time()-t0:.1f}s\n")

        if "XGBoost" not in skip:
            t0 = time.time()
            result = xgboost_model.train(X_tr, y_tr, X_val, y_val)
            self.results["XGBoost"] = result
            print(f"  [XGBoost] done in {time.time()-t0:.1f}s\n")

        if "ANN" not in skip:
            if y_tr_s is None or y_val_s is None or scaler_y is None:
                print("  [ANN] Skipped — y_tr_s / y_val_s / scaler_y not provided.")
            else:
                t0 = time.time()
                result = ann.train(
                    X_tr, y_tr_s, X_val, y_val_s, y_val, scaler_y
                )
                if result is not None:
                    self.results["ANN"] = result
                    print(f"  [ANN] done in {time.time()-t0:.1f}s\n")
 
        print(f"\nAll models trained in {time.time()-total_start:.1f}s")
        print(f"Registry contains: {list(self.results.keys())}")
        
        return self
    
    def load_all(
        self,
        scaler_y=None,
        skip: list[str] = None,
    ) -> "ModelRegistry":
        
        skip = skip or []
 
        loaders = {
            "Polynomial": lambda: polynomial_regression.load(),
            "Random Forest": lambda: random_forest.load(),
            "XGBoost": lambda: xgboost_model.load(),
            "ANN": lambda: ann.load(scaler_y=scaler_y),
        }

        for name, loader_fn in loaders.items():
            if name in skip:
                continue
            try:
                self.results[name] = loader_fn()
                print(f"  Loaded: {name}")
            except Exception as e:
                print(f"  Could not load {name}: {e}")
 
        print(f"Registry contains: {list(self.results.keys())}")
        
        return self
    
    def predict(self, model_name: str, X_scaled: np.ndarray) -> np.ndarray:

        if model_name not in self.results:
            raise KeyError(f"Model '{model_name}' not in registry. "
                           f"Available: {list(self.results.keys())}")
        return self.results[model_name].predict(X_scaled)
    
    def predict_all(self, X_scaled: np.ndarray) -> dict[str, np.ndarray]:

        return {name: res.predict(X_scaled)
                for name, res in self.results.items()}
    
    def evaluate(
        self,
        model_name: str,
        X_scaled: np.ndarray,
        y_true: np.ndarray,
    ) -> pd.DataFrame:
        
        y_pred = self.predict(model_name, X_scaled)
        
        return compute_metrics(y_true, y_pred)
    
    def evaluate_all(
        self,
        X_scaled: np.ndarray,
        y_true: np.ndarray,
    ) -> pd.DataFrame:
        
        rows = []

        for name, result in self.results.items():
            y_pred = result.predict(X_scaled)
            metrics = compute_metrics(y_true, y_pred)
            metrics = metrics.reset_index()
            metrics.insert(0, "model", name)
            rows.append(metrics)

        if not rows:
            return pd.DataFrame()
        
        combined = pd.concat(rows, ignore_index=True)
        return combined
 
    def summary(self) -> pd.DataFrame:

        rows = []

        for name, res in self.results.items():
            rows.append({
                "model": name,
                "save_path": res.save_path,
                "meta": str(res.meta),
            })

        df = pd.DataFrame(rows)
        print("\nRegistry summary:")
        print(df.to_string(index=False))
        
        return df
    
    @property
    def model_names(self) -> list[str]:
        return list(self.results.keys())
 
    def __repr__(self):
        return f"ModelRegistry(models={self.model_names})"

