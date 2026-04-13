


from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET_NAMES = ["xD", "xB", "QC_kW", "QR_kW"]
 
TARGET_LABELS = {
    "xD":    "Distillate purity xD [-]",
    "xB":    "Bottoms purity xB [-]",
    "QC_kW": "Condenser duty QC [kW]",
    "QR_kW": "Reboiler duty QR [kW]",
}
 
SEED = 42

@dataclass
class ModelResult:

    name: str
    model: Any
    predict_fn: Callable[[np.ndarray], np.ndarray]
    save_path: str = ""
    meta: dict = field(default_factory = dict)

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:

        return self.predict_fn(X_scaled)
    

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list = TARGET_NAMES) -> pd.DataFrame:

    rows = []
    for i, name in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        rows.append({
            "target": name,
            "MAE": round(mean_absolute_error(yt, yp), 6),
            "RMSE": round(float(np.sqrt(mean_squared_error(yt, yp))), 6),
            "R2": round(r2_score(yt, yp), 6),
        })

    return pd.DataFrame(rows).set_index("target")

def print_metrics(metrics: pd.DataFrame, model_name: str) -> None:

    print(f"{model_name} — validation metrics")
    print(metrics.to_string())