


import argparse
import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
 
from models.model_registry import ModelRegistry
from models.base import TARGET_NAMES
 
os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)
 
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["Polynomial","RF","Random Forest","XGBoost","ANN"])
    return parser.parse_args()
 
 
def load_data():
    required = [
        "data/X_train_s.npy","data/X_val_s.npy","data/X_test_s.npy",
        "data/y_train.npy","data/y_val.npy","data/y_test.npy",
        "data/y_train_s.npy","data/y_val_s.npy",
        "data/scaler_X.pkl","data/scaler_y.pkl","data/feature_names.csv",
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("ERROR: Missing files:", missing)
        print("Run: python preprocessing.py")
        sys.exit(1)
 
    arrays = {
        "X_tr": np.load("data/X_train_s.npy"),
        "X_val": np.load("data/X_val_s.npy"),
        "X_te": np.load("data/X_test_s.npy"),
        "y_tr": np.load("data/y_train.npy"),
        "y_val": np.load("data/y_val.npy"),
        "y_te":  np.load("data/y_test.npy"),
        "y_tr_s": np.load("data/y_train_s.npy"),
        "y_val_s": np.load("data/y_val_s.npy"),
    }
    scaler_X = joblib.load("data/scaler_X.pkl")
    scaler_y  = joblib.load("data/scaler_y.pkl")
    feat_names = pd.read_csv("data/feature_names.csv").iloc[:,0].tolist()
 
    print(f"Train: {arrays['X_tr'].shape[0]} | Val: {arrays['X_val'].shape[0]} | Test: {arrays['X_te'].shape[0]}")
    print(f"Features ({len(feat_names)}): {feat_names}")
    
    return arrays, scaler_X, scaler_y, feat_names

def plot_metrics_bar(metrics_df):

    COLORS = {"Polynomial":"#888780","Random Forest":"#1D9E75",
              "XGBoost":"#378ADD","ANN":"#D85A30"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for ax, metric in zip(axes, ["R2","RMSE","MAE"]):
        pivot = metrics_df.pivot(index="target", columns="model", values=metric)
        x = range(len(pivot))
        w = 0.8 / len(pivot.columns)
        for k, col in enumerate(pivot.columns):
            ax.bar([xi+k*w for xi in x], pivot[col], width=w*0.9,
                   label=col, color=COLORS.get(col,"#888780"), alpha=0.85)
        ax.set_xticks([xi+w*(len(pivot.columns)-1)/2 for xi in x])
        ax.set_xticklabels(pivot.index, fontsize=9)
        ax.set_title(metric, fontsize=11)
        ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        if metric == "R2":
            ax.set_ylim(max(0, metrics_df["R2"].min()-0.03), 1.01)
    plt.suptitle("Model comparison — test set metrics", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/metrics_comparison.png")
 
 
def plot_learning_curve(registry, X_tr, y_tr, model_name="Random Forest", target_idx=0):
    
    if model_name not in registry.results:
        return
    
    result = registry.results[model_name]
    target_name = TARGET_NAMES[target_idx]
    fracs = np.linspace(0.1, 1.0, 8)
    train_rmse, val_rmse = [], []
    
    import copy, warnings
    
    for frac in fracs:
        n   = max(20, int(frac * len(X_tr)))
        idx = np.random.choice(len(X_tr), n, replace=False)
        m   = copy.deepcopy(result.model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(m, list):
                m[target_idx].fit(X_tr[idx], y_tr[idx, target_idx])
                tr_p = m[target_idx].predict(X_tr[idx])
                vl_p = m[target_idx].predict(X_tr)
            else:
                m.fit(X_tr[idx], y_tr[idx])
                tr_p = m.predict(X_tr[idx])[:, target_idx]
                vl_p = m.predict(X_tr)[:, target_idx]
        train_rmse.append(float(np.sqrt(mean_squared_error(y_tr[idx, target_idx], tr_p))))
        val_rmse.append(float(np.sqrt(mean_squared_error(y_tr[:, target_idx], vl_p))))
    
    ns = [int(f*len(X_tr)) for f in fracs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, train_rmse, "o-",  label="Train RMSE", color="#378ADD")
    ax.plot(ns, val_rmse,   "s--", label="Val RMSE",   color="#D85A30")
    ax.set_xlabel("Training samples"); ax.set_ylabel("RMSE")
    ax.set_title(f"Learning curve — {model_name} — {target_name}")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fname = f"plots/learning_curve_{model_name.replace(' ','_')}_{target_name}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {fname}")
 
 
def main():
    args = parse_args()
    skip = [{"RF":"Random Forest"}.get(s, s) for s in args.skip]
    if skip:
        print(f"Skipping: {skip}")
 
    arrays, scaler_X, scaler_y, feat_names = load_data()
 
    registry = ModelRegistry()
    registry.train_all(
        X_tr = arrays["X_tr"],
        y_tr = arrays["y_tr"],
        X_val = arrays["X_val"],
        y_val = arrays["y_val"],
        y_tr_s = arrays["y_tr_s"],
        y_val_s = arrays["y_val_s"],
        scaler_y= scaler_y,
        skip = skip,
    )
 
    
    print("  FINAL EVALUATION — TEST SET")
    
    metrics_df = registry.evaluate_all(arrays["X_te"], arrays["y_te"])
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv("results_metrics.csv", index = False)
    print("\nSaved: results_metrics.csv")
 
    plot_metrics_bar(metrics_df)
    plot_learning_curve(registry, arrays["X_tr"], arrays["y_tr"])
 
    mean_r2 = metrics_df.groupby("model")["R2"].mean().sort_values(ascending=False)
    
    print("\nMean R^2 per model:")
    print(mean_r2.round(5).to_string())
    print(f"\nBest model: {mean_r2.idxmax()}  (R² = {mean_r2.max():.5f})")
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()
 
