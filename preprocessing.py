

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import os
import joblib

SEED = 42

os.makedirs("data", exist_ok = True)
os.makedirs("plots", exist_ok = True)

INPUT_FEATURES = [
    "T_Feed_C", "P_Feed_atm", "z_Feed",
    "N_Stages", "N_Feed", "Reflux_ratio",
    "B_fraction", "q_feed",
]
 
OUTPUT_TARGETS = ["xD", "xB", "QC_kW", "QR_kW"]
 
FEATURE_LABELS = {
    "T_Feed_C": "Feed temp [°C]",
    "P_Feed_atm": "Feed pressure [atm]",
    "z_Feed": "Feed composition [-]",
    "N_Stages": "No. stages",
    "N_Feed": "Feed stage",
    "Reflux_ratio": "Reflux ratio",
    "B_fraction": "Bottoms fraction",
    "q_feed": "Feed quality q",
    "xD": "Distillate purity xD",
    "xB": "Bottoms purity xB",
    "QC_kW": "Condenser duty [kW]",
    "QR_kW": "Reboiler duty [kW]",
}

def load_data(path: str = "samples.csv") -> pd.DataFrame:

    df = pd.read_csv(path)

    print(f"Loaded {len(df)} rows from {path}")
    print(f"Columns: {list(df.columns)} ")

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    n0 = len(df)

    if "converged" in df.columns:
        df = df[df["converged"] == True].copy()

    required = INPUT_FEATURES + OUTPUT_TARGETS
    available = [c for c in required if c in df.columns]
    df = df.dropna(subset = available)

    df = df[(df["xD"] > 0.01) & (df["xD"] < 0.9999)]
    df = df[(df["xB"] > 0.0001) & (df["xB"] < 0.99)]
    df = df[df["QC_kW"] < 0]      
    df = df[df["QR_kW"] > 0] 

    if all (c in df.columns for c in ["F_Feed_kmolh", "D_kmolh", "B_kmolh"]):
        lhs = df["F_Feed_kmolh"] * df["z_Feed"]
        rhs = df["D_kmolh"] * df["xD"] + df["B_kmolh"] * df["xB"]
        balance_err = (lhs - rhs).abs() / lhs
        df = df[balance_err < 0.05]

    for col in OUTPUT_TARGETS:
        if col not in df.columns:
            continue

        Q1 = df[col].quantile(0.01)
        Q3 = df[col].quantile(0.99)
        df = df[(df[col] >= Q1) & (df[col] <= Q3)]

    df = df.reset_index(drop = True)

    print(f"Cleaning: {n0} → {len(df)} rows ({n0 - len(df)} removed)")

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    df["feed_stage_rel"] = df["N_Feed"] / df["N_Stages"]

    df["RR_excess"] = df["Reflux_ratio"] - 1.3

    df["RR_x_z"] = df["Reflux_ratio"] * df["z_Feed"]

    df["log_RR"] = np.log(df["Reflux_ratio"])

    return df

def plot_eda(df: pd.DataFrame) -> None:

    available_inputs = [c for c in INPUT_FEATURES if c in df.columns]
    available_outputs = [c for c in OUTPUT_TARGETS  if c in df.columns]
    all_cols = available_inputs + available_outputs

    #---Output Distributions---#
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    
    for ax, col in zip(axes, available_outputs):

        ax.hist(df[col], bins = 40, edgecolor = "white", linewidth = 0.4, color = "#378ADD", alpha = 0.85)
        ax.set_xlabel(FEATURE_LABELS.get(col, col), fontsize = 10)
        ax.set_ylabel("Count", fontsize = 10)
        ax.set_title(f"Distribution of {col}", fontsize = 11)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("plots/eda_distributions.png", dpi = 150, bbox_inches = "tight")
    plt.close()
    print("Saved: plots/eda_distributions.png")

    #---Corr heatmap---#
    corr_cols = [c for c in all_cols if c in df.columns]
    corr = df[corr_cols].corr()
    labels = [FEATURE_LABELS.get(c, c) for c in corr_cols]

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask = mask, annot = True, fmt = ".2f", cmap = "RdBu_r",
                center = 0, vmin = -1, vmax = 1, linewidths = 0.4,
                xticklabels = labels, yticklabels = labels,
                cbar_kws = {"shrink": 0.7}, ax = ax, annot_kws = {"size": 8})
    ax.set_title("Correlation matrix — inputs and outputs", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig("plots/eda_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/eda_correlation.png")

    #--- Reflux ratio vs xD, N_Stages vs xD ---#

    fig, axes = plt.subplots(2, 4, figsize = (16, 8))
    for i, inp in enumerate(available_inputs[:4]):
        for j, out in enumerate(available_outputs[:2]):
            ax = axes[j, i]
            sc = ax.scatter(df[inp], df[out], c = df["z_Feed"],
                            cmap = "viridis", s = 8, alpha = 0.6)
            ax.set_xlabel(FEATURE_LABELS.get(inp, inp), fontsize =9)
            ax.set_ylabel(FEATURE_LABELS.get(out, out), fontsize = 9)
            ax.spines[["top", "right"]].set_visible(False)

            if i == 3:
                plt.colorbar(sc, ax = ax, label = "z_Feed")
    plt.suptitle("Input vs output scatter (coloured by feed composition)",
                 fontsize = 12, y = 1.01)
    plt.tight_layout()
    plt.savefig("plots/eda_scatter.png", dpi = 150, bbox_inches = "tight")
    plt.close()
    print("Saved: plots/eda_scatter.png")


def split_and_scale(df: pd.DataFrame):
    feature_cols = [c for c in INPUT_FEATURES + ["feed_stage_rel", "RR_excess",
                    "RR_x_z", "log_RR"] if c in df.columns]
    target_cols  = [c for c in OUTPUT_TARGETS if c in df.columns]

    X = df[feature_cols].values.astype(float)
    y = df[target_cols].values.astype(float)

    z_quartile = pd.qcut(df["z_Feed"], q = 4, labels = False)
    X_tr, X_tmp, y_tr, y_tmp, z_tr, z_tmp = train_test_split(
        X, y, z_quartile, test_size = 0.30, random_state = SEED, stratify = z_quartile
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size = 0.50, random_state = SEED
    )
    print(f"\n Split sizes -> train: {len(X_tr)}, val: {len(X_val)}, test: {len(X_te)}")

    scaler_X = StandardScaler()
    X_tr_s = scaler_X.fit_transform(X_tr)
    X_val_s = scaler_X.transform(X_val)
    X_te_s = scaler_X.transform(X_te)

    scaler_y = StandardScaler()
    y_tr_s = scaler_y.fit_transform(y_tr)
    y_val_s = scaler_y.transform(y_val)
    y_te_s = scaler_y.transform(y_te)

    joblib.dump(scaler_X, "data/scaler_X.pkl")
    joblib.dump(scaler_y, "data/scaler_y.pkl")
    print("Saved: data/scaler_X.pkl, data/scaler_y.pkl")

    for split_name, X_arr, y_arr in [("train", X_tr, y_tr), ("val", X_val, y_val), ("test", X_te, y_te)]:
        pd.DataFrame(X_arr, columns = feature_cols).to_csv(
            f"data/X_{split_name}.csv", index = False)
        
        pd.DataFrame(y_arr, columns = target_cols).to_csv(
            f"data/y_{split_name}.csv", index = False)
        
    np.save("data/X_train_s.npy", X_tr_s)
    np.save("data/X_val_s.npy", X_val_s)
    np.save("data/X_test_s.npy", X_te_s)
    np.save("data/y_train_s.npy", y_tr_s)
    np.save("data/y_val_s.npy", y_val_s)
    np.save("data/y_test_s.npy", y_te_s)
    np.save("data/y_train.npy", y_tr)
    np.save("data/y_val.npy", y_val)
    np.save("data/y_test.npy", y_te)

    pd.Series(feature_cols).to_csv("data/feature_names.csv", index = False)
    pd.Series(target_cols).to_csv("data/target_names.csv", index = False)

    print("Saved all split arrays to data/")
    
    return (X_tr_s, X_val_s, X_te_s,
            y_tr_s, y_val_s, y_te_s,
            y_tr, y_val, y_te,
            scaler_X, scaler_y,
            feature_cols, target_cols)


if __name__ == "__main__":

    df = load_data("dataset.csv")
    df = clean_data(df)
    df = engineer_features(df)
 
    print("\nFinal dataset summary:")
    display_cols = [c for c in INPUT_FEATURES + OUTPUT_TARGETS if c in df.columns]
    print(df[display_cols].describe().round(4))
 
    plot_eda(df)
    split_and_scale(df)
 
    print("\nPreprocessing complete.")




