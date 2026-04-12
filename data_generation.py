
#--- Generating samples of operating conditions for the 
#    Benzene - Toluene distillation column surrogate modelling project

import numpy as np
import pandas as pd

from scipy.stats.qmc import LatinHypercube, scale

SEED = 42
N_SAMPLES = 500
BOUNDS = {
    "T_Feed_C": (60.0, 120.0),
    "P_Feed_atm": (1.0, 3.0),
    "z_Feed": (0.20, 0.80),
    "N_Stages": (10, 35),
    "N_Feed": (0.20, 0.80),
    "Reflux_ratio": (1.5, 6.0),
    "B_fraction": (0.20, 0.80),
}

CONTINUOUS_VARS = ["T_Feed_C", "P_Feed_atm", "z_Feed", "Reflux_ratio", "B_fraction"]
INTEGER_VARS = ["N_Stages", "N_Feed"]

def generate_samples(n: int = N_SAMPLES, seed: int = SEED)-> pd.DataFrame:

    rng = np.random.default_rng(seed)
    sampler = LatinHypercube(d = len(BOUNDS), seed = rng)
    unit_sample = sampler.random( n = n)

    lowers = np.array([v[0] for v in BOUNDS.values()])
    uppers = np.array([v[1] for v in BOUNDS.values()])
    scaled = scale(unit_sample, lowers, uppers)

    df = pd.DataFrame(scaled, columns = list(BOUNDS.keys()))

    df["N_Stages"] = df["N_Stages"].round().astype(int) #---

    #---Feed Stage = Frac x N_Stages, clamped to [4, N - 4] for physical validity---#
    df["N_Feed_frac"] = df["N_Feed"]
    df["N_Feed"] = (df["N_Feed_frac"] * df["N_Stages"]).round().astype(int)
    df["N_Feed"] = df.apply(lambda r: int(np.clip(r["N_Feed"], 4, r["N_Stages"] - 4)), axis = 1)
    df.drop(columns = ["N_Feed_frac"], inplace = True)


    #---Feed Flow fixed at 100 kmol/h ---#
    df["F_Feed_kmolh"] = 100.0
    df["B_kmolh"] = df["B_fraction"] * df["F_Feed_kmolh"]
    df["D_kmolh"] = df["F_Feed_kmolh"] - df["B_kmolh"]

    df["P_col_atm"] = df["P_Feed_atm"]

    col_order= [
        "T_Feed_C", "P_Feed_atm", "z_Feed",
        "N_Stages", "N_Feed", "Reflux_ratio",
        "B_fraction", "B_kmolh", "D_kmolh",
        "F_Feed_kmolh", "P_col_atm",
    ]

    df = df[col_order]

    print(f"Generated {len(df)} samples")
    print(df.describe().round(3))

    return df

def check_physical_validity(df: pd.DataFrame) -> pd.DataFrame:

    n_before = len(df)

    #--- 1. Feed stage must be inside the col---#
    #--- 2, Bottom flow must be less than feed flow ---#
    #--- 3. Ditsillate flow must be pos---#
    #---4. Reflux ratio must be above min

    df = df[df["N_Feed"] >= 4]
    df = df[df["N_Feed"] <= df["N_Stages"] - 4]

    df = df[df["B_kmolh"] < df["F_Feed_kmolh"]]
    df = df[df["B_kmolh"] > 0]

    df = df[df["D_kmolh"] > 0]

    df = df[df["Reflux_ratio"] >= 1.5]

    n_after = len(df)

    print(f"Physical validity filter: {n_before} -> {n_after} rows"
          f"({n_before - n_after} removed)")
    df = df.reset_index(drop = True)

    return df

if __name__ == "__main__":
    samples = generate_samples()
    samples = check_physical_validity(samples)
    samples.to_csv("samples.csv", index = False)
    
    print("\n The Dataset has been saeved.")