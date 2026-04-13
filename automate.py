
#--- I run DWSIM simulations for each of teh data points in the 
#    given rows  of my dataset + record outputs ---#

import numpy as np 
import pandas as pd
import time

SEED = 42
USE_DWSIM = False
DWSIM_PATH  = r"C:\Program Files\DWSIM"  
DWSIM_FILE  = "BenzTol_Column.dwxml" 

np.random.seed(SEED)


def run_dwsim_real(samples: pd.DataFrame) -> pd.DataFrame:

    import sys

    sys.path.append(DWSIM_PATH)

    import clr

    clr.AddReference("DWSIM.Automation")
    clr.AddReference("DWSIM.Interfaces")
    clr.AddReference("DWSIM.GlobalSettings")
    clr.AddReference("DWSIM.SharedClasses")
    clr.AddReference("DWSIM.Thermodynamics")
    clr.AddReference("DWSIM.UnitOperations")

    from DWSIM.Automation import Automation3
    from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType

    interf = Automation3()
    sim = interf.LoadFlowsheet(DWSIM_FILE)

    results = []

    for idx, row in samples.iterrows():
        
        try:
            feed = sim.GetFlowsheetSimulationObject("FEED")
            feed.SetOverallTemperature(row["T_Feed_C"] + 273.15)
            feed.SetOverallPressure(row["P_Feed_atm"] * 101325)
            feed.SetMolarFlow(row["F_Feed_kmolh"] / 3.6)   
            feed.SetOverallComposition([row["z_Feed"], 1 - row["z_Feed"]])

            col = sim.GetFlowsheetSimulationObject("DISTCOL")
            col.SetNumberOfStages(int(row["N_stages"]))
            col.SetFeedStage(int(row["N_feed"]))
            col.SetRefluxRatio(row["Reflux_ratio"])

            bot_spec = sim.GetFlowsheetSimulationObject("BOTSPEC")
            bot_spec.SetMolarFlowSpec(row["B_kmolh"] / 3.6)

            interf.CalculateFlowsheet(sim)

            dist = sim.GetFlowsheetSimulationObject("DISTILLATE")
            bots  = sim.GetFlowsheetSimulationObject("BOTTOMS")
            cond  = sim.GetFlowsheetSimulationObject("CONDENSER")
            reb   = sim.GetFlowsheetSimulationObject("REBOILER")

            xD = dist.GetOverallComposition()[0]
            xB = bots.getOverallComposition()[0]
            QC = cond.GetDutykW()
            QR = reb.GetDutykW()

            results.append({**row.to_dict(), "xD": xD, "xB": xB, "QC_kW": QC,
                            "QR_kW": QR, "converged": True})
            
        except Exception as e:
            results.append({**row.to_dict(), "xD": np.nan, "xB": np.nan, "QC_kW": np.nan, "QR_kW": np.nan, 
                            "converged": False, "error": str(e) })
            
        if idx % 50 == 0:
            print(f"Row {idx} / {len(samples)} done")
        
    return pd.DataFrame(results)


def _alpha_benzene_toluene(T_C: float, P_atm: float) -> float:

    alpha_ref = 2.45
    alpha = alpha_ref - 0.008 * (T_C - 80) - 0.05 * (P_atm - 1.0)

    return max(1.2, alpha)

def _fenske_nmin(xD: float, xB: float, alpha: float) -> float:

    return np.log((xD / (1 - xD)) * ((1 - xD) / xB)) / np.log(alpha)

def _underwood_rmin(z: float, q: float, alpha: float, xD: float) -> float:

    from scipy.optimize import brentq

    def underwood_eq(theta):

        return (alpha * z) / (alpha - theta) + (1 - z) / (1 - theta) - (1 - q)
    
    try:
        theta = brentq(underwood_eq, 1.001, alpha - 0.001)
        Rmin = (alpha * xD) / (alpha - theta) + xD / (1 - theta) - 1

        return max(0.5, Rmin)
    
    except Exception:
        return 1.2
    

def _feed_quality(T_C: float, P_atm: float, z: float) -> float:

    T_bubble = 80.1 + 27.9 * (1 - z) + 10.0 * (P_atm - 1.0)
    T_K = T_C + 273.15
    T_b_K = T_bubble + 273.15
    lambda_kJ = 33.5
    Cp_kJ = 0.15
    q = 1 + Cp_kJ * (T_b_K - T_K) / lambda_kJ

    return float(np.clip(q, 0.0, 0.15))

def simulate_column_physics(row: pd.Series) -> dict:

    T = row["T_Feed_C"]
    P = row["P_Feed_atm"]
    z = row["z_Feed"]
    N = int(row["N_Stages"])
    NF = int(row["N_Feed"])
    RR = row["Reflux_ratio"]
    B = row["B_kmolh"]
    F = row["F_Feed_kmolh"]
    D = row["D_kmolh"]

    alpha = _alpha_benzene_toluene(T, P)
    q = _feed_quality(T, P, z)

    xD_max = min(0.999, z + (1 - z) * (1 - np.exp(-0.6 * alpha * N/F)))

    xD_target = 0.95

    Nmin = _fenske_nmin(xD_target, 0.05, alpha)
    Rmin = _underwood_rmin(z, q, alpha, xD_target)


    X = (RR - Rmin) / (RR + 1.0)
    X = max(0.001, min(X, 0.99))

    def gilliland_Y(X):

        return 1 - np.exp(((1 + 54.4 * X) / (11 + 117.2 * X)) * ((X - 1) / (X ** 0.5)))
    
    Y = gilliland_Y(X)

    N_eff = (Y * (N + 1) + Nmin) / (1.0) 
    separation_factor = int(0.99, N_eff / (N_eff + Nmin + 1e-6))

    NF_optimal = int(round(N * (Rmin / (Rmin + 1))))
    NF_optimal = max(4, min(NF_optimal, N - 4))
    feed_penalty = 1 - 0.3 * abs(NF - NF_optimal) / max(N, 1)
    feed_penalty = max(0.5, feed_penalty)

    xD = z + (xD_max - z) * separation_factor * feed_penalty
    xD = float(np.clip(xD, z + 0.001, 0.9995))

    xB = (F * z - D * xD) / B
    xB = float(np.clip(xB, 0.0005, z - 0.001))

    #---Condenser duty: QC = -D * (RR + 1) * lambda_vap---#
    #---Reboiler duty:  QR = QC + F*h_F - D*h_D - B*h_B ---#
    #---lambda_vap for Benz-Tol mixture ≈ 33.5 kJ/mol---#

    lambda_vap = 33.5
    Cp_liq = 0.150
    T_ref = 25.0

    #---Molar flows mol/s  (kmol/h × 1000/3600) ---#
    D_mols = D * 1000 / 3600
    F_mols = F * 1000 / 3600
    B_mols = B * 1000/ 3600

    QC = -D_mols * (RR + 1) * lambda_vap  

    T_bubble_D = 80.1 * xD + 110.6 * (1 - xD) + 10 * (P - 1)  
    T_bubble_B = 80.1 * xB + 110.6 * (1 - xB) + 10 * (P - 1)

    h_D = Cp_liq * (T_bubble_D - T_ref)   
    h_B = Cp_liq * (T_bubble_B - T_ref)
    h_F = Cp_liq * (T - T_ref) - q * lambda_vap 

    QR =  Cp_liq * (T_bubble_D - T_ref)
    h_B =  Cp_liq * (T_bubble_B - T_ref)
    h_F = Cp_liq * (T - T_ref) - q * lambda_vap

    QR = (D_mols * h_D + B_mols * h_B - F_mols * h_F + D_mols * (RR + 1) * lambda_vap)
    QR = max(10.0, QR)

    noise_scale = 0.001
    xD += np.random.normal(0, noise_scale * xD)
    xB += np.random.normal(0, noise_scale * xB)
    QC += np.random.normal(0, 0.005 * abs(QC))
    QR += np.random.normal(0, 0.005 * QR)

    xD = float(np.clip(xD, 0.001, 0.9995))
    xB = float(np.clip(xB, 0.001, 0.9995))

    return {"xD": round(xD, 6), "xB": round(xB, 6), "QC_kW": round(QC, 3), "QR_kW": round(QR, 3),
            "alpha": round(alpha, 4), "q_feed": round(q, 4), "converged": True}


def run_synthetic(samples: pd.DataFrame) -> pd.DataFrame:

    results = []

    print(f"Running synthetic simulation for {len(samples)} samples")

    for idx, row in samples.iterrows():

        try:
            out = simulate_column_physics(row)
            results.append({**row.to_dict(), **out})

        except Exception as e:
            results.append({**row.to_dict(), "xD": np.nan, "xB": np.nan, "QC_kW": np.nan, 
                            "QR_kW": np.nan, "converged": False, "error": str(e)})
            
        if (idx + 1) % 100 == 0:
            print(f"{idx + 1} / {len(samples)} rows done")

    return pd.DataFrame(results)

if __name__ == "__main__":

    samples = pd.read_csv("samples.csv")

    print(f"Loaded {len(samples)} samples from samples.csv")

    t0 = time.time()

    if USE_DWSIM:
        print("Mode: REAL DWSIM via pythonnet")
        raw = run_dwsim_real(samples)

    else:
        print("Mode: SYNTHETIC (physics-informed equations)")
        raw = run_synthetic(samples)

    
    elapsed = time.time() - t0
    print(f"\n Simulation complete in {elapsed:.1f} s")

    raw.to_csv("dataset_raw.csv", index=False)
    print(f"Saved raw dataset: dataset_raw.csv ({len(raw)} rows)")

    clean = raw[raw["converged"] == True].copy()
    clean = clean.dropna(subset=["xD", "xB", "QC_kW", "QR_kW"])
    clean = clean[(clean["xD"] > 0) & (clean["xD"] < 1)]
    clean = clean[(clean["xB"] > 0) & (clean["xB"] < 1)]
    clean = clean.reset_index(drop = True)

    clean.to_csv("dataset.csv", index =False)
    print(f"Saved clean dataset: dataset.csv ({len(clean)} rows)")
    print("\nOutput summary:")
    print(clean[["xD", "xB", "QC_kW", "QR_kW"]].describe().round(4))











 








