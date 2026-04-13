

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
 
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Distillation Surrogate",
    page_icon="FOSSEE",
    layout="wide",
    initial_sidebar_state="expanded",
)

TARGET_NAMES = ["xD", "xB", "QC_kW", "QR_kW"]
 
TARGET_LABELS = {
    "xD": "Distillate purity xD [-]",
    "xB": "Bottoms purity xB [-]",
    "QC_kW": "Condenser duty QC [kW]",
    "QR_kW": "Reboiler duty QR [kW]",
}
 
TARGET_UNITS = {"xD": "mol/mol", "xB": "mol/mol",
                "QC_kW": "kW", "QR_kW": "kW"}
 
FEATURE_LABELS = {
    "T_Feed_C": "Feed temperature [°C]",
    "P_Feed_atm": "Feed pressure [atm]",
    "z_Feed": "Feed composition (benzene) [-]",
    "N_Stages": "Number of stages",
    "N_Feed": "Feed stage location",
    "Reflux_ratio": "Reflux ratio",
    "B_fraction": "Bottoms fraction [-]",
    "q_feed": "Feed quality q [-]",
    "feed_stage_rel": "Feed stage (relative) [-]",
    "RR_excess": "Excess reflux ratio",
    "RR_x_z": "RR × z_feed",
    "log_RR": "ln(Reflux ratio)",
}
 
COLORS = {
    "Polynomial":"#888780",
    "Random Forest": "#1D9E75",
    "XGBoost": "#378ADD",
    "ANN": "#D85A30",
}

@st.cache_resource
def load_registry():
    from models.model_registry import ModelRegistry
    scaler_y = joblib.load("data/scaler_y.pkl")
    registry = ModelRegistry()
    registry.load_all(scaler_y=scaler_y)
    return registry
 
 
@st.cache_resource
def load_scalers():
    return joblib.load("data/scaler_X.pkl"), joblib.load("data/scaler_y.pkl")
 
 
@st.cache_data
def load_dataset():
    return pd.read_csv("dataset.csv")
 
 
@st.cache_data
def load_metrics():
    try:
        return pd.read_csv("results/metrics_all_models.csv")
    except FileNotFoundError:
        try:
            return pd.read_csv("results_metrics.csv")
        except FileNotFoundError:
            return None
 
 
@st.cache_data
def load_feature_names():
    return pd.read_csv("data/feature_names.csv").iloc[:, 0].tolist()
 
 
def assets_ready():
    checks = [
        "data/scaler_X.pkl",
        "data/scaler_y.pkl",
        "data/feature_names.csv",
        "models/poly_model.pkl",
        "dataset.csv",
    ]
    return all(os.path.exists(p) for p in checks)
 
 

def build_feature_row(inputs: dict, feat_names: list) -> np.ndarray:
    """Convert sidebar input dict to a (1, n_features) array."""
    rr = inputs["Reflux_ratio"]
    z = inputs["z_Feed"]
    N  = inputs["N_Stages"]
    NF = inputs["N_Feed"]
    full = {
        "T_feed_C":  inputs["T_Feed_C"],
        "P_feed_atm": inputs["P_Feed_atm"],
        "z_feed":  z,
        "N_stages": float(N),
        "N_Feed": float(NF),
        "reflux_ratio": rr,
        "B_fraction": inputs["B_fraction"],
        "q_feed": inputs["q_feed"],
        "feed_stage_rel": NF / N,
        "RR_excess":  rr - 1.3,
        "RR_x_z":   rr * z,
        "log_RR":  np.log(rr),
    }
    return np.array([[full.get(f, 0.0) for f in feat_names]])
 
 
def sidebar_inputs():
    st.sidebar.header("Operating conditions")
    inp = {}
    inp["T_Feed_C"]  = st.sidebar.slider("Feed temperature [°C]", 60.0, 120.0, 80.0, 0.5)
    inp["P_Feed_atm"] = st.sidebar.slider("Feed pressure [atm]", 1.0, 3.0,  1.5,  0.05)
    inp["z_Feed"] = st.sidebar.slider("Feed composition (benzene)",0.20,  0.80,  0.50, 0.01)
    inp["N_Stages"] = st.sidebar.slider("Number of stages", 10, 35, 20)
    max_nf  = max(5, inp["N_Stages"] - 4)
    inp["N_Feed"] = st.sidebar.slider("Feed stage location", 4, max_nf, min(10, inp["N_Stages"]//2))
    inp["Reflux_ratio"] = st.sidebar.slider("Reflux ratio", 1.5, 6.0, 2.5,  0.1)
    inp["B_fraction"] = st.sidebar.slider("Bottoms fraction [-]", 0.20,  0.80,  0.50, 0.01)
    inp["q_feed"] = st.sidebar.slider("Feed quality q [-]", 0.0, 1.5, 1.0, 0.05)
 
    st.sidebar.markdown("---")
    available = ["XGBoost", "Polynomial", "Random Forest", "ANN"]
    model_name = st.sidebar.selectbox("Active model", available)
    return inp, model_name
 
 

def page_predictor(inputs, model_name, registry, scaler_X, feat_names):
    st.title("Live column predictor")
    st.caption("Adjust the sidebar sliders — predictions update instantly.")
 
    if model_name not in registry.model_names:
        st.error(f"'{model_name}' is not loaded. Run train_models.py first.")
        return
 
    X_row  = build_feature_row(inputs, feat_names)
    X_s    = scaler_X.transform(X_row)
    y_pred = registry.predict(model_name, X_s)[0]
    xD, xB, QC, QR = y_pred
 
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distillate purity xD", f"{xD:.4f} mol/mol")
    c2.metric("Bottoms purity xB",    f"{xB:.4f} mol/mol")
    c3.metric("Condenser duty QC",    f"{QC:.1f} kW")
    c4.metric("Reboiler duty QR",     f"{QR:.1f} kW")
 
    st.markdown("---")
 

    st.subheader("Purity gauges")
    g1, g2 = st.columns(2)
    for col_ui, val, label in [(g1, xD, "Distillate purity xD"),
                                (g2, xB, "Bottoms purity xB")]:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(float(val), 4),
            title={"text": label, "font": {"size": 14}},
            number={"font": {"size": 28}},
            gauge={
                "axis":  {"range": [0, 1]},
                "bar":   {"color": "#378ADD"},
                "steps": [
                    {"range": [0,    0.5],  "color": "#f0f0f0"},
                    {"range": [0.5,  0.8],  "color": "#d4ebd4"},
                    {"range": [0.8,  0.95], "color": "#a8d5a8"},
                    {"range": [0.95, 1.0],  "color": "#5dcaa5"},
                ],
                "threshold": {"line": {"color": "#D85A30", "width": 3},
                              "thickness": 0.75, "value": 0.95},
            }
        ))
        fig.update_layout(height=240, margin=dict(t=40, b=10, l=20, r=20))
        col_ui.plotly_chart(fig, use_container_width=True)
 

    st.subheader("Energy duties")
    fig_e = go.Figure()
    fig_e.add_bar(
        x=["Condenser QC", "Reboiler QR"],
        y=[round(float(QC), 1), round(float(QR), 1)],
        marker_color=["#378ADD", "#D85A30"],
        text=[f"{QC:.1f} kW", f"{QR:.1f} kW"],
        textposition="outside",
    )
    fig_e.update_layout(yaxis_title="Duty [kW]", height=300,
                        showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_e, use_container_width=True)
    st.caption(f"Model: {model_name} | Feed: 100 kmol/h | Benzene–Toluene | PR EOS")
 
 
def page_trend_sweep(inputs, model_name, registry, scaler_X, feat_names):
    st.title("Physical trend analysis")
    st.markdown(
        "Sweep one variable across its full range while holding all others "
        "fixed at the sidebar values. Confirms the surrogate respects "
        "expected thermodynamic monotonicity."
    )
 
    if model_name not in registry.model_names:
        st.error(f"'{model_name}' not loaded.")
        return
 
    sweep_var = st.selectbox(
        "Variable to sweep",
        ["reflux_ratio", "N_stages", "z_feed", "T_feed_C", "P_feed_atm", "B_fraction"],
        format_func=lambda x: FEATURE_LABELS.get(x, x),
    )
 
    ranges = {
        "reflux_ratio": (1.5, 6.0),
        "N_stages":     (10, 35),
        "z_feed":       (0.20, 0.80),
        "T_feed_C":     (60, 120),
        "P_feed_atm":   (1.0, 3.0),
        "B_fraction":   (0.20, 0.80),
    }
    lo, hi = ranges[sweep_var]
    vals   = np.linspace(lo, hi, 80)
 
    rows = []
    for v in vals:
        inp_copy = dict(inputs)
        inp_copy[sweep_var] = v
        if sweep_var == "N_stages":
            inp_copy["N_stages"] = int(round(v))
            inp_copy["N_feed"]   = max(4, min(inp_copy["N_feed"], int(round(v)) - 4))
        rows.append(build_feature_row(inp_copy, feat_names)[0])
 
    X_sw   = scaler_X.transform(np.array(rows))
    y_sw   = registry.predict(model_name, X_sw)
 
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[TARGET_LABELS[t] for t in TARGET_NAMES])
    for i, (tname, pos) in enumerate(
        zip(TARGET_NAMES, [(1,1),(1,2),(2,1),(2,2)])
    ):
        fig.add_trace(
            go.Scatter(x=vals, y=y_sw[:, i], mode="lines",
                       line=dict(width=2.5, color=list(COLORS.values())[i]),
                       name=TARGET_LABELS[tname]),
            row=pos[0], col=pos[1],
        )
        fig.update_xaxes(title_text=FEATURE_LABELS.get(sweep_var, sweep_var),
                         row=pos[0], col=pos[1])
        fig.update_yaxes(title_text=TARGET_UNITS.get(tname, ""),
                         row=pos[0], col=pos[1])
 
    fig.update_layout(height=560, showlegend=False,
                      title_text=f"Response to: {FEATURE_LABELS.get(sweep_var, sweep_var)}",
                      margin=dict(t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)
 
    # Monotonicity check table
    expected = {
        "reflux_ratio": {"xD":"↑","xB":"↓","QC_kW":"↑(abs)","QR_kW":"↑"},
        "N_stages":     {"xD":"↑","xB":"↓"},
        "z_feed":       {"xD":"↑","xB":"↑"},
        "B_fraction":   {"xB":"↑"},
    }
    exp = expected.get(sweep_var, {})
    if exp:
        st.subheader("Monotonicity check")
        chk = []
        for i, tname in enumerate(TARGET_NAMES):
            if tname not in exp:
                continue
            diff = np.diff(y_sw[:, i])
            direction = exp[tname]
            ok = (np.all(diff >= -1e-3) if "↑" in direction
                  else np.all(diff <= 1e-3))
            chk.append({"Output": TARGET_LABELS[tname],
                         "Expected": direction,
                         "Result": "✅ Pass" if ok else "❌ Fail"})
        st.dataframe(pd.DataFrame(chk), use_container_width=True, hide_index=True)
 
 
def page_model_comparison(registry):
    st.title("Model performance comparison")
 
    metrics_df = load_metrics()
    if metrics_df is None:
        st.warning("Run evaluate.py first to generate metrics.")
        return
 
    st.subheader("Test-set R²")
    pivot_r2 = metrics_df.pivot(index="target", columns="model", values="R2").round(5)
    st.dataframe(pivot_r2.style.background_gradient(cmap="YlGn", axis=None),
                 use_container_width=True)
 
    st.subheader("Test-set RMSE")
    pivot_rmse = metrics_df.pivot(index="target", columns="model", values="RMSE").round(5)
    st.dataframe(pivot_rmse.style.background_gradient(cmap="YlOrRd_r", axis=None),
                 use_container_width=True)
 
    st.subheader("Mean R² per model")
    mean_r2 = metrics_df.groupby("model")["R2"].mean().reset_index()
    mean_r2.columns = ["Model", "Mean R²"]
    mean_r2 = mean_r2.sort_values("Mean R²", ascending=False)
 
    fig = go.Figure()
    for _, row in mean_r2.iterrows():
        fig.add_bar(
            x=[row["Model"]], y=[row["Mean R²"]],
            marker_color=COLORS.get(row["Model"], "#888780"),
            text=[f"{row['Mean R²']:.5f}"], textposition="outside",
        )
    fig.update_layout(
        yaxis=dict(range=[max(0, mean_r2["Mean R²"].min()-0.02), 1.005],
                   title="Mean R²"),
        height=350, showlegend=False, margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
 
    # Radar chart
    st.subheader("R² radar — per target per model")
    tlist = list(TARGET_LABELS.keys())
    fig_r = go.Figure()
    for mname in metrics_df["model"].unique():
        sub  = metrics_df[metrics_df["model"] == mname].set_index("target")
        vals = [sub.loc[t, "R2"] if t in sub.index else 0.0 for t in tlist]
        vals += vals[:1]
        fig_r.add_trace(go.Scatterpolar(
            r=vals,
            theta=[TARGET_LABELS[t] for t in tlist] + [TARGET_LABELS[tlist[0]]],
            name=mname,
            line=dict(color=COLORS.get(mname, "#888780"), width=2),
            fill="toself", opacity=0.25,
        ))
    fig_r.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.9, 1.0])),
        height=420, margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig_r, use_container_width=True)
 
 
def page_dataset_explorer():
    st.title("Dataset explorer")
 
    try:
        df = load_dataset()
    except FileNotFoundError:
        st.error("dataset.csv not found. Run automation.py first.")
        return
 
    st.markdown(f"**{len(df)} rows · {len(df.columns)} columns**")
 
    ca, cb = st.columns(2)
    z_range  = ca.slider("Filter: z_feed",
                          float(df["z_feed"].min()), float(df["z_feed"].max()),
                          (float(df["z_feed"].min()), float(df["z_feed"].max())), 0.01)
    rr_range = cb.slider("Filter: reflux ratio",
                          float(df["reflux_ratio"].min()), float(df["reflux_ratio"].max()),
                          (float(df["reflux_ratio"].min()), float(df["reflux_ratio"].max())), 0.1)
 
    filt = df[df["z_feed"].between(*z_range) & df["reflux_ratio"].between(*rr_range)]
    st.markdown(f"Showing **{len(filt)}** rows after filter")
    st.dataframe(filt.round(5), use_container_width=True, height=300)
 
    st.subheader("Scatter plot")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    c1, c2, c3 = st.columns(3)
    x_col = c1.selectbox("X axis",  num_cols, index=num_cols.index("reflux_ratio") if "reflux_ratio" in num_cols else 0)
    y_col = c2.selectbox("Y axis",  num_cols, index=num_cols.index("xD") if "xD" in num_cols else 1)
    col_col = c3.selectbox("Colour",  num_cols, index=num_cols.index("z_feed") if "z_feed" in num_cols else 2)
    fig_sc  = px.scatter(filt, x=x_col, y=y_col, color=col_col,
                          color_continuous_scale="viridis", opacity=0.7, height=400,
                          labels={x_col: FEATURE_LABELS.get(x_col, x_col),
                                  y_col: FEATURE_LABELS.get(y_col, y_col)})
    fig_sc.update_traces(marker=dict(size=5))
    st.plotly_chart(fig_sc, use_container_width=True)
 
    st.subheader("Correlation heatmap")
    corr_cols = [c for c in ["T_feed_C","P_feed_atm","z_feed","N_stages",
                              "N_feed","reflux_ratio","B_fraction",
                              "xD","xB","QC_kW","QR_kW"] if c in filt.columns]
    corr = filt[corr_cols].corr().round(3)
    fig_h = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                       text_auto=".2f", height=480, labels={"color": "Correlation"})
    fig_h.update_xaxes(tickangle=45)
    st.plotly_chart(fig_h, use_container_width=True)
 
 
def page_shap():
    st.title("SHAP feature importance")
    for path, caption in [
        ("plots/shap_summary_xD.png",  "SHAP summary — XGBoost — target: xD"),
        ("plots/shap_all_targets.png", "SHAP mean |value| — all targets"),
        ("plots/feature_importance.png","Feature importances — RF and XGBoost"),
    ]:
        if os.path.exists(path):
            st.subheader(caption)
            st.image(path, use_column_width=True)
        else:
            st.info(f"{path} not found. Run evaluate.py to generate it.")
 
    st.markdown("""
**How to read the SHAP beeswarm plot:**
- Each dot = one test sample. Position on x = impact on prediction.
- Red = high feature value, blue = low feature value.
- Features sorted by mean absolute impact (most important at top).
 
**Expected findings:**
- `reflux_ratio` dominates `xD` — the primary lever for purity.
- `z_feed` and `N_stages` are second and third most important.
- `T_feed_C` and `P_feed_atm` act through relative volatility (α).
""")
 
 
def page_about():
    st.title("About this project")
    st.markdown("""
### Surrogate modelling of a binary distillation column
 
**System:** Benzene–Toluene | **EOS:** Peng–Robinson | **Simulator:** DWSIM
 
---
 
### Model architecture (`models_code/` package)
 
| File | Role |
|---|---|
| `base.py` | Shared constants, `ModelResult` dataclass, metric utilities |
| `polynomial_model.py` | Polynomial Regression with Ridge — trains, saves, loads |
| `random_forest_model.py` | Random Forest with RandomizedSearchCV — trains, saves, loads |
| `xgboost_model.py` | XGBoost per-target with early stopping — trains, saves, loads |
| `ann_model.py` | Keras MLP [128→64→32] — trains, saves, loads |
| `model_registry.py` | Orchestrator — owns all models, unified predict/evaluate API |
 
### How to run
 
```bash
pip install -r requirements.txt -r requirements_app.txt
python run_pipeline.py        # generates data + trains all models
streamlit run app.py          # launches this app
```
 
**Skip a model during training:**
```bash
python train_models.py --skip ANN
```
 
**Docker:**
```bash
docker-compose up
```
""")
 

 
def main():
    inputs, model_name = sidebar_inputs()
 
    page = st.sidebar.radio("Navigation", [
        "Live predictor",
        "Trend sweep",
        "Model comparison",
        "Dataset explorer",
        "SHAP analysis",
        "About",
    ])
 
    if not assets_ready():
        st.warning("Pipeline not yet run. Execute `python run_pipeline.py` first.")
        if page == "About":
            page_about()
        return
 
    # Load shared assets
    registry   = load_registry()
    scaler_X, scaler_y = load_scalers()
    feat_names = load_feature_names()
 
    if page == "Live predictor":
        page_predictor(inputs, model_name, registry, scaler_X, feat_names)
    elif page == "Trend sweep":
        page_trend_sweep(inputs, model_name, registry, scaler_X, feat_names)
    elif page == "Model comparison":
        page_model_comparison(registry)
    elif page == "Dataset explorer":
        page_dataset_explorer()
    elif page == "SHAP analysis":
        page_shap()
    elif page == "About":
        page_about()
 
 
if __name__ == "__main__":
    main()
