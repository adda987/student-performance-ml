"""
pages/4_RF_Regressor.py — Random Forest Regressor cu feature importance
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import (PAGE_CSS, PLOT_CFG,
                   preprocess_data, run_regression, show_regression_metrics, show_regression_plots)

st.set_page_config(page_title="Random Forest Regressor", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.markdown("""
<div class="ph" style="border-left-color:#f59e0b;">
    <div class="label" style="color:#f59e0b;">04 / REGRESIE</div>
    <h1>Random Forest Regressor</h1>
</div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.markdown("##  Configurare Random Forest")
test_size = st.sidebar.slider("Proporție test set", 0.10, 0.40, 0.20, 0.05)
rand_state = st.sidebar.number_input("Random state", 0, 999, 42)
st.sidebar.markdown("---")
n_estimators = st.sidebar.slider("Nr. arbori (n_estimators)", 50, 500, 200, 50)
max_depth = st.sidebar.select_slider("Max depth", [None, 5, 10, 15, 20, 30], value=None)
min_samples_split = st.sidebar.slider("Min samples split", 2, 20, 2)
max_features = st.sidebar.selectbox("Max features", ["sqrt", "log2", None], index=0)

# ── Data ─────────────────────────────────────────────────────────
data = preprocess_data()
df = data["df_scaled"]
feat = data["feature_cols"]
X = df[feat].values
y = df[data["target_reg"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)

model = RandomForestRegressor(
    n_estimators=n_estimators, max_depth=max_depth,
    min_samples_split=min_samples_split, max_features=max_features,
    random_state=rand_state, n_jobs=-1
)

with st.spinner("Se antrenează Random Forest..."):
    res = run_regression(X_train, X_test, y_train, y_test, model, "Random Forest Regressor", "#f59e0b")

# ── Metrici ───────────────────────────────────────────────────────
st.markdown('<div class="sec">Metrici de performanță</div>', unsafe_allow_html=True)
show_regression_metrics(res)


# ── Grafice ───────────────────────────────────────────────────────
st.markdown('<div class="sec">Grafice diagnostice</div>', unsafe_allow_html=True)
show_regression_plots(res)
# ── Grafice suplimentare Random Forest ────────────────────────────
st.markdown('<div class="sec">Analiză suplimentară Random Forest</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    importances = res["model"].feature_importances_

    imp_df = pd.DataFrame({
        "feature": feat,
        "importance": importances
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(imp_df["feature"], imp_df["importance"], color="#f59e0b")
    ax.set_title("Importanța variabilelor", fontsize=11, fontweight="bold")
    ax.set_xlabel("Importanță", fontsize=9)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()

    st.pyplot(fig)

with c2:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_test, res["y_pred_test"], alpha=0.45, color="#2563eb", edgecolors="none")

    min_val = min(y_test.min(), res["y_pred_test"].min())
    max_val = max(y_test.max(), res["y_pred_test"].max())

    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#ef4444", linewidth=2)

    ax.set_title("Predicții vs valori reale", fontsize=11, fontweight="bold")
    ax.set_xlabel("Valori reale", fontsize=9)
    ax.set_ylabel("Valori prezise", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()

    st.pyplot(fig)

st.markdown("")

depths = [2, 4, 6, 8, 10, 15, 20]
scores = []

for d in depths:
    m = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=d,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=rand_state,
        n_jobs=-1
    )
    m.fit(X_train, y_train)
    scores.append(m.score(X_test, y_test))

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(depths, scores, marker="o", color="#10b981", linewidth=2)
ax.fill_between(depths, scores, alpha=0.15, color="#10b981")

ax.set_title("Performanță în funcție de adâncimea arborilor", fontsize=11, fontweight="bold")
ax.set_xlabel("Max Depth", fontsize=9)
ax.set_ylabel("R²", fontsize=9)
ax.tick_params(axis="both", labelsize=8)
fig.tight_layout()

st.pyplot(fig)