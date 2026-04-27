import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import PAGE_CSS, PLOT_CFG, preprocess_data, run_regression, show_regression_metrics, show_regression_plots, show_econometric_tests

st.set_page_config(page_title="SVR — Support Vector Regression", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.markdown("""
<div class="ph" style="border-left-color:#ec4899;">
    <div class="label" style="color:#ec4899;">05 / REGRESIE</div>
    <h1>SVR — Support Vector Regression</h1>
</div>""", unsafe_allow_html=True)

data = preprocess_data()
df = data["df_scaled"]
feat = data["feature_cols"]
X = df[feat].values
y = df[data["target_reg"]].values

st.sidebar.markdown("## Configurare SVR")
test_size  = st.sidebar.slider("Proporție test", 0.10, 0.40, 0.20, 0.05)
rand_state = st.sidebar.number_input("Random state", 0, 999, 42)
kernel     = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
C_val      = st.sidebar.select_slider("C (regularizare)", [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0], value=10.0)
epsilon    = st.sidebar.select_slider("Epsilon (tub ε)", [0.01, 0.05, 0.1, 0.2, 0.5, 1.0], value=0.1)
gamma      = st.sidebar.selectbox("Gamma", ["scale", "auto"], index=0)
st.sidebar.markdown("---")
sample_size = st.sidebar.slider("Sample size (SVR e lent)", 2000, 15000, 8000, 1000)

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)

# Subsample for speed
rng = np.random.default_rng(rand_state)
n_use = min(sample_size, len(X_train_full))
idx = rng.choice(len(X_train_full), n_use, replace=False)
X_train, y_train = X_train_full[idx], y_train_full[idx]

model = SVR(kernel=kernel, C=C_val, epsilon=epsilon, gamma=gamma)

with st.spinner(f"Antrenare SVR (kernel={kernel}, C={C_val}, ε={epsilon})..."):
    res = run_regression(X_train, X_test, y_train, y_test, model, f"SVR ({kernel})", "#ec4899")

st.markdown('<div class="sec">Metrici de Performanță — R², RMSE, MAE, MSE</div>', unsafe_allow_html=True)
show_regression_metrics(res)

# Support vector info
n_sv = int(model.n_support_[0]) if hasattr(model, "n_support_") else "N/A"
c1, c2, c3 = st.columns(3)
for col, val, lbl in [(c1, str(n_sv), "Support Vectors"), (c2, str(C_val), "C (regularizare)"), (c3, str(epsilon), "Epsilon (ε)")]:
    col.markdown(f'<div class="metric-card"><div class="val" style="color:#ec4899;">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="sec">Grafice Diagnostice</div>', unsafe_allow_html=True)
show_regression_plots(res)
show_econometric_tests(X_test, y_test, res["y_pred_test"], feat)

st.markdown('<div class="sec">Efectul parametrului C asupra performanței SVR</div>', unsafe_allow_html=True)

import matplotlib.pyplot as plt

Cs = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]
r2_vals = []

with st.spinner("Calculare performanță pentru valori diferite ale lui C..."):
    for c in Cs:
        m = SVR(kernel=kernel, C=c, epsilon=epsilon, gamma=gamma)
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        r2_vals.append(r2_score(y_test, pred))

fig, ax = plt.subplots(figsize=(4, 2.5))

# elimină fundal alb
fig.patch.set_alpha(0)
ax.set_facecolor("none")

# linie principală
ax.plot(Cs, r2_vals, marker="o", linewidth=2, color="#ec4899")

# umplere subtilă
ax.fill_between(Cs, r2_vals, alpha=0.15, color="#ec4899")

# linie verticală pentru C ales
ax.axvline(C_val, linestyle="--", linewidth=2, color="#ef4444")

# text discret
ax.text(C_val, max(r2_vals),
        f"C={C_val}",
        color="#ef4444",
        fontsize=7,
        ha="left",
        va="bottom")

# stil axă
ax.set_xscale("log")
ax.set_title("R² Test în funcție de C — SVR", fontsize=8)
ax.set_xlabel("C (log)", fontsize=7)
ax.set_ylabel("R² Test", fontsize=7)

# culori text mai soft (important)
ax.tick_params(axis="both", labelsize=8, colors="#6b7280")
ax.xaxis.label.set_color("#6b7280")
ax.yaxis.label.set_color("#6b7280")
ax.title.set_color("#374151")

# grid subtil
ax.grid(True, linestyle="--", alpha=0.2)

# elimină marginile dure
for spine in ax.spines.values():
    spine.set_visible(False)

fig.tight_layout()

st.pyplot(fig)