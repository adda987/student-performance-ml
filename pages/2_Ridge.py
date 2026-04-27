import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import PAGE_CSS, PLOT_CFG, preprocess_data, run_regression, show_regression_metrics, show_regression_plots, show_econometric_tests

st.set_page_config(page_title="Ridge Regression", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.markdown('<div class="ph" style="border-left-color:#6366f1;"><div class="label" style="color:#6366f1;">02 / REGRESIE</div><h1>Ridge Regression </h1><div class="sub"></div></div>', unsafe_allow_html=True)

data = preprocess_data()
df = data["df_scaled"]; feat = data["feature_cols"]
X = df[feat].values; y = df[data["target_reg"]].values

st.sidebar.markdown("##  Configurare Ridge")
test_size = st.sidebar.slider("Proporție test", 0.10, 0.40, 0.20, 0.05)
rand_state = st.sidebar.number_input("Random state", 0, 999, 42)
auto_alpha = st.sidebar.checkbox("Alpha optim automat (RidgeCV)", value=True)
if auto_alpha:
    alphas_cv = [0.01,0.1,1,10,100,500,1000]
    ridge_cv = RidgeCV(alphas=alphas_cv, cv=5)
    alpha_val = None
else:
    alpha_val = st.sidebar.select_slider("Alpha (λ)", [0.001,0.01,0.1,1,10,100,500,1000], value=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=rand_state)

if auto_alpha:
    ridge_cv.fit(X_train, y_train); best_alpha = ridge_cv.alpha_
    model = Ridge(alpha=best_alpha)
else:
    model = Ridge(alpha=alpha_val)
    best_alpha = alpha_val

res = run_regression(X_train,X_test,y_train,y_test,model,"Ridge","#6366f1")

st.markdown('<div class="sec">Metrici de Performanță</div>', unsafe_allow_html=True)
show_regression_metrics(res)

cv = cross_val_score(Ridge(alpha=best_alpha), X, y, cv=5, scoring="r2")

st.markdown('<div class="sec">Grafice Diagnostice</div>', unsafe_allow_html=True)
show_regression_plots(res)
show_econometric_tests(X_test, y_test, res["y_pred_test"], feat)

st.markdown('<div class="sec">Coeficienți Ridge vs Alpha</div>', unsafe_allow_html=True)

import matplotlib.pyplot as plt

alphas_plot = np.logspace(-3, 3, 100)

coefs = []
for a in alphas_plot:
    ridge_model = Ridge(alpha=a)
    ridge_model.fit(X_train, y_train)
    coefs.append(ridge_model.coef_)

coefs = np.array(coefs)

fig, ax = plt.subplots(figsize=(12, 6))

for i, col in enumerate(feat):
    ax.plot(alphas_plot, coefs[:, i], label=col)

ax.axvline(best_alpha, linestyle="--", linewidth=2)
ax.set_xscale("log")
ax.set_xlabel("Alpha (scară logaritmică)")
ax.set_ylabel("Coeficient")
ax.set_title("Traseul coeficienților Ridge în funcție de alpha")
ax.legend(fontsize=7, loc="best")
fig.tight_layout()

st.pyplot(fig)
