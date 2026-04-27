import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import PAGE_CSS, PLOT_CFG, preprocess_data, run_regression, show_regression_metrics, show_regression_plots, show_econometric_tests

st.set_page_config(page_title="OLS — Regresie Liniară", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.markdown('<div class="ph"><div class="label">01 / REGRESIE</div><h1>OLS — Regresie Liniară Multiplă</h1><div class="sub">Baza de comparație · Teste econometrice complete: Durbin-Watson, Breusch-Pagan, VIF, Jarque-Bera, Shapiro-Wilk</div></div>', unsafe_allow_html=True)

data = preprocess_data()
df = data["df_scaled"]; feat = data["feature_cols"]
X = df[feat].values; y = df[data["target_reg"]].values

st.sidebar.markdown("## Configurare OLS")
test_size = st.sidebar.slider("Proporție test", 0.10, 0.40, 0.20, 0.05)
rand_state = st.sidebar.number_input("Random state", 0, 999, 42)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=rand_state)
res = run_regression(X_train,X_test,y_train,y_test,LinearRegression(),"OLS","#3b82f6")

st.markdown('<div class="sec">Metrici de Performanță — R², RMSE, MAE, MSE</div>', unsafe_allow_html=True)
show_regression_metrics(res)

cv_scores = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2")


st.markdown('<div class="sec">Grafice Diagnostice</div>', unsafe_allow_html=True)
show_regression_plots(res)

show_econometric_tests(X_test, y_test, res["y_pred_test"], feat)

st.markdown('<div class="sec">Summary OLS Complet — statsmodels</div>', unsafe_allow_html=True)
X_sm = sm.add_constant(X_test)
ols_sm = sm.OLS(y_test, X_sm).fit()
st.text(str(ols_sm.summary()))

st.markdown('<div class="sec">Coeficienți OLS</div>', unsafe_allow_html=True)
coef_df = pd.DataFrame({"Feature": feat, "Coeficient": res["model"].coef_,
                          "Abs(Coef)": np.abs(res["model"].coef_)}).sort_values("Abs(Coef)", ascending=False)
import plotly.express as px
fig = px.bar(coef_df, x="Coeficient", y="Feature", orientation="h", color="Coeficient",
             color_continuous_scale=["#ef4444","#94a3b8","#3b82f6"], color_continuous_midpoint=0, height=400)
fig.update_layout(**PLOT_CFG); st.plotly_chart(fig, use_container_width=True)
