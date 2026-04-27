import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import PAGE_CSS, PLOT_CFG, preprocess_data, run_regression, show_regression_metrics, show_regression_plots, show_econometric_tests

st.set_page_config(page_title="Lasso Regression", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.markdown('<div class="ph" style="border-left-color:#8b5cf6;"><div class="label" style="color:#8b5cf6;">03 / REGRESIE</div><h1>Lasso Regression </h1><div class="sub"></div></div>', unsafe_allow_html=True)

data = preprocess_data()
df = data["df_scaled"]; feat = data["feature_cols"]
X = df[feat].values; y = df[data["target_reg"]].values

st.sidebar.markdown("##  Configurare Lasso")
test_size = st.sidebar.slider("Proporție test", 0.10, 0.40, 0.20, 0.05)
rand_state = st.sidebar.number_input("Random state", 0, 999, 42)
auto_alpha = st.sidebar.checkbox("Alpha optim automat (LassoCV)", value=True)
if not auto_alpha:
    alpha_val = st.sidebar.select_slider("Alpha", [0.0001,0.001,0.01,0.1,0.5,1.0,5.0,10.0], value=0.1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=rand_state)

if auto_alpha:
    with st.spinner("LassoCV: căutare alpha optim..."):
        lasso_cv = LassoCV(cv=5, max_iter=5000, random_state=rand_state)
        lasso_cv.fit(X_train, y_train)
        best_alpha = lasso_cv.alpha_
    model = Lasso(alpha=best_alpha, max_iter=5000)
else:
    best_alpha = alpha_val
    model = Lasso(alpha=best_alpha, max_iter=5000)

res = run_regression(X_train,X_test,y_train,y_test,model,"Lasso","#8b5cf6")

st.markdown('<div class="sec">Metrici de Performanță</div>', unsafe_allow_html=True)
show_regression_metrics(res)

cv = cross_val_score(Lasso(alpha=best_alpha,max_iter=5000), X, y, cv=5, scoring="r2")

n_zero = int(np.sum(np.abs(res["model"].coef_) < 1e-10))
n_nonzero = len(feat) - n_zero

st.markdown('<div class="sec">Grafice Diagnostice</div>', unsafe_allow_html=True)
show_regression_plots(res)
show_econometric_tests(X_test, y_test, res["y_pred_test"], feat)

st.markdown('<div class="sec">Coeficienți Lasso (features zerorizate în roșu)</div>', unsafe_allow_html=True)
coef_df = pd.DataFrame({"Feature":feat,"Coeficient":res["model"].coef_,"Zero":np.abs(res["model"].coef_)<1e-10}).sort_values("Coeficient")
coef_df["Culoare"] = coef_df["Zero"].map({True:"Zerorizat","N/A":"Activ"})
fig = px.bar(coef_df, x="Coeficient", y="Feature", orientation="h", color="Zero",
             color_discrete_map={True:"#ef4444",False:"#8b5cf6"}, height=max(350,len(feat)*30))
fig.update_layout(**PLOT_CFG); st.plotly_chart(fig, use_container_width=True)
