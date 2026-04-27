import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import PAGE_CSS, PLOT_CFG, preprocess_data, run_classification, show_clf_metrics, show_confusion_matrix, show_clf_plots

st.set_page_config(page_title="Logistic Regression", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.markdown("""
<div class="ph" style="border-left-color:#ec4899;">
    <div class="label" style="color:#db2777;">09 / CLASIFICARE</div>
    <h1>Logistic Regression</h1>
</div>""", unsafe_allow_html=True)


data = preprocess_data()
target_names = data["target_names"]
df = data["df_scaled"]
feat = data["feature_cols"]
X = df[feat].values
y = df[data["target_clf"]].values

st.sidebar.markdown("##  Configurare Logistic Regression")
test_size  = st.sidebar.slider("Proporție test", 0.10, 0.40, 0.20, 0.05)
rand_state = st.sidebar.number_input("Random state", 0, 999, 42)
C          = st.sidebar.select_slider("C (invers regularizare)", [0.001,0.01,0.1,0.5,1,5,10,100], value=1)
penalty    = st.sidebar.selectbox("Penalizare", ["l2", "l1", "elasticnet", "none"], index=0)
solver     = "saga" if penalty in ["l1","elasticnet"] else "lbfgs"
class_weight = st.sidebar.selectbox("Class weight", ["balanced", None], index=0)
max_iter   = st.sidebar.slider("Max iterații", 100, 2000, 500, 100)

l1_ratio = None
if penalty == "elasticnet":
    l1_ratio = st.sidebar.slider("L1 ratio (Elastic Net)", 0.0, 1.0, 0.5, 0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                      random_state=rand_state, stratify=y)

model_kwargs = dict(C=C, penalty=penalty, solver=solver, class_weight=class_weight,
                    max_iter=max_iter, random_state=rand_state, multi_class="auto")
if penalty == "elasticnet":
    model_kwargs["l1_ratio"] = l1_ratio
if penalty == "none":
    model_kwargs["penalty"] = None

model = LogisticRegression(**model_kwargs)

with st.spinner("Antrenare Logistic Regression..."):
    res = run_classification(X_train, X_test, y_train, y_test, model, "Logistic Regression", "#ec4899")

st.markdown('<div class="sec">Metrici de Clasificare</div>', unsafe_allow_html=True)
show_clf_metrics(res)

cv = cross_val_score(LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter,
                                         random_state=rand_state, solver=solver),
                     X, y, cv=5, scoring="f1_weighted", n_jobs=-1)

show_confusion_matrix(res, target_names)

st.markdown('<div class="sec">Grafice Predicții</div>', unsafe_allow_html=True)
show_clf_plots(res, target_names)

# Coefficients — one per class (OvR)
st.markdown('<div class="sec">Coeficienți Logistici per Clasă</div>', unsafe_allow_html=True)
st.markdown('<div class="info-box" style="font-size:13px;">Coeficienții reprezintă impactul fiecărei variabile în <strong>log-odds</strong>. Un coeficient pozitiv crește probabilitatea clasei respective.</div>', unsafe_allow_html=True)

coef_matrix = res["model"].coef_  # shape: (n_classes, n_features)
tab_labels = [f"Clasa: {c}" for c in target_names]
tabs = st.tabs(tab_labels)
for i, (tab, cls) in enumerate(zip(tabs, target_names)):
    with tab:
        coef_df = pd.DataFrame({"Feature": feat, "Coeficient": coef_matrix[i]}).sort_values("Coeficient")
        fig = px.bar(coef_df, x="Coeficient", y="Feature", orientation="h", color="Coeficient",
                     color_continuous_scale=["#ef4444","#94a3b8","#ec4899"], color_continuous_midpoint=0,
                     title=f"Coeficienți — {cls}", height=max(300, len(feat)*28))
        fig.update_layout(**PLOT_CFG, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# Save for comparison
if "clf_results" not in st.session_state:
    st.session_state["clf_results"] = {}
st.session_state["clf_results"]["Logistic Reg."] = {
    "Acuratețe": res["accuracy"], "F1-Score": res["f1"],
    "Precizie": res["precision"], "Recall": res["recall"]
}
