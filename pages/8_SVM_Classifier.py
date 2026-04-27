import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import PAGE_CSS, PLOT_CFG, preprocess_data, run_classification, show_clf_metrics, show_confusion_matrix, show_clf_plots

st.set_page_config(page_title="SVM Classifier", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.markdown("""
<div class="ph" style="border-left-color:#0ea5e9;">
    <div class="label" style="color:#0284c7;">08 / CLASIFICARE</div>
    <h1>SVM Classifier (Support Vector Machine)</h1>
</div>""", unsafe_allow_html=True)


data = preprocess_data()
target_names = data["target_names"]
df = data["df_scaled"]
feat = data["feature_cols"]
X = df[feat].values
y = df[data["target_clf"]].values

st.sidebar.markdown("##  Configurare SVM")
test_size  = st.sidebar.slider("Proporție test", 0.10, 0.40, 0.20, 0.05)
rand_state = st.sidebar.number_input("Random state", 0, 999, 42)
C          = st.sidebar.select_slider("C (regularizare)", [0.01, 0.1, 0.5, 1, 5, 10, 50, 100], value=1)
kernel     = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
gamma      = st.sidebar.selectbox("Gamma", ["scale", "auto"], index=0)
class_weight = st.sidebar.selectbox("Class weight", ["balanced", None], index=0)

# SVM is slow on large datasets — use a sample
n_total = len(X)
max_svm = 8000
use_sample = n_total > max_svm

if use_sample:
    st.warning(f"SVM e lent pe {n_total:,} înregistrări. Se utilizează un sample de {max_svm:,} pentru antrenament.")
    rng = np.random.default_rng(rand_state)
    idx = rng.choice(n_total, max_svm, replace=False)
    X_s, y_s = X[idx], y[idx]
else:
    X_s, y_s = X, y

X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=test_size,
                                                      random_state=rand_state, stratify=y_s)

model = SVC(kernel=kernel, C=C, gamma=gamma, class_weight=class_weight,
             probability=True, random_state=rand_state)

with st.spinner(f"Antrenare SVM (kernel={kernel}, C={C})... poate dura 30-60 sec."):
    res = run_classification(X_train, X_test, y_train, y_test, model, f"SVM ({kernel})", "#0ea5e9")

st.markdown('<div class="sec">Metrici de Clasificare</div>', unsafe_allow_html=True)
show_clf_metrics(res)

st.markdown('<div class="sec">Distribuția Claselor Prezise</div>', unsafe_allow_html=True)
show_clf_plots(res, target_names)

show_confusion_matrix(res, target_names)

# Decision boundary info
c1, c2, c3 = st.columns(3)
for col, lbl, val in [
    (c1, "Kernel", kernel),
    (c2, "C (regularizare)", str(C)),
    (c3, "Support Vectors", str(sum(model.n_support_)))
]:
    col.markdown(f'<div class="metric-card"><div class="val" style="color:#0ea5e9;">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

# Save for comparison
if "clf_results" not in st.session_state:
    st.session_state["clf_results"] = {}
st.session_state["clf_results"]["SVM"] = {
    "Acuratețe": res["accuracy"], "F1-Score": res["f1"],
    "Precizie": res["precision"], "Recall": res["recall"]
}
