import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import PAGE_CSS, PLOT_CFG, preprocess_data, run_classification, show_clf_metrics, show_confusion_matrix, show_clf_plots

st.set_page_config(page_title="KNN Classifier", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.markdown("""
<div class="ph" style="border-left-color:#a855f7;">
    <div class="label" style="color:#9333ea;">11 / CLASIFICARE</div>
    <h1>KNN — K Nearest Neighbors Classifier</h1>
</div>""", unsafe_allow_html=True)

data = preprocess_data()
target_names = data["target_names"]
df = data["df_scaled"]
feat = data["feature_cols"]
X = df[feat].values
y = df[data["target_clf"]].values

st.sidebar.markdown("## Configurare KNN")
test_size  = st.sidebar.slider("Proporție test", 0.10, 0.40, 0.20, 0.05)
rand_state = st.sidebar.number_input("Random state", 0, 999, 42)
k_val      = st.sidebar.slider("K (nr. vecini)", 1, 30, 7)
metric     = st.sidebar.selectbox("Metrică distanță", ["euclidean","manhattan","chebyshev","minkowski"], index=0)
weights    = st.sidebar.selectbox("Ponderare", ["uniform","distance"], index=0)
algorithm  = st.sidebar.selectbox("Algoritm", ["auto","ball_tree","kd_tree","brute"], index=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                      random_state=rand_state, stratify=y)

model = KNeighborsClassifier(n_neighbors=k_val, metric=metric, weights=weights, algorithm=algorithm, n_jobs=-1)

with st.spinner(f"Antrenare KNN (K={k_val}, metric={metric})..."):
    res = run_classification(X_train, X_test, y_train, y_test, model, f"KNN (K={k_val})", "#a855f7")

st.markdown('<div class="sec">Metrici de Clasificare</div>', unsafe_allow_html=True)
show_clf_metrics(res)


show_confusion_matrix(res, target_names)

st.markdown('<div class="sec">Grafice Predicții</div>', unsafe_allow_html=True)
show_clf_plots(res, target_names)

# K vs Performance curve — cel mai important grafic pentru KNN
st.markdown('<div class="sec">Alegerea K Optim — K vs F1 Score</div>', unsafe_allow_html=True)
k_range = list(range(1, 31))
f1_train, f1_test = [], []

with st.spinner("Calculare K optim (1–30)..."):
    from sklearn.metrics import f1_score
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights, n_jobs=-1)
        knn.fit(X_train, y_train)
        f1_train.append(f1_score(y_train, knn.predict(X_train), average="weighted", zero_division=0))
        f1_test.append(f1_score(y_test,  knn.predict(X_test),  average="weighted", zero_division=0))

best_k = k_range[np.argmax(f1_test)]
fig = go.Figure()
fig.add_trace(go.Scatter(x=k_range, y=f1_train, mode="lines+markers", name="Train F1", line=dict(color="#a855f7")))
fig.add_trace(go.Scatter(x=k_range, y=f1_test,  mode="lines+markers", name="Test F1",  line=dict(color="#3b82f6")))
fig.add_vline(x=best_k, line_dash="dash", line_color="#ef4444",
               annotation_text=f"K optim={best_k} (F1={max(f1_test):.4f})",
               annotation_position="top right")
fig.add_vline(x=k_val, line_dash="dot", line_color="#f59e0b",
               annotation_text=f"K ales={k_val}", annotation_position="bottom right")
fig.update_layout(title="F1 Weighted vs K — Alegerea numărului optim de vecini",
                   xaxis_title="K (nr. vecini)", yaxis_title="F1 Weighted",
                   **PLOT_CFG, height=400)
st.plotly_chart(fig, use_container_width=True)


# Save for comparison
if "clf_results" not in st.session_state:
    st.session_state["clf_results"] = {}
st.session_state["clf_results"]["KNN"] = {
    "Acuratețe": res["accuracy"], "F1-Score": res["f1"],
    "Precizie": res["precision"], "Recall": res["recall"]
}
