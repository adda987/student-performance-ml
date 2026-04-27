import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, cross_val_score
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import PAGE_CSS, PLOT_CFG, preprocess_data, run_classification, show_clf_metrics, show_confusion_matrix, show_clf_plots

st.set_page_config(page_title="Decision Tree", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.markdown("""
<div class="ph" style="border-left-color:#14b8a6;">
    <div class="label" style="color:#0d9488;">10 / CLASIFICARE</div>
    <h1>Decision Tree Classifier</h1>
</div>""", unsafe_allow_html=True)


data = preprocess_data()
target_names = data["target_names"]
df = data["df_scaled"]
feat = data["feature_cols"]
X = df[feat].values
y = df[data["target_clf"]].values

st.sidebar.markdown("## Configurare Decision Tree")
test_size   = st.sidebar.slider("Proporție test", 0.10, 0.40, 0.20, 0.05)
rand_state  = st.sidebar.number_input("Random state", 0, 999, 42)
max_depth   = st.sidebar.select_slider("Max depth", [2, 3, 4, 5, 7, 10, 15, None], value=5)
criterion   = st.sidebar.selectbox("Criteriu split", ["gini", "entropy", "log_loss"], index=0)
min_samples_split = st.sidebar.slider("min_samples_split", 2, 100, 20)
min_samples_leaf  = st.sidebar.slider("min_samples_leaf", 1, 50, 10)
class_weight = st.sidebar.selectbox("Class weight", ["balanced", None], index=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                      random_state=rand_state, stratify=y)

model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                class_weight=class_weight, random_state=rand_state)

with st.spinner("Antrenare Decision Tree..."):
    res = run_classification(X_train, X_test, y_train, y_test, model, "Decision Tree", "#14b8a6")

tree_depth = res["model"].get_depth()
n_leaves   = res["model"].get_n_leaves()

c1, c2, c3 = st.columns(3)
for col, val, lbl in [(c1, str(tree_depth), "Adâncime reală"), (c2, str(n_leaves), "Nr. frunze"), (c3, criterion, "Criteriu")]:
    col.markdown(f'<div class="metric-card"><div class="val" style="color:#14b8a6;">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="sec">Metrici de Clasificare</div>', unsafe_allow_html=True)
show_clf_metrics(res)


show_confusion_matrix(res, target_names)

st.markdown('<div class="sec">Grafice Predicții</div>', unsafe_allow_html=True)
show_clf_plots(res, target_names)

# Feature Importance
st.markdown('<div class="sec">Feature Importance (Gini Impurity)</div>', unsafe_allow_html=True)
fi = pd.DataFrame({"Feature": feat, "Importance": res["model"].feature_importances_}).sort_values("Importance", ascending=True)
fig = px.bar(fi, x="Importance", y="Feature", orientation="h", color="Importance",
             color_continuous_scale=["#ccfbf1", "#14b8a6"], title="Feature Importance — Decision Tree",
             height=max(300, len(feat)*32))
fig.update_layout(**PLOT_CFG, coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)

# Overfitting analysis: depth vs accuracy
st.markdown('<div class="sec">Analiza Overfitting — Adâncime vs Performanță</div>', unsafe_allow_html=True)
depths = list(range(1, 16))
tr_acc, te_acc = [], []
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, criterion=criterion,
                                 class_weight=class_weight, random_state=rand_state)
    dt.fit(X_train, y_train)
    tr_acc.append(dt.score(X_train, y_train))
    te_acc.append(dt.score(X_test, y_test))

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=depths, y=tr_acc, mode="lines+markers", name="Train Accuracy", line=dict(color="#14b8a6")))
fig2.add_trace(go.Scatter(x=depths, y=te_acc, mode="lines+markers", name="Test Accuracy", line=dict(color="#3b82f6")))
if max_depth:
    fig2.add_vline(x=max_depth, line_dash="dash", line_color="#ef4444",
                    annotation_text=f"max_depth={max_depth}")
fig2.update_layout(title="Acuratețe vs Adâncime (Overfitting)", xaxis_title="Max Depth",
                    yaxis_title="Accuracy", **PLOT_CFG, height=350)
st.plotly_chart(fig2, use_container_width=True)

st.markdown('<div class="sec">Reguli Arbore (primele 3 nivele)</div>', unsafe_allow_html=True)
small_tree = DecisionTreeClassifier(max_depth=3, criterion=criterion,
                                     class_weight=class_weight, random_state=rand_state)
small_tree.fit(X_train, y_train)
tree_rules = export_text(small_tree, feature_names=feat)
st.code(tree_rules, language="")

if "clf_results" not in st.session_state:
    st.session_state["clf_results"] = {}
st.session_state["clf_results"]["Decision Tree"] = {
    "Acuratețe": res["accuracy"], "F1-Score": res["f1"],
    "Precizie": res["precision"], "Recall": res["recall"]
}
