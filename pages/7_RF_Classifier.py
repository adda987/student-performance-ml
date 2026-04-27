import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
import sys, os
import altair as alt
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import PAGE_CSS, PLOT_CFG, preprocess_data, run_classification, show_clf_metrics, show_confusion_matrix, show_clf_plots

st.set_page_config(page_title="Random Forest Classifier", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.markdown("""
<div class="ph" style="border-left-color:#8b5cf6;background:linear-gradient(90deg,#f5f3ff,white);">
    <div class="label" style="color:#7c3aed;">07 / CLASIFICARE</div>
    <h1>Random Forest Classifier</h1>
    <div class="sub">F1-Score · Recall · Precizie · Acuratețe · Matrice confuzie TP/TN/FP/FN · Feature Importance</div>
</div>""", unsafe_allow_html=True)


data = preprocess_data()
target_names = data["target_names"]

st.sidebar.markdown("## Configurare RF Classifier")
test_size    = st.sidebar.slider("Proporție test", 0.10, 0.40, 0.20, 0.05)
rand_state   = st.sidebar.number_input("Random state", 0, 999, 42)
n_estimators = st.sidebar.slider("Nr. arbori", 50, 500, 200, 50)
max_depth    = st.sidebar.select_slider("Max depth", [3, 5, 7, 10, 15, None], value=None)
class_weight = st.sidebar.selectbox("Class weight", ["balanced", None], index=0)
min_leaf     = st.sidebar.slider("min_samples_leaf", 1, 20, 5)

df = data["df_scaled"]
feat = data["feature_cols"]
X = df[feat].values
y = df[data["target_clf"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                      random_state=rand_state, stratify=y)

# Show class distribution
st.markdown('<div class="sec">Distribuția Claselor</div>', unsafe_allow_html=True)
c1, c2 = st.columns([1, 2])
with c1:
    unique, counts = np.unique(y_train, return_counts=True)
    dist_df = pd.DataFrame({
        "Clasă": [target_names[i] for i in unique],
        "Nr. Train": counts,
        "Proporție": [f"{c/len(y_train)*100:.1f}%" for c in counts]
    })
    st.dataframe(dist_df, use_container_width=True, hide_index=True)
with c2:
    fig_d = px.pie(dist_df, names="Clasă", values="Nr. Train", title="Distribuție clase în train set",
                   color_discrete_sequence=px.colors.qualitative.Set2)
    fig_d.update_layout(**PLOT_CFG, height=250)
    st.plotly_chart(fig_d, use_container_width=True)

model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                class_weight=class_weight, min_samples_leaf=min_leaf,
                                random_state=rand_state, n_jobs=-1, oob_score=True)

with st.spinner("Antrenare Random Forest Classifier..."):
    res = run_classification(X_train, X_test, y_train, y_test, model, "Random Forest Classifier", "#8b5cf6")

st.markdown('<div class="sec">Metrici de Clasificare</div>', unsafe_allow_html=True)
show_clf_metrics(res)

cv = cross_val_score(RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                             random_state=rand_state, n_jobs=-1),
                     X, y, cv=5, scoring="f1_weighted")

show_confusion_matrix(res, target_names)

st.markdown('<div class="sec">Grafice Predicții</div>', unsafe_allow_html=True)
show_clf_plots(res, target_names)

# ── Feature Importance + Learning Curve ───────────────────────────
st.markdown('<div class="sec">Analiză model</div>', unsafe_allow_html=True)

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### Importanța variabilelor")

    fi = pd.DataFrame({
        "Feature": feat,
        "Importance": res["model"].feature_importances_
    }).sort_values("Importance", ascending=True)

    fig_fi = px.bar(
        fi,
        x="Importance",
        y="Feature",
        orientation="h",
        color_discrete_sequence=["#8b5cf6"],
        title=None
    )

    fig_fi.update_layout(
        **PLOT_CFG,
        height=360,
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=30)
    )

    st.plotly_chart(fig_fi, use_container_width=True)

with right_col:
    st.markdown("### Learning Curve")

    with st.spinner("Calculare learning curve..."):
        tr_sz, tr_sc, val_sc = learning_curve(
            RandomForestClassifier(
                n_estimators=50,
                class_weight="balanced",
                random_state=rand_state,
                n_jobs=-1
            ),
            X,
            y,
            cv=5,
            scoring="f1_weighted",
            train_sizes=np.linspace(0.1, 1.0, 8),
            n_jobs=-1
        )

    lc_df = pd.DataFrame({
        "n_examples": tr_sz.astype(int),
        "train_f1": tr_sc.mean(axis=1),
        "validation_f1": val_sc.mean(axis=1)
    })

    lc_long = lc_df.melt(
        id_vars="n_examples",
        value_vars=["train_f1", "validation_f1"],
        var_name="set_type",
        value_name="f1_score"
    )

    lc_long["set_type"] = lc_long["set_type"].replace({
        "train_f1": "Train",
        "validation_f1": "Validation"
    })

    chart = (
        alt.Chart(lc_long)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("n_examples:Q", title="Nr. exemple"),
            y=alt.Y("f1_score:Q", title="F1", scale=alt.Scale(domain=[0, 1.05])),
            color=alt.Color(
                "set_type:N",
                scale=alt.Scale(
                    domain=["Train", "Validation"],
                    range=["#8b5cf6", "#3b82f6"]
                ),
                legend=alt.Legend(title=None, orient="top")
            ),
            tooltip=[
                alt.Tooltip("set_type:N", title="Set"),
                alt.Tooltip("n_examples:Q", title="Nr. exemple"),
                alt.Tooltip("f1_score:Q", title="F1", format=".3f")
            ]
        )
        .properties(height=360)
        .configure_axis(
            labelFontSize=11,
            titleFontSize=12,
            labelColor="#6b7280",
            titleColor="#6b7280",
            gridColor="#e5e7eb"
        )
        .configure_view(strokeWidth=0)
        .configure_legend(labelFontSize=11)
    )

    st.altair_chart(chart, use_container_width=True)

# Save results for comparison page
if "clf_results" not in st.session_state:
    st.session_state["clf_results"] = {}
st.session_state["clf_results"]["Random Forest"] = {
    "Acuratețe": res["accuracy"], "F1-Score": res["f1"],
    "Precizie": res["precision"], "Recall": res["recall"]
}
