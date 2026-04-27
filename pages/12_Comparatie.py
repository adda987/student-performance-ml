import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import PAGE_CSS, PLOT_CFG, preprocess_data

st.set_page_config(page_title="Comparație modele", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.markdown("""
<div class="ph" style="border-left-color:#0ea5e9;background:linear-gradient(90deg,#f0f9ff,white);">
    <div class="label" style="color:#0284c7;">12 / COMPARAȚIE</div>
    <h1> Comparație — Toate Modelele</h1>
    <div class="sub">Regresie: R², RMSE, MAE, MSE · Clasificare: Acuratețe, F1, Precizie, Recall </div>
</div>""", unsafe_allow_html=True)

data = preprocess_data()
df = data["df_scaled"]
feat = data["feature_cols"]
target_names = data["target_names"]

st.sidebar.markdown("## Configurare Comparație")
test_size  = st.sidebar.slider("Proporție test", 0.10, 0.40, 0.20, 0.05)
rand_state = st.sidebar.number_input("Random state", 0, 999, 42)
run_all    = st.sidebar.button("Rulează toate modelele", use_container_width=True)

# ── Regression data
X_reg = df[feat].values
y_reg = df[data["target_reg"]].values
X_rtr, X_rte, y_rtr, y_rte = train_test_split(X_reg, y_reg, test_size=test_size, random_state=rand_state)

# ── Classification data
y_clf = df[data["target_clf"]].values
X_ctr, X_cte, y_ctr, y_cte = train_test_split(X_reg, y_clf, test_size=test_size, random_state=rand_state, stratify=y_clf)

# ── SVM sample (speed)
n = len(X_reg)
if n > 8000:
    rng = np.random.default_rng(rand_state)
    idx = rng.choice(n, 8000, replace=False)
    X_svm, y_svm_clf = X_reg[idx], y_clf[idx]
    X_svm_tr, X_svm_te, y_svm_tr, y_svm_te = train_test_split(X_svm, y_svm_clf, test_size=test_size,
                                                                 random_state=rand_state, stratify=y_svm_clf)
    X_svr_tr, X_svr_te, y_svr_tr, y_svr_te = train_test_split(X_svm, y_reg[idx], test_size=test_size, random_state=rand_state)
else:
    X_svm_tr, X_svm_te, y_svm_tr, y_svm_te = X_ctr, X_cte, y_ctr, y_cte
    X_svr_tr, X_svr_te, y_svr_tr, y_svr_te = X_rtr, X_rte, y_rtr, y_rte

REGRESSION_MODELS = [
    ("OLS",          LinearRegression()),
    ("Ridge",        Ridge(alpha=1.0)),
    ("Lasso",        Lasso(alpha=0.01, max_iter=5000)),
    ("RF Regressor", RandomForestRegressor(n_estimators=150, max_depth=7, random_state=rand_state, n_jobs=-1)),
    ("SVR (RBF)",    SVR(kernel="rbf", C=10)),
]

CLASSIFICATION_MODELS = [
    ("Random Forest",        RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=rand_state, n_jobs=-1)),
    ("SVM (RBF)",            SVC(kernel="rbf", C=1, class_weight="balanced", random_state=rand_state, probability=True)),
    ("Logistic Regression",  LogisticRegression(C=1, class_weight="balanced", max_iter=500, random_state=rand_state, solver="lbfgs")),
    ("Decision Tree",        DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=rand_state)),
    ("KNN",                  KNeighborsClassifier(n_neighbors=7, n_jobs=-1)),
]

if run_all or "comparison_done" not in st.session_state:
    prog = st.progress(0, text="Se antrenează modelele...")
    total = len(REGRESSION_MODELS) + len(CLASSIFICATION_MODELS)
    step = 0

    # ── Regression
    reg_rows = []
    for name, model in REGRESSION_MODELS:
        prog.progress(step/total, text=f"Regresie: {name}...")
        if name == "SVR (RBF)":
            model.fit(X_svr_tr, y_svr_tr); yp = model.predict(X_svr_te)
            y_te_use = y_svr_te
        else:
            model.fit(X_rtr, y_rtr); yp = model.predict(X_rte)
            y_te_use = y_rte
        yp_tr = model.predict(X_svr_tr if name=="SVR (RBF)" else X_rtr)
        y_tr_use = y_svr_tr if name=="SVR (RBF)" else y_rtr
        reg_rows.append({
            "Model": name,
            "R² Train": round(r2_score(y_tr_use, yp_tr), 4),
            "R² Test":  round(r2_score(y_te_use, yp), 4),
            "RMSE":     round(float(np.sqrt(mean_squared_error(y_te_use, yp))), 4),
            "MAE":      round(float(mean_absolute_error(y_te_use, yp)), 4),
            "MSE":      round(float(mean_squared_error(y_te_use, yp)), 4),
            "ΔR²":      round(abs(r2_score(y_tr_use, yp_tr) - r2_score(y_te_use, yp)), 4),
        })
        step += 1

    # ── Classification
    clf_rows = []
    for name, model in CLASSIFICATION_MODELS:
        prog.progress(step/total, text=f"Clasificare: {name}...")
        if name == "SVM (RBF)":
            model.fit(X_svm_tr, y_svm_tr); yp = model.predict(X_svm_te)
            y_te_use = y_svm_te
        else:
            model.fit(X_ctr, y_ctr); yp = model.predict(X_cte)
            y_te_use = y_cte
        clf_rows.append({
            "Model": name,
            "Acuratețe": round(float(accuracy_score(y_te_use, yp)), 4),
            "F1-Score":  round(float(f1_score(y_te_use, yp, average="weighted", zero_division=0)), 4),
            "Precizie":  round(float(precision_score(y_te_use, yp, average="weighted", zero_division=0)), 4),
            "Recall":    round(float(recall_score(y_te_use, yp, average="weighted", zero_division=0)), 4),
        })
        step += 1

    prog.progress(1.0)
    st.session_state["comparison_done"] = True
    st.session_state["reg_rows"] = reg_rows
    st.session_state["clf_rows"] = clf_rows

reg_rows = st.session_state.get("reg_rows", [])
clf_rows = st.session_state.get("clf_rows", [])

if not reg_rows:
    st.info(" Apasă **Rulează toate modelele** în sidebar pentru a genera comparația.")
    st.stop()

# ══════════════════════════════════════════════
# REGRESSION COMPARISON
# ══════════════════════════════════════════════
st.markdown('<div class="sec">Comparație Modele Regresie</div>', unsafe_allow_html=True)
reg_df = pd.DataFrame(reg_rows).sort_values("R² Test", ascending=False)

# Color best rows
def color_best(val, col_name, df):
    if col_name == "R² Test":
        return "background-color:#dcfce7;font-weight:700;" if val == df[col_name].max() else ""
    if col_name in ["RMSE","MAE","MSE","ΔR²"]:
        return "background-color:#dcfce7;font-weight:700;" if val == df[col_name].min() else ""
    return ""

st.dataframe(reg_df.reset_index(drop=True), use_container_width=True, hide_index=True)

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(reg_df, x="Model", y="R² Test", color="R² Test",
                 color_continuous_scale=["#fef9c3","#22c55e"], title="R² Test per Model Regresie")
    fig.update_layout(**PLOT_CFG, height=350, coloraxis_showscale=False, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig2 = px.bar(reg_df, x="Model", y="RMSE", color="RMSE",
                  color_continuous_scale=["#22c55e","#ef4444"], title="RMSE per Model (mai mic = mai bun)")
    fig2.update_layout(**PLOT_CFG, height=350, coloraxis_showscale=False, xaxis_tickangle=-30)
    st.plotly_chart(fig2, use_container_width=True)

# Overfitting radar
fig3 = go.Figure()
fig3.add_trace(go.Bar(x=reg_df["Model"], y=reg_df["R² Train"], name="R² Train", marker_color="#93c5fd"))
fig3.add_trace(go.Bar(x=reg_df["Model"], y=reg_df["R² Test"],  name="R² Test",  marker_color="#3b82f6"))
fig3.add_hline(y=0, line_color="#374151")
fig3.update_layout(barmode="group", title="R² Train vs Test — Overfitting Check",
                    **PLOT_CFG, height=350, xaxis_tickangle=-30)
st.plotly_chart(fig3, use_container_width=True)

best_reg = reg_df.iloc[0]

# ══════════════════════════════════════════════
# CLASSIFICATION COMPARISON
# ══════════════════════════════════════════════
st.markdown('<div class="sec">Comparație Modele Clasificare</div>', unsafe_allow_html=True)
clf_df = pd.DataFrame(clf_rows).sort_values("F1-Score", ascending=False)
st.dataframe(clf_df.reset_index(drop=True), use_container_width=True, hide_index=True)

c3, c4 = st.columns(2)
with c3:
    metrics_long = clf_df.melt(id_vars="Model", value_vars=["Acuratețe","F1-Score","Precizie","Recall"],
                                var_name="Metrică", value_name="Valoare")
    fig4 = px.bar(metrics_long, x="Model", y="Valoare", color="Metrică", barmode="group",
                  title="Toate metricile per model clasificare",
                  color_discrete_sequence=["#3b82f6","#6366f1","#10b981","#f59e0b"])
    fig4.update_layout(**PLOT_CFG, height=400, xaxis_tickangle=-30)
    st.plotly_chart(fig4, use_container_width=True)
with c4:
    # Radar chart per model
    categories = ["Acuratețe","F1-Score","Precizie","Recall"]
    fig5 = go.Figure()
    colors = ["#8b5cf6","#0ea5e9","#ec4899","#14b8a6","#a855f7"]
    for i, row in clf_df.iterrows():
        vals = [row[c] for c in categories] + [row[categories[0]]]
        fig5.add_trace(go.Scatterpolar(r=vals, theta=categories+[categories[0]],
                                        fill="toself", name=row["Model"],
                                        line=dict(color=colors[list(clf_df.index).index(i) % len(colors)])))
    fig5.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                        title="Radar — Metrici clasificare", **PLOT_CFG, height=400)
    st.plotly_chart(fig5, use_container_width=True)

