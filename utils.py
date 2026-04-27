"""
utils.py — Shared utilities for OULAD ML App
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; background-color: #f8fafc !important; color: #1e293b !important; }
h1,h2,h3 { color: #1e293b !important; font-weight: 700 !important; }
[data-testid="stSidebar"] { background-color: #f1f5f9 !important; border-right: 1px solid #e2e8f0; }
[data-testid="stSidebar"] * { color: #1e293b !important; }
.ph { border-left: 5px solid #3b82f6; padding: 20px 28px; background: white; border-radius: 8px; margin-bottom: 28px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
.ph .label { font-size: 11px; font-family: 'IBM Plex Mono',monospace; color: #3b82f6; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 6px; }
.ph h1 { font-size: 34px !important; font-weight: 800 !important; margin: 0 !important; }
.ph .sub { font-size: 14px; color: #64748b; margin-top: 6px; }
.sec { font-size: 11px; font-family: 'IBM Plex Mono',monospace; color: #3b82f6; letter-spacing: 3px; text-transform: uppercase; padding-bottom: 8px; border-bottom: 1px solid #e2e8f0; margin: 32px 0 14px 0; }
.metric-card { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
.metric-card .val { font-size: 26px; font-weight: 700; color: #3b82f6; }
.metric-card .lbl { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
.test-card { background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px 18px; margin-bottom: 8px; }
.test-card .tc-name { font-weight: 600; font-size: 13px; color: #1e293b; margin-bottom: 6px; }
.test-card .tc-val { font-family: 'IBM Plex Mono',monospace; font-size: 12px; color: #1e40af; margin-bottom: 4px; }
.test-card .tc-h0 { font-size: 11px; color: #64748b; margin-bottom: 6px; font-style: italic; }
.badge-ok { background:#dcfce7; color:#166534; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; display:inline-block; }
.badge-warn { background:#fef9c3; color:#854d0e; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; display:inline-block; }
.badge-err { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; display:inline-block; }
.info-box { background: #eff6ff; border: 1px solid #bfdbfe; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 14px 18px; font-size: 14px; color: #1e40af; line-height: 1.7; margin-bottom: 12px; }
.warn-box { background: #fffbeb; border: 1px solid #fde68a; border-left: 4px solid #f59e0b; border-radius: 8px; padding: 14px 18px; font-size: 14px; color: #78350f; line-height: 1.7; margin-bottom: 12px; }
</style>
"""

PLOT_CFG = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc", font=dict(family="Inter", color="#1e293b"))
DATA_PATH = "oulad_students.csv"

@st.cache_data
def load_raw():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def preprocess_data():
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    df = load_raw().copy()
    df = df.drop(columns=["id_student","code_module","code_presentation"], errors="ignore")
    # Convert avg_score to numeric and count missing
    df["avg_score"] = pd.to_numeric(df["avg_score"], errors="coerce")
    missing_before = int(df.isnull().sum().sum())
    # Impute avg_score with median
    df["avg_score"] = df["avg_score"].fillna(df["avg_score"].median())
    # Replace ? with NaN in imd_band
    df["imd_band"] = df["imd_band"].replace("?", np.nan)
    # Impute all remaining categoricals with mode
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    missing_after = int(df.isnull().sum().sum())
    # Impute any remaining numeric NaNs with median (e.g. total_clicks, active_days)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    missing_after = int(df.isnull().sum().sum())
    outlier_flags = {}
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)
        outlier_flags[col] = int(mask.sum())
        df[col] = df[col].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_features = [c for c in cat_cols if c != "final_result"]
    encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        df[col+"_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    le_target = LabelEncoder()
    df["final_result_enc"] = le_target.fit_transform(df["final_result"])
    feature_cols = [c for c in df.columns if c not in ["final_result","avg_score","final_result_enc"] + cat_features]
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    return {"df_raw": df, "df_scaled": df_scaled, "feature_cols": feature_cols,
            "cat_features": cat_features, "encoders": encoders, "le_target": le_target,
            "scaler": scaler, "target_clf": "final_result_enc", "target_reg": "avg_score",
            "target_names": le_target.classes_.tolist(), "missing_before": missing_before,
            "missing_after": missing_after, "outlier_flags": outlier_flags}

def run_regression(X_train, X_test, y_train, y_test, model, model_name, color="#3b82f6"):
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    model.fit(X_train, y_train)
    yp_tr = model.predict(X_train); yp_te = model.predict(X_test)
    return {"model": model, "model_name": model_name, "color": color,
            "y_pred_train": yp_tr, "y_pred_test": yp_te, "y_test": y_test,
            "residuals": y_test - yp_te,
            "r2_train": float(r2_score(y_train, yp_tr)), "r2_test": float(r2_score(y_test, yp_te)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, yp_te))),
            "mae": float(mean_absolute_error(y_test, yp_te)),
            "mse": float(mean_squared_error(y_test, yp_te))}

def show_regression_metrics(res):
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,val,lbl in [(c1,f"{res['r2_train']:.4f}","R² Train"),(c2,f"{res['r2_test']:.4f}","R² Test"),
                         (c3,f"{res['rmse']:.4f}","RMSE"),(c4,f"{res['mae']:.4f}","MAE"),(c5,f"{res['mse']:.4f}","MSE")]:
        col.markdown(f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

def show_regression_plots(res):
    c1,c2 = st.columns(2)
    with c1:
        fig = px.scatter(x=res["y_test"], y=res["y_pred_test"], labels={"x":"Real","y":"Prezis"},
                         title="Predicted vs Actual", color_discrete_sequence=[res["color"]])
        mn,mx = float(res["y_test"].min()), float(res["y_test"].max())
        fig.add_shape(type="line",x0=mn,y0=mn,x1=mx,y1=mx,line=dict(color="#ef4444",dash="dash"))
        fig.update_layout(**PLOT_CFG, height=350); st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.scatter(x=res["y_pred_test"], y=res["residuals"], labels={"x":"Predicted","y":"Reziduu"},
                          title="Reziduu vs Predicted", color_discrete_sequence=[res["color"]])
        fig2.add_hline(y=0, line_dash="dash", line_color="#ef4444")
        fig2.update_layout(**PLOT_CFG, height=350); st.plotly_chart(fig2, use_container_width=True)
    c3,c4 = st.columns(2)
    with c3:
        import matplotlib.pyplot as plt

        residuals = pd.Series(res["residuals"]).dropna()

        fig3, ax = plt.subplots(figsize=(7, 4))
        ax.hist(residuals, bins=40)

        ax.axvline(residuals.mean(), linestyle="--", linewidth=2)
        ax.set_title("Distribuția reziduurilor")
        ax.set_xlabel("Reziduu")
        ax.set_ylabel("Frecvență")

        fig3.tight_layout()
        st.pyplot(fig3)
    with c4:
        sr = np.sort(res["residuals"]); n = len(sr)
        tq = stats.norm.ppf(np.arange(1,n+1)/(n+1))
        fig4 = go.Figure(); fig4.add_trace(go.Scatter(x=tq,y=sr,mode="markers",marker=dict(color=res["color"],size=3)))
        mn2,mx2 = min(tq.min(),float(sr.min())), max(tq.max(),float(sr.max()))
        fig4.add_shape(type="line",x0=mn2,y0=mn2,x1=mx2,y1=mx2,line=dict(color="#ef4444",dash="dash"))
        fig4.update_layout(title="Q-Q Plot",xaxis_title="Cuantile teoretice",yaxis_title="Cuantile empirice",**PLOT_CFG,height=300)
        st.plotly_chart(fig4, use_container_width=True)

def show_econometric_tests(X_test, y_test, y_pred, feature_names):
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm
    residuals = y_test - y_pred; n = len(residuals)
    jb_stat, jb_p = stats.jarque_bera(residuals)
    dw_stat = durbin_watson(residuals)
    try:
        X_c = sm.add_constant(X_test); bp_lm, bp_p, _, _ = het_breuschpagan(residuals, X_c)
    except: bp_lm = bp_p = np.nan
    sample = residuals[:5000] if n > 5000 else residuals
    sw_stat, sw_p = stats.shapiro(sample)
    try:
        X_c2 = sm.add_constant(pd.DataFrame(X_test, columns=feature_names))
        vif_vals = [variance_inflation_factor(X_c2.values, i) for i in range(1, X_c2.shape[1])]
        max_vif = max(vif_vals)
    except: vif_vals=[]; max_vif=np.nan
    st.markdown('<div class="sec">Teste Econometrice</div>', unsafe_allow_html=True)
    def badge(ok, warn):
        if ok: return '<span class="badge-ok">✓ OK</span>'
        if warn: return '<span class="badge-warn">⚠ Atenție</span>'
        return '<span class="badge-err">✗ Respins</span>'
    tests = [
        ("Jarque-Bera", f"Stat={jb_stat:.4f}, p={jb_p:.4f}", "H₀: Reziduuri normal distribuite", badge(jb_p>0.05,jb_p>0.01)),
        ("Shapiro-Wilk", f"Stat={sw_stat:.4f}, p={sw_p:.4f}", "H₀: Reziduuri normale", badge(sw_p>0.05,sw_p>0.01)),
        ("Durbin-Watson", f"DW = {dw_stat:.4f}", "H₀: Fără autocorelare (val ~2)", badge(1.5<dw_stat<2.5,1.0<dw_stat<3.0)),
        ("Breusch-Pagan", f"LM={bp_lm:.4f}, p={bp_p:.4f}" if not np.isnan(bp_lm) else "N/A", "H₀: Homoscedasticitate", badge(bp_p>0.05 if not np.isnan(bp_p) else False, bp_p>0.01 if not np.isnan(bp_p) else False)),
        ("VIF max", f"VIF_max = {max_vif:.2f}" if not np.isnan(max_vif) else "N/A", "VIF<5 OK, 5-10 Moderat, >10 Sever", badge(max_vif<5 if not np.isnan(max_vif) else False, max_vif<10 if not np.isnan(max_vif) else False)),
    ]
    cols = st.columns(3)
    for i,(name,val,h0,bdg) in enumerate(tests):
        with cols[i%3]:
            st.markdown(f'<div class="test-card"><div class="tc-name">{name}</div><div class="tc-val">{val}</div><div class="tc-h0">{h0}</div>{bdg}</div>', unsafe_allow_html=True)
    if vif_vals:
        with st.expander(" VIF per variabilă"):
            st.dataframe(pd.DataFrame({"Feature":feature_names,"VIF":[round(v,3) for v in vif_vals]}).sort_values("VIF",ascending=False), use_container_width=True, hide_index=True)
    return {"jb_p":jb_p,"sw_p":sw_p,"dw":dw_stat,"bp_p":bp_p,"max_vif":max_vif}

def run_classification(X_train, X_test, y_train, y_test, model, model_name, color="#6366f1"):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
    model.fit(X_train, y_train); y_pred = model.predict(X_test)
    return {"model":model,"model_name":model_name,"color":color,"y_pred":y_pred,"y_test":y_test,
            "accuracy": float(accuracy_score(y_test,y_pred)),
            "f1": float(f1_score(y_test,y_pred,average="weighted",zero_division=0)),
            "precision": float(precision_score(y_test,y_pred,average="weighted",zero_division=0)),
            "recall": float(recall_score(y_test,y_pred,average="weighted",zero_division=0)),
            "cm": confusion_matrix(y_test,y_pred)}

def show_clf_metrics(res):
    c1,c2,c3,c4 = st.columns(4)
    for col,val,lbl,clr in [(c1,f"{res['accuracy']:.4f}","Acuratețe","#3b82f6"),(c2,f"{res['f1']:.4f}","F1-Score","#6366f1"),(c3,f"{res['precision']:.4f}","Precizie","#10b981"),(c4,f"{res['recall']:.4f}","Recall","#f59e0b")]:
        col.markdown(f'<div class="metric-card"><div class="val" style="color:{clr};">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

def show_confusion_matrix(res, target_names):
    import pandas as pd
    import altair as alt

    cm = res["cm"]

    st.markdown('<div class="sec">Matricea de Confuzie</div>', unsafe_allow_html=True)

    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_long = cm_df.reset_index().melt(
        id_vars="index",
        var_name="Predicted",
        value_name="Count"
    ).rename(columns={"index": "Actual"})

    base = alt.Chart(cm_long)

    heatmap = base.mark_rect(cornerRadius=4).encode(
        x=alt.X("Predicted:N", title="Prezis"),
        y=alt.Y("Actual:N", title="Real"),
        color=alt.Color(
            "Count:Q",
            scale=alt.Scale(scheme="purples"),
            legend=alt.Legend(title="Nr.")
        ),
        tooltip=[
            alt.Tooltip("Actual:N", title="Real"),
            alt.Tooltip("Predicted:N", title="Prezis"),
            alt.Tooltip("Count:Q", title="Număr")
        ]
    )

    text = base.mark_text(fontSize=13, fontWeight="bold").encode(
        x=alt.X("Predicted:N"),
        y=alt.Y("Actual:N"),
        text=alt.Text("Count:Q"),
        color=alt.condition(
            alt.datum.Count > cm.max() / 2,
            alt.value("white"),
            alt.value("#374151")
        )
    )

    chart = (
        (heatmap + text)
        .properties(
            width=420,
            height=320,
            title="Confusion Matrix"
        )
        .configure_title(
            fontSize=16,
            fontWeight="bold",
            color="#1f2937",
            anchor="start"
        )
        .configure_axis(
            labelFontSize=11,
            titleFontSize=12,
            labelColor="#6b7280",
            titleColor="#6b7280",
            grid=False
        )
        .configure_view(strokeWidth=0)
        .configure_legend(labelFontSize=10, titleFontSize=11)

    )

    st.altair_chart(chart, use_container_width=False)

    if len(target_names) != 2:
        with st.expander("TP / TN / FP / FN per clasă"):
            rows = []
            total = cm.sum()
            for i, cls in enumerate(target_names):
                TP = int(cm[i, i])
                FP = int(cm[:, i].sum() - TP)
                FN = int(cm[i, :].sum() - TP)
                TN = int(total - TP - FP - FN)
                rows.append({"Clasă": cls, "TP": TP, "TN": TN, "FP": FP, "FN": FN})

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def show_clf_plots(res, target_names):
    y_test,y_pred = res["y_test"],res["y_pred"]
    c1,c2 = st.columns(2)
    with c1:
        df_p = pd.DataFrame({"Real":[target_names[i] for i in y_test],"Prezis":[target_names[i] for i in y_pred]})
        counts = df_p.groupby(["Real","Prezis"]).size().reset_index(name="Count")
        fig = px.bar(counts,x="Real",y="Count",color="Prezis",title="Distribuție Predicții vs Real",barmode="group",color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(**PLOT_CFG,height=350); st.plotly_chart(fig, use_container_width=True)
    with c2:
        from sklearn.metrics import classification_report
        cr = classification_report(y_test,y_pred,target_names=target_names,output_dict=True,zero_division=0)
        cr_df = pd.DataFrame(cr).T.drop(["accuracy","macro avg","weighted avg"],errors="ignore")[["precision","recall","f1-score"]].astype(float).round(3)
        fig2 = go.Figure()
        for m in ["precision","recall","f1-score"]:
            fig2.add_trace(go.Bar(name=m,x=cr_df.index.tolist(),y=cr_df[m].tolist()))
        fig2.update_layout(barmode="group",title="Metrici per clasă",**PLOT_CFG,height=350)
        st.plotly_chart(fig2, use_container_width=True)
