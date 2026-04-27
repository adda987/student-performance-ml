import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from utils import PAGE_CSS, preprocess_data, load_raw

st.set_page_config(page_title="OULAD ML Platform", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)

st.markdown("""
<style>
.hero { background: linear-gradient(135deg, #1e40af 0%, #6366f1 60%, #8b5cf6 100%); padding: 48px 52px; border-radius: 16px; margin-bottom: 32px; color: white; }
.hero h1 { color: white !important; font-size: 40px; margin: 0 0 12px 0; }
.hero p  { color: rgba(255,255,255,0.88); font-size: 16px; max-width: 700px; line-height: 1.8; margin: 0; }
.step-card { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 22px; margin-bottom: 12px; border-left: 4px solid #3b82f6; }
.step-card h4 { margin: 0 0 8px 0; color: #1e40af; font-size: 15px; }
.step-card p  { margin: 0; color: #475569; font-size: 13px; line-height: 1.6; }
.nav-card { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin-bottom: 8px; }
.nav-card .nc-tag { font-size: 10px; font-family: monospace; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 6px; }
.nav-card .nc-title { font-size: 15px; font-weight: 600; color: #1e293b; margin-bottom: 4px; }
.nav-card .nc-desc { font-size: 12px; color: #64748b; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>Machine Learning Performance</h1>
    <p>Platformă de Machine Learning pentru analiza și predicția performanței studenților pe baza
    <strong>Open University Learning Analytics Dataset (OULAD)</strong> — 32,593 studenți,
    modele de regresie și clasificare, teste econometrice complete.</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Se încarcă și preprocesează datele OULAD..."):
    data = preprocess_data()
    df_raw = data["df_raw"]

# ── KPIs
c1, c2, c3, c4, c5, c6 = st.columns(6)

for col, val, lbl in [
    (c1, f"{len(df_raw):,}", "Înregistrări"),
    (c2, f"{len(data['feature_cols'])}", "Features"),
    (c3, f"{data['missing_before']:,}", "Val. lipsă (brut)"),
    (c4, f"{data['missing_after']}", "Val. lipsă (după)"),
    (c5, f"{df_raw[data['target_reg']].mean():.1f}", "Scor mediu"),
    (c6, f"{len(data['target_names'])}", "Clase target"),
]:
    col.markdown(
        f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
        unsafe_allow_html=True
    )

# ── Preprocesare
st.markdown('<div class="sec">Preprocesare Date</div>', unsafe_allow_html=True)

steps = [
    ("#3b82f6", "1", "Valori Lipsă", f"avg_score: imputare mediană ({data['missing_before']:,} val. lipsă detectate). imd_band: '?' → NaN → imputare modă. Rezultat: {data['missing_after']} valori lipsă rămase."),
    ("#8b5cf6", "2", "Outlieri", f"Metodă IQR: clip la Q1−1.5×IQR și Q3+1.5×IQR. {sum(data['outlier_flags'].values())} outlieri tratați pe {len(data['outlier_flags'])} variabile numerice."),
    ("#10b981", "3", "Encoding", f"LabelEncoder pe {len(data['cat_features'])} coloane categoriale: gender, region, highest_education, imd_band, age_band, disability. final_result → 4 clase numerice."),
    ("#f59e0b", "4", "Scalare", "StandardScaler pe toate variabilele numerice. Esențial pentru OLS, Ridge, Lasso, SVR, KNN și Logistic Regression."),
]

cp_cols = st.columns(4)

for col, (clr, num, title, desc) in zip(cp_cols, steps):
    col.markdown(
        f"""
        <div class="step-card" style="border-left-color:{clr};">
            <h4>
                <span style="background:{clr};color:white;border-radius:50%;width:22px;height:22px;display:inline-flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;margin-right:8px;">{num}</span>
                {title}
            </h4>
            <p>{desc}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with st.expander("Detaliu outlieri per variabilă"):
    out_df = pd.DataFrame({
        "Variabilă": list(data["outlier_flags"].keys()),
        "Nr. outlieri": list(data["outlier_flags"].values()),
        "Procent (%)": [round(v / len(df_raw) * 100, 2) for v in data["outlier_flags"].values()]
    })
    st.dataframe(out_df, use_container_width=True, hide_index=True)

# ── Statistici descriptive
st.markdown('<div class="sec">Statistici Descriptive</div>', unsafe_allow_html=True)
st.dataframe(load_raw().describe(include="all").round(3), use_container_width=True)

# ── Target distribution
st.markdown('<div class="sec">Distribuția Variabilelor Target</div>', unsafe_allow_html=True)

raw_df = load_raw().copy()
raw_df["avg_score"] = pd.to_numeric(raw_df["avg_score"], errors="coerce")

tc1, tc2 = st.columns(2)

with tc1:
    avg_clean = raw_df["avg_score"].dropna()
    median_val = avg_clean.median()

    avg_df = pd.DataFrame({"avg_score": avg_clean})

    hist = (
        alt.Chart(avg_df)
        .mark_bar(
            color="#3b82f6",
            opacity=0.85,
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3
        )
        .encode(
            x=alt.X("avg_score:Q", bin=alt.Bin(maxbins=25), title="Scor mediu evaluări"),
            y=alt.Y("count():Q", title="Nr. studenți"),
            tooltip=[alt.Tooltip("count():Q", title="Nr. studenți")]
        )
    )

    median_line = (
        alt.Chart(pd.DataFrame({"median": [median_val]}))
        .mark_rule(color="#ef4444", strokeDash=[6, 4], strokeWidth=2)
        .encode(x="median:Q")
    )

    median_text = (
        alt.Chart(pd.DataFrame({
            "median": [median_val],
            "label": [f"Mediană = {median_val:.1f}"]
        }))
        .mark_text(
            align="left",
            dx=6,
            dy=-8,
            color="#ef4444",
            fontSize=11,
            fontWeight="bold"
        )
        .encode(
            x="median:Q",
            y=alt.value(20),
            text="label:N"
        )
    )

    chart_avg = (
        (hist + median_line + median_text)
        .properties(height=320, title="Distribuția avg_score")
        .configure_title(fontSize=16, fontWeight="bold", color="#1f2937", anchor="start")
        .configure_axis(
            labelFontSize=10,
            titleFontSize=11,
            labelColor="#6b7280",
            titleColor="#6b7280",
            gridColor="#e5e7eb"
        )
        .configure_view(strokeWidth=0)
    )

    st.altair_chart(chart_avg, use_container_width=True)

with tc2:
    fr = raw_df["final_result"].dropna().value_counts().reset_index()
    fr.columns = ["result", "count"]

    chart_fr = (
        alt.Chart(fr)
        .mark_bar(
            cornerRadiusTopLeft=4,
            cornerRadiusTopRight=4,
            opacity=0.9
        )
        .encode(
            x=alt.X("result:N", title="Rezultat", sort="-y"),
            y=alt.Y("count:Q", title="Nr. studenți"),
            color=alt.Color(
                "result:N",
                scale=alt.Scale(
                    range=["#3b82f6", "#8b5cf6", "#ef4444", "#f59e0b"]
                ),
                legend=None
            ),
            tooltip=[
                alt.Tooltip("result:N", title="Rezultat"),
                alt.Tooltip("count:Q", title="Nr. studenți")
            ]
        )
        .properties(height=320, title="Distribuția final_result")
        .configure_title(fontSize=16, fontWeight="bold", color="#1f2937", anchor="start")
        .configure_axis(
            labelFontSize=10,
            titleFontSize=11,
            labelColor="#6b7280",
            titleColor="#6b7280",
            gridColor="#e5e7eb"
        )
        .configure_view(strokeWidth=0)
    )

    st.altair_chart(chart_fr, use_container_width=True)

# ── Correlation matrix
st.markdown('<div class="sec">Matrice Corelații</div>', unsafe_allow_html=True)

corr_cols = data["feature_cols"] + [data["target_reg"]]
corr_df = df_raw[corr_cols].copy()

for col in corr_df.columns:
    corr_df[col] = pd.to_numeric(corr_df[col], errors="coerce")

corr_df = corr_df.dropna(axis=1, how="all")
corr_df = corr_df.loc[:, corr_df.nunique(dropna=True) > 1]

corr = corr_df.corr(numeric_only=True)

if corr.empty:
    st.warning("Matricea de corelații nu poate fi calculată: nu există suficiente variabile numerice valide.")
else:
    corr_long = (
        corr.reset_index()
        .melt(id_vars="index", var_name="Variabila X", value_name="Corelație")
        .rename(columns={"index": "Variabila Y"})
    )

    base = alt.Chart(corr_long)

    heatmap = base.mark_rect(cornerRadius=3).encode(
        x=alt.X("Variabila X:N", title=None, sort=list(corr.columns)),
        y=alt.Y("Variabila Y:N", title=None, sort=list(corr.index)),
        color=alt.Color(
            "Corelație:Q",
            scale=alt.Scale(
                domain=[-1, 0, 1],
                range=["#dbeafe", "#f8fafc", "#6366f1"]
            ),
            legend=alt.Legend(title="Corelație")
        ),
        tooltip=[
            alt.Tooltip("Variabila Y:N", title="Y"),
            alt.Tooltip("Variabila X:N", title="X"),
            alt.Tooltip("Corelație:Q", format=".2f")
        ]
    )

    text = base.mark_text(fontSize=9).encode(
        x=alt.X("Variabila X:N", sort=list(corr.columns)),
        y=alt.Y("Variabila Y:N", sort=list(corr.index)),
        text=alt.Text("Corelație:Q", format=".2f"),
        color=alt.condition(
            "abs(datum['Corelație']) > 0.55",
            alt.value("white"),
            alt.value("#374151")
        )
    )

    chart_corr = (
        (heatmap + text)
        .properties(width=680, height=430, title="Corelații Pearson")
        .configure_title(fontSize=16, fontWeight="bold", color="#1f2937", anchor="start")
        .configure_axis(
            labelFontSize=10,
            titleFontSize=11,
            labelColor="#6b7280",
            titleColor="#6b7280",
            labelAngle=-35,
            grid=False
        )
        .configure_view(strokeWidth=0)
        .configure_legend(labelFontSize=10, titleFontSize=11)
    )

    st.altair_chart(chart_corr, use_container_width=False)

# ── Navigation
st.markdown('<div class="sec">Modele disponibile</div>', unsafe_allow_html=True)

pages = [
    ("REGRESIE", "01", "OLS — Regresie Liniară", "Baza de comparație. Teste econometrice complete.", "#3b82f6"),
    ("REGRESIE", "02", "Ridge Regression", "Penalizare L2. Reduce multicoliniaritatea.", "#6366f1"),
    ("REGRESIE", "03", "Lasso Regression", "Penalizare L1. Selecție automată features.", "#8b5cf6"),
    ("REGRESIE", "04", "Random Forest Regressor", "Ensemble non-liniar. Feature importance.", "#22c55e"),
    ("REGRESIE", "05", "SVR & Extra Trees", "SVR cu kernel RBF + Extra Trees Regressor.", "#10b981"),
    ("CLASIFICARE", "06", "Random Forest Clf", "Cel mai robust clasificator. OOB score.", "#f59e0b"),
    ("CLASIFICARE", "07", "SVM Classifier", "Kernel RBF. Hiperplan de separare optim.", "#ef4444"),
    ("CLASIFICARE", "08", "Logistic Regression", "Clasificare liniară. Probabilități per clasă.", "#ec4899"),
    ("CLASIFICARE", "09", "Decision Tree", "Arbore de decizie. Vizualizare reguli.", "#14b8a6"),
    ("CLASIFICARE", "10", "KNN Classifier", "K vecini apropiați. Alegerea K optim.", "#a855f7"),
    ("COMPARATIE", "11", "Comparație", "Tabel comparativ toate modelele regresie + clasificare.", "#0ea5e9"),
]

cols = st.columns(3)

for i, (typ, num, title, desc, clr) in enumerate(pages):
    with cols[i % 3]:
        st.markdown(
            f"""
            <div class="nav-card">
                <div class="nc-tag" style="color:{clr};">{typ} · {num}</div>
                <div class="nc-title">{title}</div>
                <div class="nc-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)