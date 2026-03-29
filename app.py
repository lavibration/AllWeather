import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 1. SETUP ---
st.set_page_config(page_title="Stratégie Macro-Financière", layout="wide")
REGIME_COLORS = {
    "GOLDILOCKS": "green",
    "REFLATION": "blue",
    "STAGFLATION": "orange",
    "DEFLATION": "grey",
    "UNKNOWN": "white"
}

@st.cache_data
def load_data():
    res = pd.read_csv("backtest_results.csv", index_col=0, parse_dates=True)
    signals = pd.read_csv("signals_analysis.csv", index_col=0, parse_dates=True)
    stats = pd.read_csv("strategy_stats.csv")
    return res, signals, stats

res, signals, stats = load_data()

# --- 2. TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["Cockpit d'Exécution", "Signaux & Hystérésis", "Analyse des Performances", "Méthodologie"])

# --- TAB 1: COCKPIT ---
with tab1:
    st.header("Cockpit d'Exécution")

    col1, col2 = st.columns(2)
    for zone, c in [("US", col1), ("EU", col2)]:
        current_regime = res[f"Regime_{zone}"].iloc[-1]
        c.metric(f"Régime Actuel {zone}", current_regime)
        c.markdown(f"<div style='background-color:{REGIME_COLORS.get(current_regime, 'white')}; height:20px; width:100%; border-radius:5px;'></div>", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Allocations Cibles (Macro Pocket 90% + Gold 10%)")
    col1, col2 = st.columns(2)

    # Target Allocations: 90% Macro Pocket + 10% Gold
    # Weights revised for v2 strategy
    alloc_us = {
        "GOLDILOCKS":  {"XLK": 30.0, "XLY": 30.0, "XLF": 30.0, "GLD": 10.0},
        "REFLATION":   {"XLE": 36.0, "DBC": 27.0, "XLB": 27.0, "GLD": 10.0},
        "STAGFLATION": {"DBC": 36.0, "CASH": 54.0, "GLD": 10.0},
        "DEFLATION":   {"XLY": 36.0, "XLC": 27.0, "CASH": 27.0, "GLD": 10.0}
    }

    alloc_eu = {
        "GOLDILOCKS":  {"EXV4.DE": 30.0, "EXV3.DE": 30.0, "EXI5.DE": 30.0, "GLD": 10.0},
        "REFLATION":   {"EXV6.DE": 36.0, "EXV1.DE": 30.0, "DBC": 24.0, "GLD": 10.0},
        "STAGFLATION": {"DBC": 30.0, "CASH": 60.0, "GLD": 10.0},
        "DEFLATION":   {"EXV3.DE": 36.0, "EXV8.DE": 30.0, "EXV7.DE": 24.0, "GLD": 10.0}
    }

    for zone, alloc_map, c in [("US", alloc_us, col1), ("EU", alloc_eu, col2)]:
        reg = res[f"Regime_{zone}"].iloc[-1]
        if reg in alloc_map:
            weights = alloc_map[reg]
            fig = px.pie(names=list(weights.keys()), values=list(weights.values()), title=f"Allocation {zone} ({reg})")
            c.plotly_chart(fig, use_container_width=True)
        else:
            c.warning(f"Régime {reg} inconnu pour {zone}")

    st.markdown("---")
    st.subheader("Performance Historique et Régimes")

    def plot_perf_with_regimes(zone):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res.index, y=res[f"Strategy_{zone}"], name=f"Stratégie {zone}", line=dict(color='black', width=2)))
        fig.add_trace(go.Scatter(x=res.index, y=res[f"Benchmark_{zone}"], name=f"Benchmark {zone}", line=dict(color='darkred', width=1, dash='dot')))

        regime_series = res[f"Regime_{zone}"]
        changes = regime_series != regime_series.shift(1)
        change_indices = res.index[changes].tolist() + [res.index[-1]]

        for i in range(len(change_indices)-1):
            start = change_indices[i]
            end = change_indices[i+1]
            reg = regime_series.loc[start]
            if reg != "UNKNOWN":
                fig.add_vrect(x0=start, x1=end, fillcolor=REGIME_COLORS.get(reg, "white"), opacity=0.2, layer="below", line_width=0)

        fig.update_layout(title=f"Performance vs Benchmark {zone} (Base 100)", yaxis_type="log", height=500)
        return fig

    st.plotly_chart(plot_perf_with_regimes("US"), use_container_width=True)
    st.plotly_chart(plot_perf_with_regimes("EU"), use_container_width=True)

# --- TAB 2: SIGNALS ---
with tab2:
    st.header("Signaux v2 (Médiane, Buffer Adaptatif, Filtre Persistance)")

    for zone in ["US", "EU"]:
        st.subheader(f"Zone {zone}")
        col1, col2 = st.columns(2)

        # Growth Signal (Price vs Med200)
        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=signals.index, y=signals[f"Price_{zone}"], name="Prix", line=dict(color='black')))
        fig_g.add_trace(go.Scatter(x=signals.index, y=signals[f"Med200_{zone}"], name="Médiane 200j", line=dict(color='blue')))
        # Upper and Lower adaptive bands
        upper_g = signals[f"Med200_{zone}"] * (1 + signals[f"BufGrowth_{zone}"])
        lower_g = signals[f"Med200_{zone}"] * (1 - signals[f"BufGrowth_{zone}"])
        fig_g.add_trace(go.Scatter(x=signals.index, y=upper_g, name="Upper Buffer", line=dict(dash='dash', color='gray')))
        fig_g.add_trace(go.Scatter(x=signals.index, y=lower_g, name="Lower Buffer", line=dict(dash='dash', color='gray')))
        fig_g.update_layout(title=f"Signal Croissance {zone} (v2)", height=400)
        col1.plotly_chart(fig_g, use_container_width=True)

        # Inflation Signal (Ratio vs Med200Ratio)
        fig_i = go.Figure()
        fig_i.add_trace(go.Scatter(x=signals.index, y=signals[f"Ratio_{zone}"], name="Ratio B+/B-", line=dict(color='darkgreen')))
        fig_i.add_trace(go.Scatter(x=signals.index, y=signals[f"Med200Ratio_{zone}"], name="Médiane 200j", line=dict(color='purple')))
        # Upper and Lower adaptive bands
        upper_i = signals[f"Med200Ratio_{zone}"] * (1 + signals[f"BufInfl_{zone}"])
        lower_i = signals[f"Med200Ratio_{zone}"] * (1 - signals[f"BufInfl_{zone}"])
        fig_i.add_trace(go.Scatter(x=signals.index, y=upper_i, name="Upper Buffer", line=dict(dash='dash', color='gray')))
        fig_i.add_trace(go.Scatter(x=signals.index, y=lower_i, name="Lower Buffer", line=dict(dash='dash', color='gray')))
        fig_i.update_layout(title=f"Signal Inflation {zone} (v2)", height=400)
        col2.plotly_chart(fig_i, use_container_width=True)

# --- TAB 3: PERFORMANCE ---
with tab3:
    st.header("Analyse des Performances")

    st.subheader("Statistiques Globales")
    formatted_stats = stats.copy()
    for col in ["CAGR", "Vol", "MaxDD", "HitRate", "Alpha"]:
        if col in formatted_stats.columns:
            formatted_stats[col] = (formatted_stats[col] * 100).map("{:.2f}%".format)
    formatted_stats["Sharpe"] = formatted_stats["Sharpe"].map("{:.2f}".format)
    st.table(formatted_stats)

    st.markdown("---")
    st.subheader("Performance des Secteurs par Régime (Période Globale)")

    # Load Global Sector Performance
    try:
        global_perf = pd.read_csv("sector_perf_TOTAL.csv")
    except:
        global_perf = pd.DataFrame()

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Zone US (Total)**")
        pivot_us = global_perf[global_perf["Zone"] == "US"].pivot(index="Sector", columns="Regime", values="Ann_Return")
        st.dataframe(pivot_us.style.format("{:.2%}") if not pivot_us.empty else pivot_us)
        fig_h_us = px.imshow(pivot_us, text_auto=".1%", title="Heatmap Secteurs US", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig_h_us, use_container_width=True)

    with col2:
        st.write("**Zone EU (Total)**")
        pivot_eu = global_perf[global_perf["Zone"] == "EU"].pivot(index="Sector", columns="Regime", values="Ann_Return")
        st.dataframe(pivot_eu.style.format("{:.2%}") if not pivot_eu.empty else pivot_eu)
        fig_h_eu = px.imshow(pivot_eu, text_auto=".1%", title="Heatmap Secteurs EU", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig_h_eu, use_container_width=True)

    st.markdown("---")
    st.subheader("Signaux & Graphique de Stratégie")
    st.image("strategy_vs_benchmark.png", caption="Comparaison des deux stratégies vs leurs benchmarks respectifs (Log scale)")

    st.markdown("---")
    st.subheader("Analyse Historique par Périodes")
    p_select = st.selectbox("Choisir une période d'analyse", ["P1", "P2", "P3", "P4"])
    p_df = pd.read_csv(f"sector_perf_{p_select}.csv")

    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Zone US ({p_select})**")
        p_pivot_us = p_df[p_df["Zone"] == "US"].pivot(index="Sector", columns="Regime", values="Ann_Return")
        st.dataframe(p_pivot_us.style.format("{:.2%}") if not p_pivot_us.empty else p_pivot_us)
    with col_b:
        st.write(f"**Zone EU ({p_select})**")
        p_pivot_eu = p_df[p_df["Zone"] == "EU"].pivot(index="Sector", columns="Regime", values="Ann_Return")
        st.dataframe(p_pivot_eu.style.format("{:.2%}") if not p_pivot_eu.empty else p_pivot_eu)

# --- TAB 4: METHODOLOGY ---
with tab4:
    st.header("Méthodologie")
    st.markdown("""
    ### 1. Structure du Portefeuille
    - **Poche Or (10%)** : Allocation statique via l'ETF GLD, rééquilibrée mensuellement.
    - **Poche Macro (90%)** : Pilotée dynamiquement par zone (US et EU).
    - **Frais de Transaction** : 0,10% appliqués sur la valeur totale de chaque ligne modifiée (turnover).
    - **Plafond** : Maximum 33,3% par secteur au sein de la poche macro (soit 30% du portefeuille total).

    ### 2. Calcul des Signaux (v2 : Médiane & Buffers Adaptatifs)
    - **Filtre de Persistance** : Une confirmation est requise pour valider tout changement de régime (réduction des faux signaux).
      - **Zone US** : 10 jours (réactif).
      - **Zone Europe** : 15 jours (résistant).
    - **Signal Croissance** : Indice (S&P 500 ou Euro Stoxx 50) vs sa **Médiane 200j**.
      - **Bandes Adaptatives** : Seuil dynamique basé sur 0,5 * Volatilité (60j).
      - Passage à **UP** : Clôture > (1 + Buffer) * Médiane.
      - Passage à **DOWN** : Clôture < (1 - Buffer) * Médiane.
    - **Signal Inflation** : Ratio (Bêta Positif / Bêta Négatif) vs sa **Médiane 200j**.
      - Passage à **UP** : Ratio > (1 + Buffer) * Médiane.
      - Passage à **DOWN** : Ratio < (1 - Buffer) * Médiane.

    ### 3. Composition des Portefeuilles Bêta
    - **Bêta Positif** (Sensibles Inflation) : Energy (35%), Financials (25%), Materials (20%), Commodities (20%).
    - **Bêta Négatif** (Défensifs/Growth) : Growth Tech (40%), Cons. Discretionary (30%), Utilities (15%), Growth Real Estate (15%).

    ### 4. Matrice d'Allocation
    - Quatre régimes définis par la combinaison des signaux :
      - **GOLDILOCKS** : Croissance UP / Inflation DOWN
      - **REFLATION** : Croissance UP / Inflation UP
      - **STAGFLATION** : Croissance DOWN / Inflation UP
      - **DEFLATION** : Croissance DOWN / Inflation DOWN
    """)
