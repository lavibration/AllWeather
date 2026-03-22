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
    sector_perf = pd.read_csv("sector_performance.csv")
    stats = pd.read_csv("strategy_stats.csv")
    return res, signals, sector_perf, stats

res, signals, sector_perf, stats = load_data()

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

    st.divider()

    st.subheader("Allocations Cibles (Macro Pocket 90% + Gold 10%)")
    col1, col2 = st.columns(2)

    # Target Allocations based on strict rules
    alloc_us = {
        "GOLDILOCKS": {"XLK": 30, "XLY": 30, "XLF": 30, "GLD": 10},
        "REFLATION": {"XLE": 30, "XLB": 30, "XLF": 30, "GLD": 10},
        "STAGFLATION": {"Commodities (DBC)": 30, "CASH": 60, "GLD": 10},
        "DEFLATION": {"XLY": 30, "XLRE": 30, "XLC": 30, "GLD": 10}
    }

    alloc_eu = {
        "GOLDILOCKS": {"Santé (EXV4)": 30, "Tech (EXV3)": 30, "Real Estate (EXI5)": 30, "GLD": 10},
        "REFLATION": {"Banks (EXV1)": 30, "Resources (EXV6)": 30, "Commodities (DBC)": 30, "GLD": 10},
        "STAGFLATION": {"Commodities (DBC)": 30, "CASH": 60, "GLD": 10},
        "DEFLATION": {"Santé (EXV4)": 30, "Auto (EXV2)": 30, "Telecom (EXV7)": 30, "GLD": 10}
    }

    for zone, alloc_map, c in [("US", alloc_us, col1), ("EU", alloc_eu, col2)]:
        reg = res[f"Regime_{zone}"].iloc[-1]
        if reg in alloc_map:
            weights = alloc_map[reg]
            fig = px.pie(names=list(weights.keys()), values=list(weights.values()), title=f"Allocation {zone} ({reg})")
            c.plotly_chart(fig, use_container_width=True)
        else:
            c.warning(f"Régime {reg} inconnu pour {zone}")

    st.divider()
    st.subheader("Performance Historique et Régimes")

    def plot_perf_with_regimes(zone):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res.index, y=res[f"Strategy_{zone}"], name=f"Stratégie {zone}", line=dict(color='black', width=2)))
        fig.add_trace(go.Scatter(x=res.index, y=res[f"Benchmark_{zone}"], name=f"Benchmark {zone}", line=dict(color='darkred', width=1, dash='dot')))

        # Add background colors for regimes
        regime_series = res[f"Regime_{zone}"]
        changes = regime_series != regime_series.shift(1)
        change_indices = res.index[changes].tolist() + [res.index[-1]]

        for i in range(len(change_indices)-1):
            start = change_indices[i]
            end = change_indices[i+1]
            reg = regime_series.loc[start]
            fig.add_vrect(x0=start, x1=end, fillcolor=REGIME_COLORS.get(reg, "white"), opacity=0.2, layer="below", line_width=0)

        fig.update_layout(title=f"Performance vs Benchmark {zone} (Log Scale)", yaxis_type="log", height=500)
        return fig

    st.plotly_chart(plot_perf_with_regimes("US"), use_container_width=True)
    st.plotly_chart(plot_perf_with_regimes("EU"), use_container_width=True)

# --- TAB 2: SIGNALS ---
with tab2:
    st.header("Signaux & Hystérésis")

    for zone in ["US", "EU"]:
        st.subheader(f"Zone {zone}")
        col1, col2 = st.columns(2)

        # Growth Signal (Price vs MA200)
        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=signals.index, y=signals[f"Price_{zone}"], name="Prix", line=dict(color='black')))
        fig_g.add_trace(go.Scatter(x=signals.index, y=signals[f"MA200_{zone}"], name="MA200", line=dict(color='blue')))
        # Hysteresis bands
        fig_g.add_trace(go.Scatter(x=signals.index, y=signals[f"MA200_{zone}"]*1.01, name="+1% Band", line=dict(dash='dash', color='gray')))
        fig_g.add_trace(go.Scatter(x=signals.index, y=signals[f"MA200_{zone}"]*0.99, name="-1% Band", line=dict(dash='dash', color='gray')))
        fig_g.update_layout(title=f"Signal Croissance {zone}", height=400)
        col1.plotly_chart(fig_g, use_container_width=True)

        # Inflation Signal (Ratio vs Median)
        fig_i = go.Figure()
        fig_i.add_trace(go.Scatter(x=signals.index, y=signals[f"Ratio_{zone}"], name="Ratio B+/B-", line=dict(color='darkgreen')))
        fig_i.add_trace(go.Scatter(x=signals.index, y=signals[f"Median200_{zone}"], name="Médiane 200j", line=dict(color='purple')))
        # Hysteresis bands
        fig_i.add_trace(go.Scatter(x=signals.index, y=signals[f"Median200_{zone}"]*1.01, name="+1% Band", line=dict(dash='dash', color='gray')))
        fig_i.add_trace(go.Scatter(x=signals.index, y=signals[f"Median200_{zone}"]*0.99, name="-1% Band", line=dict(dash='dash', color='gray')))
        fig_i.update_layout(title=f"Signal Inflation {zone}", height=400)
        col2.plotly_chart(fig_i, use_container_width=True)

# --- TAB 3: PERFORMANCE ---
with tab3:
    st.header("Analyse des Performances")

    st.subheader("Statistiques de la Stratégie")
    formatted_stats = stats.copy()
    for col in ["CAGR", "Vol", "MaxDD", "HitRate"]:
        formatted_stats[col] = (formatted_stats[col] * 100).map("{:.2f}%".format)
    formatted_stats["Sharpe"] = formatted_stats["Sharpe"].map("{:.2f}".format)
    st.table(formatted_stats)

    st.divider()
    st.subheader("Performance des Secteurs par Régime")

    pivot_us = sector_perf[sector_perf["Zone"] == "US"].pivot(index="Sector", columns="Regime", values="Ann_Return")
    pivot_eu = sector_perf[sector_perf["Zone"] == "EU"].pivot(index="Sector", columns="Regime", values="Ann_Return")

    col1, col2 = st.columns(2)
    fig_h_us = px.imshow(pivot_us, text_auto=".1%", title="Heatmap Secteurs US", color_continuous_scale="RdYlGn")
    col1.plotly_chart(fig_h_us, use_container_width=True)

    fig_h_eu = px.imshow(pivot_eu, text_auto=".1%", title="Heatmap Secteurs EU", color_continuous_scale="RdYlGn")
    col2.plotly_chart(fig_h_eu, use_container_width=True)

# --- TAB 4: METHODOLOGY ---
with tab4:
    st.header("Méthodologie")
    st.markdown("""
    ### 1. Structure du Portefeuille
    - **Poche Or (10%)** : Allocation statique via l'ETF GLD, rééquilibrée mensuellement.
    - **Poche Macro (90%)** : Pilotée dynamiquement par zone (US et EU).
    - **Frais de Transaction** : 0,10% appliqués sur la valeur totale de chaque ligne modifiée (turnover).
    - **Plafond** : Maximum 33,3% par secteur au sein de la poche macro (soit 30% du portefeuille total).

    ### 2. Calcul des Signaux (Hystérésis 1,0 %)
    - **Signal Croissance** : Indice (S&P 500 ou Euro Stoxx 50) vs sa Moyenne Mobile 200j.
      - Passage à **UP** : Clôture > 1,01 * MM200.
      - Passage à **DOWN** : Clôture < 0,99 * MM200.
    - **Signal Inflation** : Ratio (Bêta Positif / Bêta Négatif) vs sa Médiane 200j.
      - Passage à **UP** : Ratio > 1,01 * Médiane.
      - Passage à **DOWN** : Ratio < 0,99 * Médiane.

    ### 3. Composition des Portefeuilles Bêta
    - **Bêta Positif** (Sensibles Inflation) : Energy (35%), Financials/Banks (25%), Materials (20%), Commodities (20%).
    - **Bêta Négatif** (Défensifs/Growth) : Growth Tech (40%), Cons. Discretionary (30%), Utilities (15%), Growth Real Estate (15%).

    ### 4. Matrice d'Allocation
    - Quatre régimes définis par la combinaison des signaux :
      - **GOLDILOCKS** : Croissance UP / Inflation DOWN
      - **REFLATION** : Croissance UP / Inflation UP
      - **STAGFLATION** : Croissance DOWN / Inflation UP
      - **DEFLATION** : Croissance DOWN / Inflation DOWN
    """)
