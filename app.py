import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Financial Strategy Dashboard - Mirror Architecture", layout="wide")

@st.cache_data
def load_data():
    bt = pd.read_csv("backtest_results.csv", index_col="Date", parse_dates=True)
    sig = pd.read_csv("signals_analysis.csv", index_col="Date", parse_dates=True)
    perf = pd.read_csv("sector_performance.csv")
    stats = pd.read_csv("strategy_stats.csv")
    return bt, sig, perf, stats

try:
    bt, sig, perf, stats = load_data()
except Exception as e:
    st.error(f"Error loading CSV files: {e}. Run the engine first.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["🚀 Cockpit", "📊 Signaux", "📈 Performance", "📜 Méthodologie"])
COLORS = {"GOLDILOCKS": "rgba(0, 255, 0, 0.2)", "REFLATION": "rgba(0, 0, 255, 0.2)", "STAGFLATION": "rgba(255, 165, 0, 0.2)", "DEFLATION": "rgba(128, 128, 128, 0.2)", "UNKNOWN": "rgba(255, 255, 255, 0)", "CIRCUIT_BREAKER": "rgba(255, 0, 0, 0.3)"}
DOT_COLORS = {"GOLDILOCKS": "green", "REFLATION": "blue", "STAGFLATION": "orange", "DEFLATION": "gray", "UNKNOWN": "white", "CIRCUIT_BREAKER": "red"}

with tab1:
    st.header("Cockpit Miroir (US vs EU)")
    col1, col2 = st.columns(2)

    # Check if Circuit Breaker is active in Backtest for current status display
    cur_us = bt["Regime_Backtest_US"].iloc[-1]
    cur_eu = bt["Regime_Backtest_EU"].iloc[-1]

    with col1:
        st.subheader("Poche États-Unis")
        if cur_us == "CIRCUIT_BREAKER":
            st.error("⚠️ CIRCUIT BREAKER ACTIF (100% CASH)")
        else:
            st.markdown(f"**Régime Actuel :** :{DOT_COLORS.get(cur_us)}[{cur_us}]")
    with col2:
        st.subheader("Poche Europe")
        if cur_eu == "CIRCUIT_BREAKER":
            st.error("⚠️ CIRCUIT BREAKER ACTIF (100% CASH)")
        else:
            st.markdown(f"**Régime Actuel :** :{DOT_COLORS.get(cur_eu)}[{cur_eu}]")
    st.markdown("---")

    col1, col2 = st.columns(2)
    def get_pie(regime, zone):
        if regime == "CIRCUIT_BREAKER":
            return px.pie(pd.DataFrame([["CASH", 100.0]], columns=["Asset", "Weight"]), values="Weight", names="Asset", title=f"Allocation Cible {zone} (SAFE MODE)")

        base = {"Or (GLD)": 10.0}
        macro_map = {
            "US": {
                "GOLDILOCKS": {"XLK (Tech)": 30.0, "XLY (Conso)": 30.0, "XLV (Health)": 30.0},
                "REFLATION": {"XLE (Eng)": 30.0, "DBC (Comm)": 30.0, "XLU (Util)": 30.0},
                "STAGFLATION": {"XLU (Util)": 30.0, "DBC (Comm)": 30.0, "XLV (Health)": 30.0},
                "DEFLATION": {"XLK (Tech)": 30.0, "XLY (Conso)": 30.0, "XLU (Util)": 30.0}
            },
            "EU": {
                "GOLDILOCKS": {"EXV3 (Tech)": 30.0, "EXV8 (Conso)": 30.0, "EXV7 (Telco)": 30.0},
                "REFLATION": {"EXV1 (Banks)": 30.0, "EXV6 (Res)": 30.0, "EXW1 (Ins)": 30.0},
                "STAGFLATION": {"EXV6 (Res)": 30.0, "EXV7 (Telco)": 30.0, "EXV9 (Util)": 30.0},
                "DEFLATION": {"EXW1 (Ins)": 30.0, "EXV3 (Tech)": 30.0, "EXV7 (Telco)": 30.0}
            }
        }
        alloc = macro_map[zone].get(regime, {"CASH": 90.0})
        base.update(alloc)
        return px.pie(pd.DataFrame(list(base.items()), columns=["Asset", "Weight"]), values="Weight", names="Asset", title=f"Allocation Cible {zone} ({regime})")

    with col1: st.plotly_chart(get_pie(cur_us, "US"))
    with col2: st.plotly_chart(get_pie(cur_eu, "EU"))

    def plot_price_with_bg(zone):
        p = sig[f"Price_{zone}"]
        # Use regime from backtest for background coloring to include Circuit Breaker
        reg = bt[f"Regime_Backtest_{zone}"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p.index, y=p, name="Prix", line=dict(color='black', width=1.5)))

        # Add background coloring (Original Architecture Style)
        change = (reg != reg.shift(1))
        starts = reg.index[change]
        for i in range(len(starts)):
            start = starts[i]
            end = starts[i+1] if i+1 < len(starts) else reg.index[-1]
            r_val = reg.loc[start]
            if r_val in COLORS:
                fig.add_vrect(x0=start, x1=end, fillcolor=COLORS[r_val], opacity=0.5, layer="below", line_width=0, annotation_text=r_val if i==0 else "")

        fig.update_layout(title=f"Historique des Prix et Régimes {zone}", xaxis_title="Date", yaxis_title="Prix")
        return fig

    st.plotly_chart(plot_price_with_bg("US"), use_container_width=True)
    st.plotly_chart(plot_price_with_bg("EU"), use_container_width=True)

with tab2:
    st.header("Signaux & Hystérésis (±1%)")
    z = st.radio("Zone", ["US", "EU"], horizontal=True, key="sig_zone")
    st.subheader("Signal Croissance")
    fig_g = go.Figure()
    fig_g.add_trace(go.Scatter(x=sig.index, y=sig[f"Price_{z}"], name="Prix"))
    fig_g.add_trace(go.Scatter(x=sig.index, y=sig[f"MA200_{z}"], name="MM200"))
    fig_g.add_trace(go.Scatter(x=sig.index, y=sig[f"MA200_{z}"]*1.01, name="+1%", line=dict(dash='dash', color='green')))
    fig_g.add_trace(go.Scatter(x=sig.index, y=sig[f"MA200_{z}"]*0.99, name="-1%", line=dict(dash='dash', color='red')))
    st.plotly_chart(fig_g, use_container_width=True)

    st.subheader("Signal Inflation")
    fig_i = go.Figure()
    fig_i.add_trace(go.Scatter(x=sig.index, y=sig[f"Ratio_{z}"], name="Ratio Beta"))
    fig_i.add_trace(go.Scatter(x=sig.index, y=sig[f"Median_{z}"], name="Médiane 200j"))
    fig_i.add_trace(go.Scatter(x=sig.index, y=sig[f"Median_{z}"]*1.01, name="+1%", line=dict(dash='dash', color='green')))
    fig_i.add_trace(go.Scatter(x=sig.index, y=sig[f"Median_{z}"]*0.99, name="-1%", line=dict(dash='dash', color='red')))
    st.plotly_chart(fig_i, use_container_width=True)

with tab3:
    st.header("Analyse des Performances")
    c1, c2 = st.columns(2)
    for i, z in enumerate(["US", "EU"]):
        with [c1, c2][i]:
            s = stats[stats["Pocket"] == z].iloc[0]
            st.metric(f"CAGR {z}", f"{s['CAGR']*100:.2f}%")
            st.metric(f"Sharpe {z}", f"{s['Sharpe']:.2f}")
            st.metric(f"Max Drawdown {z}", f"{s['Max_Drawdown']*100:.2f}%")
            st.metric(f"Hit Rate {z}", f"{s['Hit_Rate']*100:.2f}%")

    st.subheader("Performance Cumulative")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt.index, y=bt["Value_US"], name="Poche US"))
    fig.add_trace(go.Scatter(x=bt.index, y=bt["Value_EU"], name="Poche EU"))
    fig.add_trace(go.Scatter(x=bt.index, y=bt["Benchmark_Value"], name="Benchmark (S&P 500)", line=dict(dash='dot')))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Heatmap des Secteurs")
    z_h = st.radio("Secteurs Zone", ["US", "EU"], horizontal=True, key="heat")
    p_f = perf[perf["Zone"] == z_h].pivot_table(index="Sector", columns="Regime", values="Avg_Annual_Return") * 100
    st.plotly_chart(px.imshow(p_f, text_auto=".2f", color_continuous_scale="RdYlGn"), use_container_width=True)

with tab4:
    st.header("Méthodologie")
    st.markdown("""
    ### 1. Architecture Miroir
    - **US** : 10% Or / 90% Macro US.
    - **EU** : 10% Or / 90% Macro EU.

    ### 2. Signaux Hystérésis
    Transitions validées à +/- 1,0 % pour éviter le bruit.

    ### 3. Allocation Optimisée
    Poids issus du scan large univers et validés par audit statistique.

    ### 4. Circuit Breaker (Nouveau)
    - **Trigger** : Si l'indice (S&P 500 ou Stoxx 50) perd **7% en 5 jours**.
    - **Action** : Passage immédiat à **100% CASH**.
    - **Reset** : Ré-initialisation uniquement lors du prochain changement de régime.

    ### 5. Paramètres
    - Frais : 0,10 % par transaction.
    - Cap : 33,3 % max par secteur (dans les 90 %).
    """)
