import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Financial Strategy Dashboard", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    backtest_results = pd.read_csv("backtest_results.csv", index_col="Date", parse_dates=True)
    signals_analysis = pd.read_csv("signals_analysis.csv", index_col="Date", parse_dates=True)
    sector_performance = pd.read_csv("sector_performance.csv")
    strategy_stats = pd.read_csv("strategy_stats.csv")
    return backtest_results, signals_analysis, sector_performance, strategy_stats

try:
    backtest_results, signals_analysis, sector_performance, strategy_stats = load_data()
except Exception as e:
    st.error(f"Error loading CSV files: {e}. Please run the data engine first.")
    st.stop()

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Cockpit d'Exécution",
    "📊 Signaux & Hystérésis",
    "📈 Performance",
    "📜 Méthodologie"
])

COLORS = {
    "GOLDILOCKS": "green",
    "REFLATION": "blue",
    "STAGFLATION": "orange",
    "DEFLATION": "gray",
    "UNKNOWN": "white"
}

# --- TAB 1: COCKPIT ---
with tab1:
    st.header("Cockpit d'Exécution")
    col1, col2 = st.columns(2)
    cur_us = backtest_results["Regime_US"].iloc[-1]
    cur_eu = backtest_results["Regime_EU"].iloc[-1]
    with col1:
        st.subheader("États-Unis (S&P 500)")
        st.markdown(f"**Régime Actuel :** :{COLORS.get(cur_us, 'white')}[{cur_us}]")
    with col2:
        st.subheader("Europe (Euro Stoxx 50)")
        st.markdown(f"**Régime Actuel :** :{COLORS.get(cur_eu, 'white')}[{cur_eu}]")
    st.divider()

    col1, col2 = st.columns(2)
    def get_pie_chart(regime, zone):
        allocs = {
            "US": {
                "GOLDILOCKS": {"Tech (XLK)": 33.3, "Cons. Disc (XLY)": 33.3, "Finance (XLF)": 33.3},
                "REFLATION": {"Energie (XLE)": 33.3, "Matériaux (XLB)": 33.3, "Finance (XLF)": 33.3},
                "STAGFLATION": {"Commodities": 33.3, "CASH": 66.7},
                "DEFLATION": {"Cons. Disc (XLY)": 33.3, "Immobilier (XLRE)": 33.3, "Comm (XLC)": 33.3}
            },
            "EU": {
                "GOLDILOCKS": {"Santé": 33.3, "Tech": 33.3, "Immobilier": 33.3},
                "REFLATION": {"Banques": 33.3, "Ressources": 33.3, "Commodities": 33.3},
                "STAGFLATION": {"Commodities": 33.3, "CASH": 66.7},
                "DEFLATION": {"Santé": 33.3, "Automobile": 33.3, "Télécoms": 33.3}
            }
        }
        data = allocs[zone].get(regime, {"CASH": 100})
        df = pd.DataFrame(list(data.items()), columns=["Asset", "Weight"])
        return px.pie(df, values="Weight", names="Asset", title=f"Allocation Cible {zone} ({regime})")

    with col1: st.plotly_chart(get_pie_chart(cur_us, "US"))
    with col2: st.plotly_chart(get_pie_chart(cur_eu, "EU"))

    st.subheader("Historique des Régimes")
    def plot_price_with_regime(zone):
        ticker = "US" if zone == "US" else "EU"
        df_plot = pd.concat([signals_analysis[f"Price_{ticker}"], backtest_results[f"Regime_{ticker}"]], axis=1).reset_index()
        df_plot.columns = ["Date", "Prix", "Regime"]
        fig = px.scatter(df_plot, x="Date", y="Prix", color="Regime", color_discrete_map={
            "GOLDILOCKS": "green", "REFLATION": "blue", "STAGFLATION": "orange", "DEFLATION": "gray", "UNKNOWN": "lightgray"
        }, title=f"Indice {zone} coloré par Régime")
        fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["Prix"], mode="lines", line=dict(color="black", width=1), showlegend=False))
        fig.update_traces(marker=dict(size=4))
        return fig
    st.plotly_chart(plot_price_with_regime("US"), use_container_width=True)
    st.plotly_chart(plot_price_with_regime("EU"), use_container_width=True)

# --- TAB 2: SIGNALS ---
with tab2:
    st.header("Analyse des Signaux")
    st.subheader("Signal Croissance (Indice vs MA200)")
    zone_choice = st.selectbox("Choisir la Zone", ["US", "EU"])
    p, m = signals_analysis[f"Price_{zone_choice}"], signals_analysis[f"MA200_{zone_choice}"]
    fig_g = go.Figure()
    fig_g.add_trace(go.Scatter(x=p.index, y=p, name="Prix"))
    fig_g.add_trace(go.Scatter(x=m.index, y=m, name="MA200"))
    fig_g.add_trace(go.Scatter(x=m.index, y=m*1.01, name="+1% Hystérésis", line=dict(dash='dash', color='green')))
    fig_g.add_trace(go.Scatter(x=m.index, y=m*0.99, name="-1% Hystérésis", line=dict(dash='dash', color='red')))
    st.plotly_chart(fig_g, use_container_width=True)

    st.subheader("Signal Inflation (Ratio Beta vs Médiane)")
    ratio, med = signals_analysis["Inflation_Ratio"], signals_analysis["Inflation_Median"]
    fig_i = go.Figure()
    fig_i.add_trace(go.Scatter(x=ratio.index, y=ratio, name="Ratio (Pos/Neg)"))
    fig_i.add_trace(go.Scatter(x=med.index, y=med, name="Médiane 200j"))
    fig_i.add_trace(go.Scatter(x=med.index, y=med*1.01, name="+1% Hystérésis", line=dict(dash='dash', color='green')))
    fig_i.add_trace(go.Scatter(x=med.index, y=med*0.99, name="-1% Hystérésis", line=dict(dash='dash', color='red')))
    st.plotly_chart(fig_i, use_container_width=True)

# --- TAB 3: PERFORMANCE ---
with tab3:
    st.header("Analyse des Performances")

    st.subheader("Statistiques de la Stratégie")
    tabs_stats = st.tabs(["Global", "Poche US", "Poche EU"])
    for i, port in enumerate(["Global", "US", "EU"]):
        with tabs_stats[i]:
            s = strategy_stats[strategy_stats["Portfolio"] == port].iloc[0]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("CAGR", f"{s['CAGR']*100:.2f}%")
            c2.metric("Volatilité", f"{s['Volatility']*100:.2f}%")
            c3.metric("Sharpe", f"{s['Sharpe']:.2f}")
            c4.metric("Max Drawdown", f"{s['Max_Drawdown']*100:.2f}%")
            c5.metric("Hit Rate", f"{s['Hit_Rate']*100:.2f}%")

    st.subheader("Performance Cumulative")
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=backtest_results.index, y=backtest_results["Value_Global"], name="Stratégie Globale", line=dict(width=3)))
    fig_perf.add_trace(go.Scatter(x=backtest_results.index, y=backtest_results["Value_US"], name="Poche Macro US", line=dict(dash='dash', width=1)))
    fig_perf.add_trace(go.Scatter(x=backtest_results.index, y=backtest_results["Value_EU"], name="Poche Macro EU", line=dict(dash='dash', width=1)))
    fig_perf.add_trace(go.Scatter(x=backtest_results.index, y=backtest_results["Benchmark_Value"], name="Benchmark (S&P 500)", line=dict(dash='dot', color='rgba(100,100,100,0.5)')))
    fig_perf.update_layout(hovermode="x unified", yaxis_title="Valeur du Portefeuille (Base 1000)")
    st.plotly_chart(fig_perf, use_container_width=True)

    st.subheader("Performance des Secteurs par Régime")
    zone_heat = st.radio("Zone Heatmap", ["US", "EU"], horizontal=True)
    filtered_perf = sector_performance[sector_performance["Zone"] == zone_heat]
    pivot_perf = filtered_perf.pivot_table(index="Sector", columns="Regime", values="Avg_Annual_Return") * 100
    fig_heat = px.imshow(pivot_perf, text_auto=".2f", color_continuous_scale="RdYlGn",
                         title=f"Rendement Annuel Moyen (%) - Zone {zone_heat}", labels=dict(color="Rendement (%)"))
    fig_heat.update_layout(height=800)
    st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB 4: METHODOLOGY ---
with tab4:
    st.header("Méthodologie")
    st.markdown("""
    ### 1. Structure du Portefeuille (Statique)
    - **10 % OR** (ETF GLD) : Allocation fixe, rééquilibrée mensuellement.
    - **45 % STRATÉGIE US** : Pilotée exclusivement par les signaux US.
    - **45 % STRATÉGIE EU** : Pilotée exclusivement par les signaux EU.

    ### 2. Calcul des Signaux (Hystérésis 1,0 %)
    - **Anti-Look-Ahead** : Les poids du jour *T* sont calculés sur la base des signaux arrêtés au jour *T-1*.
    - **Croissance** : Indice vs sa MM200 (±1% Hystérésis).
    - **Inflation** : Ratio (Bêta+ / Bêta-) vs sa Médiane 200j (±1% Hystérésis).

    ### 3. Composition & Contraintes
    - **Bêta Positif** : Energy, Finance, Materials, Commodities.
    - **Bêta Négatif** : Tech, Cons. Disc, Utilities, Real Estate.
    - **Frais** : 0,10 % de frais de transaction sur tout changement de poids (Turnover).
    - **Max Secteur** : 33,3% au sein de chaque poche macro.
    """)
