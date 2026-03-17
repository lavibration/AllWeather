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

# Colors
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

    current_regime_us = backtest_results["Regime_US"].iloc[-1]
    current_regime_eu = backtest_results["Regime_EU"].iloc[-1]

    with col1:
        st.subheader("États-Unis (S&P 500)")
        st.markdown(f"**Régime Actuel :** :{COLORS.get(current_regime_us, 'white')}[{current_regime_us}]")

    with col2:
        st.subheader("Europe (Euro Stoxx 50)")
        st.markdown(f"**Régime Actuel :** :{COLORS.get(current_regime_eu, 'white')}[{current_regime_eu}]")

    st.divider()

    # Pie Charts
    col1, col2 = st.columns(2)

    # Mocking target allocation for display (last row)
    # In a real app, we'd want the exact weights from the last day
    # We can reconstruct it or pass it through a CSV

    def get_pie_chart(regime, zone):
        # Simplification for the UI
        allocs = {
            "US": {
                "GOLDILOCKS": {"XLK (Tech)": 33.3, "XLY (Disc)": 33.3, "XLF (Fin)": 33.3},
                "REFLATION": {"XLE (Eng)": 33.3, "XLB (Mat)": 33.3, "XLF (Fin)": 33.3},
                "STAGFLATION": {"Commodities": 33.3, "CASH": 66.7},
                "DEFLATION": {"XLY (Disc)": 33.3, "XLRE (RE)": 33.3, "XLC (Comm)": 33.3}
            },
            "EU": {
                "GOLDILOCKS": {"Santé": 33.3, "Tech": 33.3, "Real Estate": 33.3},
                "REFLATION": {"Banks": 33.3, "Basic Resources": 33.3, "Commodities": 33.3},
                "STAGFLATION": {"Commodities": 33.3, "CASH": 66.7},
                "DEFLATION": {"Santé": 33.3, "Automobile": 33.3, "Telecoms": 33.3}
            }
        }
        data = allocs[zone].get(regime, {"CASH": 100})
        df = pd.DataFrame(list(data.items()), columns=["Asset", "Weight"])
        fig = px.pie(df, values="Weight", names="Asset", title=f"Allocation Cible {zone} ({regime})")
        return fig

    with col1:
        st.plotly_chart(get_pie_chart(current_regime_us, "US"))
    with col2:
        st.plotly_chart(get_pie_chart(current_regime_eu, "EU"))

    # Price charts with regime background
    st.subheader("Historique des Régimes")

    def plot_price_with_regime(zone):
        ticker = "US" if zone == "US" else "EU"
        price = signals_analysis[f"Price_{ticker}"]
        regime = backtest_results[f"Regime_{ticker}"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price.index, y=price, name=f"Indice {zone}", line=dict(color='black')))

        # Add background colors
        # To optimize, we find continuous blocks of the same regime
        regime_diff = (regime != regime.shift()).cumsum()
        for r_id in regime_diff.unique():
            subset = regime[regime_diff == r_id]
            r_name = subset.iloc[0]
            if r_name != "UNKNOWN":
                fig.add_vrect(
                    x0=subset.index[0], x1=subset.index[-1],
                    fillcolor=COLORS.get(r_name, "white"), opacity=0.2,
                    layer="below", line_width=0,
                )

        fig.update_layout(title=f"Prix {zone} et Régimes", xaxis_title="Date", yaxis_title="Prix")
        return fig

    st.plotly_chart(plot_price_with_regime("US"), use_container_width=True)
    st.plotly_chart(plot_price_with_regime("EU"), use_container_width=True)

# --- TAB 2: SIGNALS ---
with tab2:
    st.header("Analyse des Signaux")

    # Growth
    st.subheader("Signal Croissance (Indice vs MA200)")
    zone_choice = st.selectbox("Choisir la Zone", ["US", "EU"])

    p = signals_analysis[f"Price_{zone_choice}"]
    m = signals_analysis[f"MA200_{zone_choice}"]

    fig_g = go.Figure()
    fig_g.add_trace(go.Scatter(x=p.index, y=p, name="Prix"))
    fig_g.add_trace(go.Scatter(x=m.index, y=m, name="MA200", line=dict(dash='solid')))
    fig_g.add_trace(go.Scatter(x=m.index, y=m*1.01, name="+1% Hystérésis", line=dict(dash='dash', color='green')))
    fig_g.add_trace(go.Scatter(x=m.index, y=m*0.99, name="-1% Hystérésis", line=dict(dash='dash', color='red')))
    st.plotly_chart(fig_g, use_container_width=True)

    # Inflation
    st.subheader("Signal Inflation (Ratio Beta vs Médiane)")
    ratio = signals_analysis["Inflation_Ratio"]
    med = signals_analysis["Inflation_Median"]

    fig_i = go.Figure()
    fig_i.add_trace(go.Scatter(x=ratio.index, y=ratio, name="Ratio (Pos/Neg)"))
    fig_i.add_trace(go.Scatter(x=med.index, y=med, name="Médiane 200j"))
    fig_i.add_trace(go.Scatter(x=med.index, y=med*1.01, name="+1% Hystérésis", line=dict(dash='dash', color='green')))
    fig_i.add_trace(go.Scatter(x=med.index, y=med*0.99, name="-1% Hystérésis", line=dict(dash='dash', color='red')))
    st.plotly_chart(fig_i, use_container_width=True)

# --- TAB 3: PERFORMANCE ---
with tab3:
    st.header("Analyse des Performances")

    # Statistics Table
    st.subheader("Statistiques de la Stratégie")
    st.table(strategy_stats)

    # Cumulative Performance Chart
    st.subheader("Performance Cumulative (Stratégie vs Benchmark)")
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=backtest_results.index, y=backtest_results["Portfolio_Value"], name="Stratégie"))
    fig_perf.add_trace(go.Scatter(x=backtest_results.index, y=backtest_results["Benchmark_Value"], name="Benchmark (S&P 500)"))
    st.plotly_chart(fig_perf, use_container_width=True)

    # Heatmap
    st.subheader("Performance des Secteurs par Régime")
    pivot_perf = sector_performance.pivot_table(index="Sector", columns="Regime", values="Avg_Annual_Return")
    fig_heat = px.imshow(pivot_perf, text_auto=True, color_continuous_scale="RdYlGn", title="Annualized Avg Return per Regime")
    st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB 4: METHODOLOGY ---
with tab4:
    st.header("Méthodologie")
    st.markdown("""
    ### 1. Structure du Portefeuille
    - **10 % OR** (ETF GLD) : Rééquilibré mensuellement.
    - **90 % POCHE MACRO** : Pilotée dynamiquement par zone (US et EU).

    ### 2. Calcul des Signaux (Hystérésis 1,0 %)
    - **Signal Croissance** : Indice vs sa Moyenne Mobile 200j.
        - UP : Clôture > 1,01 * MM200
        - DOWN : Clôture < 0,99 * MM200
    - **Signal Inflation** : Ratio (Bêta Positif / Bêta Négatif) vs sa Médiane 200j.
        - UP : Ratio > 1,01 * Médiane
        - DOWN : Ratio < 0,99 * Médiane

    ### 3. Composition des Portefeuilles Bêta
    - **Bêta Positif** : Energy (35%), Financials/Banks (25%), Materials (20%), Commodities (20%).
    - **Bêta Négatif** : Growth Tech (40%), Cons. Discretionary (30%), Utilities (15%), Growth Real Estate (15%).

    ### 4. Matrice d'Allocation (Max 33,3% par secteur)
    - **GOLDILOCKS** (C+ / I-) : Tech, Cons. Disc, Financials (US) / Santé, Tech, RE (EU).
    - **REFLATION** (C+ / I+) : Energy, Materials, Financials (US) / Banks, Basic Res, Commodities (EU).
    - **STAGFLATION** (C- / I+) : Commodities (33.3%), CASH (66.7%).
    - **DEFLATION** (C- / I-) : Cons. Disc, Real Estate, Comm (US) / Santé, Automobile, Telecoms (EU).

    ### 5. Frais & Contraintes
    - **Frais de transaction** : 0,10 % sur la valeur totale de chaque changement de poids.
    - **Plafond** : Max 33,3% par secteur dans la poche macro.
    """)
