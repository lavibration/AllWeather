import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
COLORS = {"GOLDILOCKS": "green", "REFLATION": "blue", "STAGFLATION": "orange", "DEFLATION": "gray", "UNKNOWN": "white"}

with tab1:
    st.header("Cockpit Miroir (US vs EU)")
    col1, col2 = st.columns(2)
    # Using 'Regime' from signals_analysis for the "Live" regime
    cur_us, cur_eu = sig["Regime_US"].iloc[-1], sig["Regime_EU"].iloc[-1]
    with col1:
        st.subheader("Poche États-Unis")
        st.markdown(f"**Régime Actuel :** :{COLORS.get(cur_us)}[{cur_us}]")
    with col2:
        st.subheader("Poche Europe")
        st.markdown(f"**Régime Actuel :** :{COLORS.get(cur_eu)}[{cur_eu}]")
    st.divider()

    col1, col2 = st.columns(2)
    def get_pie(regime, zone):
        base = {"Or (GLD)": 10.0}
        macro_map = {
            "US": {
                "GOLDILOCKS": {"Tech (XLK)": 30, "Disc (XLY)": 30, "Fin (XLF)": 30},
                "REFLATION": {"Eng (XLE)": 30, "Mat (XLB)": 30, "Fin (XLF)": 30},
                "STAGFLATION": {"Commodities": 30, "CASH": 60},
                "DEFLATION": {"Disc (XLY)": 30, "RE (XLRE)": 30, "Comm (XLC)": 30}
            },
            "EU": {
                "GOLDILOCKS": {"Santé": 30, "Tech": 30, "RE": 30},
                "REFLATION": {"Banks": 30, "Resources": 30, "Commodities": 30},
                "STAGFLATION": {"Commodities": 30, "CASH": 60},
                "DEFLATION": {"Santé": 30, "Auto": 30, "Telecoms": 30}
            }
        }
        alloc = macro_map[zone].get(regime, {"CASH": 90})
        base.update(alloc)
        return px.pie(pd.DataFrame(list(base.items()), columns=["Asset", "Weight"]), values="Weight", names="Asset", title=f"Allocation {zone} ({regime})")

    with col1: st.plotly_chart(get_pie(cur_us, "US"))
    with col2: st.plotly_chart(get_pie(cur_eu, "EU"))

    def plot_regime_hist(zone):
        p, r = sig[f"Price_{zone}"], sig[f"Regime_{zone}"]
        df = pd.concat([p, r], axis=1).reset_index()
        df.columns = ["Date", "Price", "Regime"]
        fig = px.scatter(df, x="Date", y="Price", color="Regime", color_discrete_map=COLORS, title=f"Historique des Régimes {zone}")
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Price"], mode="lines", line=dict(color="black", width=1), showlegend=False))
        return fig
    st.plotly_chart(plot_regime_hist("US"), use_container_width=True)
    st.plotly_chart(plot_regime_hist("EU"), use_container_width=True)

with tab2:
    st.header("Analyse des Signaux Hystérésis (±1%)")
    z = st.radio("Zone", ["US", "EU"], horizontal=True)
    st.subheader("Signal Croissance")
    p, m = sig[f"Price_{z}"], sig[f"MA200_{z}"]
    fig_g = go.Figure()
    fig_g.add_trace(go.Scatter(x=p.index, y=p, name="Prix"))
    fig_g.add_trace(go.Scatter(x=m.index, y=m, name="MM200"))
    fig_g.add_trace(go.Scatter(x=m.index, y=m*1.01, name="+1%", line=dict(dash='dash', color='green')))
    fig_g.add_trace(go.Scatter(x=m.index, y=m*0.99, name="-1%", line=dict(dash='dash', color='red')))
    st.plotly_chart(fig_g, use_container_width=True)
    st.subheader("Signal Inflation")
    r, md = sig[f"Ratio_{z}"], sig[f"Median_{z}"]
    fig_i = go.Figure()
    fig_i.add_trace(go.Scatter(x=r.index, y=r, name="Ratio Beta"))
    fig_i.add_trace(go.Scatter(x=md.index, y=md, name="Médiane 200j"))
    fig_i.add_trace(go.Scatter(x=md.index, y=md*1.01, name="+1%", line=dict(dash='dash', color='green')))
    fig_i.add_trace(go.Scatter(x=md.index, y=md*0.99, name="-1%", line=dict(dash='dash', color='red')))
    st.plotly_chart(fig_i, use_container_width=True)

with tab3:
    st.header("Analyse des Performances")
    st.subheader("Statistiques des Poches")
    c1, c2 = st.columns(2)
    for i, z in enumerate(["US", "EU"]):
        with [c1, c2][i]:
            s = stats[stats["Pocket"] == z].iloc[0]
            st.metric(f"CAGR {z}", f"{s['CAGR']*100:.2f}%")
            st.metric(f"Max Drawdown {z}", f"{s['Max_Drawdown']*100:.2f}%")
            st.metric(f"Sharpe {z}", f"{s['Sharpe']:.2f}")
            st.metric(f"Hit Rate {z}", f"{s['Hit_Rate']*100:.2f}%")

    st.subheader("Performance Cumulative")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt.index, y=bt["Value_US"], name="Poche US"))
    fig.add_trace(go.Scatter(x=bt.index, y=bt["Value_EU"], name="Poche EU"))
    fig.add_trace(go.Scatter(x=bt.index, y=bt["Benchmark_Value"], name="Benchmark (S&P 500)", line=dict(dash='dot')))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Heatmap des Secteurs par Régime")
    z_h = st.radio("Secteurs Zone", ["US", "EU"], horizontal=True, key="heat")
    p_f = perf[perf["Zone"] == z_h].pivot_table(index="Sector", columns="Regime", values="Avg_Annual_Return") * 100
    st.plotly_chart(px.imshow(p_f, text_auto=".2f", color_continuous_scale="RdYlGn", title=f"Rendement (%) - {z_h}"), use_container_width=True)

with tab4:
    st.header("Méthodologie Miroir")
    st.markdown("""
    ### 1. Architecture des Poches Indépendantes
    - **Poche A (US)** : 10 % Or (GLD) + 90 % Macro US.
    - **Poche B (EU)** : 10 % Or (GLD) + 90 % Macro EU.
    Chaque poche gère son propre risque et ses propres signaux.

    ### 2. Filtres d'Hystérésis (±1,0 %)
    - **Croissance** : Indice vs MM200.
    - **Inflation** : Ratio (Beta Positif / Beta Négatif) vs sa Médiane 200j.

    ### 3. Composition Beta (Pondérations exactes)
    - **Beta Positif** : Energy (35%), Finance (25%), Materials (20%), Commodities (20%).
    - **Beta Négatif** : Tech (40%), Cons. Disc (30%), Utilities (15%), RE (15%).

    ### 4. Allocation Macro (Cap 33,3 %)
    Dans les 90 % de chaque poche, 3 secteurs max à 33,3 % chacun (soit 30 % du capital total de la poche). Le reste est en Cash.
    Frais de transaction de **0,10 %** appliqués sur tout changement de poids.
    """)
