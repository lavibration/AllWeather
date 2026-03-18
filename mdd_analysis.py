import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os

# Configuration (same as 1_data_engine.py for consistency)
TICKERS_EU = {
    "Index": "^STOXX50E",
    "Gold": "GLD",
    "Beta_Pos": {"EXV5.DE": 0.35, "EXV1.DE": 0.25, "EXV6.DE": 0.20, "SXRS.DE": 0.20},
    "Beta_Neg": {"EXV3.DE": 0.40, "EXV8.DE": 0.30, "EXV9.DE": 0.15, "EXI5.DE": 0.15},
    "Regimes_Optimized": {
        "GOLDILOCKS": {"EXV1.DE": 0.333, "EXV9.DE": 0.333, "EXV8.DE": 0.286, "CASH": 0.048},
        "REFLATION": {"EXV1.DE": 0.333, "EXV6.DE": 0.333, "EXW1.DE": 0.056, "CASH": 0.277},
        "STAGFLATION": {"EXI5.DE": 0.333, "EXV8.DE": 0.333, "EXV3.DE": 0.318, "CASH": 0.015},
        "DEFLATION": {"EXV9.DE": 0.333, "EXV3.DE": 0.333, "EXV1.DE": 0.324, "CASH": 0.010}
    },
    "Regimes_Historical": {
        "GOLDILOCKS": {"EXV4.DE": 0.333, "EXV3.DE": 0.333, "EXI5.DE": 0.333},
        "REFLATION": {"EXV1.DE": 0.333, "EXV6.DE": 0.333, "SXRS.DE": 0.333},
        "STAGFLATION": {"SXRS.DE": 0.333, "CASH": 0.667},
        "DEFLATION": {"EXV4.DE": 0.333, "EXV2.DE": 0.333, "EXV7.DE": 0.333}
    }
}

def get_data(tickers, start):
    data = yf.download(tickers, start=start)['Close']
    if isinstance(data.columns, pd.MultiIndex):
        data = data.stack(level=0) # Or handle properly
    return data

def calculate_signals_custom(data, index_ticker, h_val=0.01):
    price = data[index_ticker]
    ma200 = price.rolling(200).mean()
    status_g = pd.Series("UNKNOWN", index=data.index)
    curr = None
    for i in range(len(data)):
        p, m = price.iloc[i], ma200.iloc[i]
        if not pd.isna(m):
            if p > (1 + h_val) * m: curr = "UP"
            elif p < (1 - h_val) * m: curr = "DOWN"
            status_g.iloc[i] = curr if curr else "UNKNOWN"
    return status_g

def backtest_one_off(data, growth_signals, regime_map, fee_rate=0.0010):
    # Simplified backtest for comparison
    rets = data.pct_change().fillna(0)
    rets["CASH"] = 0.0
    val = 1000.0
    history = []
    cur_w = {}

    # We assume inflation signal is constant for this specific crash analysis
    # to simplify or we use the pre-calculated ones if available.
    # Let's use the one from signals_analysis.csv if possible.
    sig_file = "signals_analysis.csv"
    if os.path.exists(sig_file):
        sig_df = pd.read_csv(sig_file, index_col=0, parse_dates=True)
        inf_sig = sig_df["Inflation_EU"]
    else:
        inf_sig = pd.Series("DOWN", index=data.index) # Default

    for i in range(len(data)):
        date = data.index[i]
        g = growth_signals.shift(1).get(date, "UNKNOWN")
        inf = inf_sig.shift(1).get(date, "UNKNOWN")

        reg = "UNKNOWN"
        if g == "UP" and inf == "DOWN": reg = "GOLDILOCKS"
        elif g == "UP" and inf == "UP": reg = "REFLATION"
        elif g == "DOWN" and inf == "UP": reg = "STAGFLATION"
        elif g == "DOWN" and inf == "DOWN": reg = "DEFLATION"

        tar_w = {"GLD": 0.10}
        if reg != "UNKNOWN":
            for tk, wt in regime_map[reg].items():
                tar_w[tk] = tar_w.get(tk, 0) + wt * 0.90
        else: tar_w["CASH"] = 0.90

        tw = sum(tar_w.values())
        if abs(1.0 - tw) > 1e-6: tar_w["CASH"] = tar_w.get("CASH", 0) + (1.0 - tw)

        to = sum(abs(tar_w.get(tk, 0) - cur_w.get(tk, 0)) for tk in set(list(cur_w.keys()) + list(tar_w.keys())))
        fee = to * val * fee_rate
        val -= fee
        r = sum(wt * rets.loc[date].get(tk, 0) for tk, wt in tar_w.items())
        val *= (1 + r)
        cur_w = tar_w.copy()
        history.append(val)
    return pd.Series(history, index=data.index)

def run_decomposition():
    res = pd.read_csv("backtest_results.csv", index_col=0, parse_dates=True)
    sig = pd.read_csv("signals_analysis.csv", index_col=0, parse_dates=True)

    v = res['Value_EU']
    dd = (v - v.cummax()) / v.cummax()
    mdd = dd.min()
    trough_date = dd.idxmin()
    peak_date = v[:trough_date].idxmax()

    # Recovery
    rec_val = v[trough_date:][v[trough_date:] >= v[peak_date]]
    recovery_date = rec_val.index[0] if not rec_val.empty else None

    # 1. WINDOW
    print(f"--- 1. IDENTIFICATION DE LA FENÊTRE CRITIQUE ---")
    print(f"Date du Sommet (Peak): {peak_date.date()}")
    print(f"Date du Point Bas (Trough): {trough_date.date()}")
    print(f"Date de récupération: {recovery_date.date() if recovery_date else 'En cours'}")
    print(f"Durée Peak-to-Trough: {(trough_date - peak_date).days} jours")
    print(f"Durée Underwater: {(recovery_date - peak_date).days if recovery_date else (v.index[-1] - peak_date).days} jours")
    print(f"Max Drawdown: {mdd*100:.2f}%")

    # 2. ATTRIBUTION
    print(f"\n--- 2. ATTRIBUTION DE LA PERTE ---")
    # Benchmark Stoxx 50
    stoxx = yf.download("^STOXX50E", start=peak_date, end=trough_date + pd.Timedelta(days=1))['Close']
    if isinstance(stoxx, pd.DataFrame): stoxx = stoxx.iloc[:, 0]
    bench_perf = (stoxx.iloc[-1] / stoxx.iloc[0]) - 1
    print(f"Performance Benchmark (Euro Stoxx 50): {float(bench_perf)*100:.2f}%")

    # Gold
    gld = yf.download("GLD", start=peak_date, end=trough_date + pd.Timedelta(days=1))['Close']
    if isinstance(gld, pd.DataFrame): gld = gld.iloc[:, 0]
    gld_perf = (gld.iloc[-1] / gld.iloc[0]) - 1
    print(f"Performance Poche OR (10%): {float(gld_perf)*100:.2f}%")

    # Sectors & Cash
    period_sig = sig.loc[peak_date:trough_date]
    print(f"Regimes durant la chute: {period_sig['Regime_EU'].unique()}")

    # 3. ANALYSE DU RÉGIME ET DES SIGNAUX
    print(f"\n--- 3. ANALYSE DU RÉGIME ET DES SIGNAUX ---")
    # Growth Signal Lag
    price = sig['Price_EU']
    ma200 = sig['MA200_EU']
    # Trigger point: when price < 0.99 * ma200
    trigger_date = price[(price < 0.99 * ma200) & (price.index >= peak_date)].index[0]
    lag_days = (trigger_date - peak_date).days
    print(f"Déclenchement Signal de Sortie (Hystérésis 1%): {trigger_date.date()} ({lag_days} jours après le peak)")

    # Hysteresis Impact
    all_tickers = ["^STOXX50E", "GLD", "EXV1.DE", "EXV6.DE", "EXV8.DE", "EXV9.DE", "EXV3.DE", "EXV4.DE", "EXV2.DE", "EXV7.DE", "EXI5.DE", "EXW1.DE", "SXRS.DE"]
    full_data = yf.download(all_tickers, start="2019-01-01")['Close']

    for h in [0.0, 0.005, 0.01]:
        growth = calculate_signals_custom(full_data, "^STOXX50E", h)
        bt = backtest_one_off(full_data, growth, TICKERS_EU["Regimes_Optimized"])
        bt_dd = (bt - bt.cummax()) / bt.cummax()
        p_mdd = bt_dd.loc[peak_date:trough_date].min()
        print(f"Impact Hystérésis {h*100:.1f}% -> Drawdown sur période: {p_mdd*100:.2f}%")

    # 4. OPTIMIZED VS HISTORICAL
    print(f"\n--- 4. COMPARAISON OPTIMISÉ VS HISTORIQUE ---")
    growth_fix = calculate_signals_custom(full_data, "^STOXX50E", 0.01)
    bt_opt = backtest_one_off(full_data, growth_fix, TICKERS_EU["Regimes_Optimized"])
    bt_hist = backtest_one_off(full_data, growth_fix, TICKERS_EU["Regimes_Historical"])

    mdd_opt = ((bt_opt - bt_opt.cummax()) / bt_opt.cummax()).loc[peak_date:trough_date].min()
    mdd_hist = ((bt_hist - bt_hist.cummax()) / bt_hist.cummax()).loc[peak_date:trough_date].min()

    print(f"Drawdown 'Optimisé' (Banques/Auto/Ressources): {mdd_opt*100:.2f}%")
    print(f"Drawdown 'Historique' (Santé/Tech/Immo + CASH): {mdd_hist*100:.2f}%")

    # PLOT
    plt.figure(figsize=(12, 6))
    plt.fill_between(dd.index, dd, 0, color='red', alpha=0.3, label="EU Pocket Drawdown")
    plt.axvspan(peak_date, trough_date, color='gray', alpha=0.2, label="MDD Window")
    plt.title(f"Underwater Plot - Europe Pocket (Focus: {peak_date.date()} to {trough_date.date()})")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)
    plt.savefig("underwater_plot_eu.png")

if __name__ == "__main__":
    run_decomposition()
