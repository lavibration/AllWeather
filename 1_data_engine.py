import yfinance as yf
import pandas as pd
import numpy as np
import os

# --- 1. PARAMETERS & CONFIGURATION ---
TICKERS = {
    "US": {
        "Index": "^GSPC",
        "Gold": "GLD",
        "Beta_Pos": {"XLE": 0.35, "XLF": 0.25, "XLB": 0.20, "DBC": 0.20},
        "Beta_Neg": {"XLK": 0.40, "XLY": 0.30, "XLU": 0.15, "XLRE": 0.15},
        "Regimes": {
            "GOLDILOCKS": {"XLK": 0.333, "XLY": 0.333, "XLF": 0.025, "CASH": 0.309},
            "REFLATION": {"XLE": 0.333, "DBC": 0.333, "XLU": 0.328, "CASH": 0.006},
            "STAGFLATION": {"XLY": 0.333, "XLRE": 0.333, "XLF": 0.314, "CASH": 0.019},
            "DEFLATION": {"XLK": 0.333, "XLC": 0.333, "XLU": 0.204, "CASH": 0.129}
        }
    },
    "EU": {
        "Index": "^STOXX50E",
        "Gold": "GLD",
        "Beta_Pos": {"EXV5.DE": 0.35, "EXV1.DE": 0.25, "EXV6.DE": 0.20, "SXRS.DE": 0.20},
        "Beta_Neg": {"EXV3.DE": 0.40, "EXV8.DE": 0.30, "EXV9.DE": 0.15, "EXI5.DE": 0.15},
        "Regimes": {
            "GOLDILOCKS": {"EXV1.DE": 0.333, "EXV9.DE": 0.333, "EXV8.DE": 0.286, "CASH": 0.048},
            "REFLATION": {"EXV1.DE": 0.333, "EXV6.DE": 0.333, "EXW1.DE": 0.056, "CASH": 0.277},
            "STAGFLATION": {"EXI5.DE": 0.333, "EXV8.DE": 0.333, "EXV3.DE": 0.318, "CASH": 0.015},
            "DEFLATION": {"EXV9.DE": 0.333, "EXV3.DE": 0.333, "EXV1.DE": 0.324, "CASH": 0.010}
        }
    }
}

TICKER_TO_NAME = {
    "XLK": "Tech US", "XLY": "Conso US", "XLF": "Finance US", "XLE": "Energie US", "XLB": "Matériaux US",
    "XLRE": "Immo US", "XLC": "Telecoms US", "XLU": "Utilities US", "XLV": "Santé US", "XLI": "Industrie US",
    "DBC": "Commodities US", "EXV3.DE": "Tech EU", "EXV8.DE": "Conso EU", "EXV1.DE": "Banques EU",
    "EXV5.DE": "Energie EU", "EXV6.DE": "Ressources EU", "EXV4.DE": "Santé EU", "EXV2.DE": "Auto EU",
    "EXW1.DE": "Assurance EU", "EXI5.DE": "Immo EU", "EXV9.DE": "Utilities EU", "EXV7.DE": "Telecoms EU",
    "SXRS.DE": "Commodities EU", "GLD": "Or", "CASH": "CASH", "^GSPC": "S&P 500", "^STOXX50E": "Euro Stoxx 50"
}

ALL_TICKERS = list(TICKER_TO_NAME.keys())
if "CASH" in ALL_TICKERS: ALL_TICKERS.remove("CASH")

def download_data():
    all_data = {}
    for t in ALL_TICKERS:
        try:
            d = yf.download(t, start="2019-01-01")
            if not d.empty:
                if isinstance(d.columns, pd.MultiIndex):
                    if 'Close' in d.columns.get_level_values(0): all_data[t] = d['Close'][t]
                    elif t in d.columns.get_level_values(0): all_data[t] = d.xs(t, axis=1, level=0)['Close']
                else: all_data[t] = d['Close']
        except: pass
    df = pd.DataFrame(all_data).ffill().dropna(subset=['^GSPC', '^STOXX50E'])
    return df

def calculate_signals(data, zone):
    conf = TICKERS[zone]
    signals = pd.DataFrame(index=data.index)
    price = data[conf["Index"]]
    ma200 = price.rolling(200).mean()
    status_g = pd.Series(index=data.index, dtype="string")
    curr = None
    for i in range(len(data)):
        p, m = price.iloc[i], ma200.iloc[i]
        if not pd.isna(m):
            if p > 1.01 * m: curr = "UP"
            elif p < 0.99 * m: curr = "DOWN"
            status_g.iloc[i] = curr
    signals["Price"], signals["MA200"], signals["Growth"] = price, ma200, status_g

    rets = data.pct_change()
    pos = sum(rets[tk] * wt for tk, wt in conf["Beta_Pos"].items() if tk in rets.columns).fillna(0)
    neg = sum(rets[tk] * wt for tk, wt in conf["Beta_Neg"].items() if tk in rets.columns).fillna(0)
    ratio = (1+pos).cumprod() / (1+neg).cumprod()
    ratio = ratio.loc[data.index]
    med200 = ratio.rolling(200).median()
    status_i = pd.Series(index=data.index, dtype="string")
    curr = None
    for i in range(len(data)):
        r, m = ratio.iloc[i], med200.iloc[i]
        if not pd.isna(m):
            if r > 1.01 * m: curr = "UP"
            elif r < 0.99 * m: curr = "DOWN"
            status_i.iloc[i] = curr
    signals["Ratio"], signals["Median"], signals["Inflation"] = ratio, med200, status_i

    regime = pd.Series("UNKNOWN", index=data.index, dtype="string")
    regime[(signals["Growth"]=="UP") & (signals["Inflation"]=="DOWN")] = "GOLDILOCKS"
    regime[(signals["Growth"]=="UP") & (signals["Inflation"]=="UP")] = "REFLATION"
    regime[(signals["Growth"]=="DOWN") & (signals["Inflation"]=="UP")] = "STAGFLATION"
    regime[(signals["Growth"]=="DOWN") & (signals["Inflation"]=="DOWN")] = "DEFLATION"
    signals["Regime"] = regime
    return signals

def backtest_pocket(data, signals, zone):
    rets = data.pct_change().fillna(0)
    rets["CASH"] = 0.0
    sig_shifted = signals["Regime"].shift(1).fillna("UNKNOWN")
    val = 1000.0
    cur_w = {}
    history = []
    for i in range(len(data)):
        date = data.index[i]
        reg = sig_shifted.iloc[i]

        # Determine target allocation (10% Gold rebalanced monthly, 90% Macro)
        is_rebalance_day = (i == 0) or (date.month != data.index[i-1].month)

        tar_w = {"GLD": 0.10}
        if reg != "UNKNOWN":
            for tk, wt in TICKERS[zone]["Regimes"][reg].items():
                tar_w[tk] = tar_w.get(tk, 0) + wt * 0.90
        else: tar_w["CASH"] = 0.90

        tw = sum(tar_w.values())
        if abs(1.0 - tw) > 1e-6: tar_w["CASH"] = tar_w.get("CASH", 0) + (1.0 - tw)

        # Monthly rebalancing logic for the entire portfolio (including Gold)
        if is_rebalance_day:
            to = sum(abs(tar_w.get(tk, 0) - cur_w.get(tk, 0)) for tk in set(list(cur_w.keys()) + list(tar_w.keys())))
            fee = to * val * 0.0010
            val -= fee
            current_pocket_weights = tar_w.copy()
            # Apply daily return after rebalancing
            r_day = sum(wt * rets.iloc[i].get(tk, 0) for tk, wt in current_pocket_weights.items())
            val *= (1 + r_day)
            # Update weights after market move
            current_pocket_weights = {tk: (wt * (1 + rets.iloc[i].get(tk, 0))) / (1 + r_day) for tk, wt in current_pocket_weights.items()}
        else:
            # Intra-month: Let weights drift
            day_rets = np.array([current_pocket_weights.get(tk, 0) * (1 + rets.iloc[i].get(tk, 0)) for tk in current_pocket_weights])
            total_ret = day_rets.sum()
            current_pocket_weights = {tk: (current_pocket_weights.get(tk, 0) * (1 + rets.iloc[i].get(tk, 0))) / total_ret for tk in current_pocket_weights}
            val *= total_ret
            # No fees intra-month since no trade (unless regime change)
            # Re-check for regime changes intra-month
            if reg != sig_shifted.iloc[i-1 if i > 0 else 0]:
                to = sum(abs(tar_w.get(tk, 0) - current_pocket_weights.get(tk, 0)) for tk in set(list(current_pocket_weights.keys()) + list(tar_w.keys())))
                fee = to * val * 0.0010
                val -= fee
                current_pocket_weights = tar_w.copy()

        cur_w = current_pocket_weights.copy()
        r = (val / history[-1]["Value"] - 1) if history else 0
        history.append({"Date": date, f"Value_{zone}": val, f"Return_{zone}": r, f"Regime_Backtest_{zone}": reg, "Value": val})
    return pd.DataFrame(history).set_index("Date").drop(columns=["Value"])

if __name__ == "__main__":
    data = download_data()
    sig_us = calculate_signals(data, "US")
    sig_eu = calculate_signals(data, "EU")
    bt_us = backtest_pocket(data, sig_us, "US")
    bt_eu = backtest_pocket(data, sig_eu, "EU")
    results = pd.concat([bt_us, bt_eu], axis=1)
    results["Benchmark_Value"] = (1 + data["^GSPC"].pct_change().fillna(0)).cumprod() * 1000.0

    stats_list = []
    for z in ["US", "EU"]:
        v, r = results[f"Value_{z}"], results[f"Return_{z}"]
        years = len(results)/252.0
        cagr = (v.iloc[-1]/1000)**(1/years)-1
        vol = r.std()*np.sqrt(252)
        hit_rate = (r[r > 0].count() / r[r != 0].count()) if r[r != 0].count() > 0 else 0
        stats_list.append({"Pocket": z, "CAGR": cagr, "Volatility": vol, "Sharpe": cagr/vol if vol!=0 else 0, "Max_Drawdown": ((v-v.cummax())/v.cummax()).min(), "Hit_Rate": hit_rate})

    perf_list = []
    for z, sig in zip(["US", "EU"], [sig_us, sig_eu]):
        returns = data.pct_change().fillna(0)
        for reg in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]:
            mask = sig["Regime"] == reg
            if mask.any():
                reg_rets = returns[mask].mean() * 252
                for tk, ret in reg_rets.items():
                    if tk in TICKER_TO_NAME:
                        name = TICKER_TO_NAME[tk]
                        if (z=="US" and "US" in name) or (z=="EU" and "EU" in name) or name in ["Or", "S&P 500", "Euro Stoxx 50"]:
                            perf_list.append({"Zone": z, "Regime": reg, "Sector": name, "Avg_Annual_Return": ret})

    results.to_csv("backtest_results.csv")
    pd.concat([sig_us.add_suffix("_US"), sig_eu.add_suffix("_EU")], axis=1).to_csv("signals_analysis.csv")
    pd.DataFrame(stats_list).to_csv("strategy_stats.csv", index=False)
    pd.DataFrame(perf_list).to_csv("sector_performance.csv", index=False)
    print("Final Production CSVs exported successfully.")
