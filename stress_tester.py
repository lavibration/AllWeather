import yfinance as yf
import pandas as pd
import numpy as np
import re

# --- 1. CONFIGURATION ---
UNIVERSE = {
    "US": {"Benchmark": "^GSPC", "Gold": "GLD"},
    "EU": {"Benchmark": "^STOXX50E", "Gold": "GLD"}
}

HISTORIC_EU_REGIMES = {
    "GOLDILOCKS": {"EXV4.DE": 0.3333, "EXV3.DE": 0.3333, "EXI5.DE": 0.3333},
    "REFLATION": {"EXV1.DE": 0.3333, "EXV6.DE": 0.3333, "SXRS.DE": 0.3333}, # Historic used SXRS.DE for commodities
    "STAGFLATION": {"SXRS.DE": 0.3333, "CASH": 0.6667},
    "DEFLATION": {"EXV4.DE": 0.3333, "EXV2.DE": 0.3333, "EXV7.DE": 0.3333}
}

# Inflation Signal Beta Portfolios (from 1_data_engine.py logic)
BETA_CONFIG = {
    "US": {
        "Pos": {"XLE": 0.35, "XLF": 0.25, "XLB": 0.20, "DBC": 0.20},
        "Neg": {"XLK": 0.40, "XLY": 0.30, "XLU": 0.15, "XLRE": 0.15}
    },
    "EU": {
        "Pos": {"EXV5.DE": 0.35, "EXV1.DE": 0.25, "EXV6.DE": 0.20, "SXRS.DE": 0.20},
        "Neg": {"EXV3.DE": 0.40, "EXV8.DE": 0.30, "EXV9.DE": 0.15, "EXI5.DE": 0.15}
    }
}

def parse_alloc(row_str):
    alloc = {}
    parts = row_str.split(", ")
    for p in parts:
        m = re.match(r"([\^A-Z0-9\.]+)\s\((\d+\.\d+)\%\)", p)
        if m:
            tk, wt = m.groups()
            alloc[tk] = float(wt) / 100.0
    return alloc

def get_data():
    all_tk = set(["^GSPC", "^STOXX50E", "GLD", "SXRS.DE"])
    for z in BETA_CONFIG.values():
        all_tk.update(z["Pos"].keys())
        all_tk.update(z["Neg"].keys())
    # Broad scan tickers
    all_tk.update(["XLK", "XLY", "XLF", "XLE", "XLB", "XLV", "XLI", "XLU", "XLRE", "XLC", "DBC",
                  "EXV3.DE", "EXV8.DE", "EXV1.DE", "EXV5.DE", "EXV6.DE", "EXV4.DE", "EXV2.DE", "EXW1.DE", "EXI5.DE", "EXV9.DE", "EXV7.DE"])
    data = yf.download(list(all_tk), start="2012-01-01", end="2020-01-01")['Close']
    return data.ffill()

def calc_regimes(data, zone):
    conf = UNIVERSE[zone]
    idx = data[conf["Benchmark"]].dropna()
    ma200 = idx.rolling(200).mean()
    g_sig = pd.Series(index=idx.index, dtype='string')
    curr = None
    for i in range(len(idx)):
        p, m = idx.iloc[i], ma200.iloc[i]
        if not pd.isna(m):
            if p > 1.01 * m: curr = "UP"
            elif p < 0.99 * m: curr = "DOWN"
            g_sig.iloc[i] = curr
    rets = data.pct_change()
    pos = sum(rets[tk] * wt for tk, wt in BETA_CONFIG[zone]["Pos"].items() if tk in rets.columns).fillna(0)
    neg = sum(rets[tk] * wt for tk, wt in BETA_CONFIG[zone]["Neg"].items() if tk in rets.columns).fillna(0)
    ratio = (1+pos).cumprod() / (1+neg).cumprod()
    ratio = ratio.loc[idx.index]
    med200 = ratio.rolling(200).median()
    i_sig = pd.Series(index=idx.index, dtype='string')
    curr = None
    for i in range(len(idx)):
        r, m = ratio.iloc[i], med200.iloc[i]
        if not pd.isna(m):
            if r > 1.01 * m: curr = "UP"
            elif r < 0.99 * m: curr = "DOWN"
            i_sig.iloc[i] = curr
    regime = pd.Series(index=idx.index, dtype='string')
    regime[(g_sig == "UP") & (i_sig == "DOWN")] = "GOLDILOCKS"
    regime[(g_sig == "UP") & (i_sig == "UP")] = "REFLATION"
    regime[(g_sig == "DOWN") & (i_sig == "UP")] = "STAGFLATION"
    regime[(g_sig == "DOWN") & (i_sig == "DOWN")] = "DEFLATION"
    return regime

def run_backtest(data, regimes, zone, allocation_map):
    rets = data.pct_change().fillna(0)
    rets["CASH"] = 0.0
    sig_shifted = regimes.shift(1).fillna("UNKNOWN")
    val = 1000.0
    history = []
    for i in range(len(regimes)):
        reg = sig_shifted.iloc[i]
        tar_w = {"GLD": 0.10}
        if reg in allocation_map:
            for tk, wt in allocation_map[reg].items():
                tar_w[tk] = tar_w.get(tk, 0) + wt * 0.90
        else: tar_w["CASH"] = 0.90
        tw = sum(tar_w.values())
        if abs(1.0 - tw) > 1e-6: tar_w["CASH"] = tar_w.get("CASH", 0) + (1.0 - tw)
        r = sum(wt * rets.iloc[i].get(tk, 0) for tk, wt in tar_w.items())
        val *= (1 + r)
        history.append(val)
    v = pd.Series(history, index=regimes.index)
    returns = v.pct_change().fillna(0)
    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    mdd = (v / v.cummax() - 1).min()
    return sharpe, mdd

if __name__ == "__main__":
    matrix = pd.read_csv("master_optimization_matrix.csv")
    new_allocs = {}
    for _, row in matrix.iterrows():
        new_allocs[(row["Zone"], row["Regime"])] = parse_alloc(row["Recommended_Sectors"])

    data = get_data()
    # Stress-test Period 2: 2013-2019
    data_p2 = data[(data.index >= "2013-01-01") & (data.index <= "2019-12-31")]

    results = []
    for zone in ["US", "EU"]:
        regimes = calc_regimes(data, zone).loc["2013-01-01":"2019-12-31"]

        # 1. P4 Optimized Allocation
        zone_new_alloc = {r: new_allocs[(zone, r)] for r in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"] if (zone, r) in new_allocs}
        sharpe_new, mdd_new = run_backtest(data_p2, regimes, zone, zone_new_alloc)

        # 2. Historic Baseline (EU only specified by user, but let's use 1_data_engine.py's US as historic US baseline)
        if zone == "EU":
            hist_alloc = HISTORIC_EU_REGIMES
        else:
            # Historic US (from 1_data_engine.py initial setup)
            hist_alloc = {
                "GOLDILOCKS": {"XLK": 0.3333, "XLY": 0.3333, "XLF": 0.3333},
                "REFLATION": {"XLE": 0.3333, "XLB": 0.3333, "XLF": 0.3333},
                "STAGFLATION": {"DBC": 0.3333, "CASH": 0.6667},
                "DEFLATION": {"XLY": 0.3333, "XLRE": 0.3333, "XLC": 0.3333}
            }
        sharpe_hist, mdd_hist = run_backtest(data_p2, regimes, zone, hist_alloc)

        results.append({
            "Zone": zone,
            "Sharpe_New": sharpe_new, "MDD_New": mdd_new,
            "Sharpe_Hist": sharpe_hist, "MDD_Hist": mdd_hist
        })

    df_res = pd.DataFrame(results)
    print("\n--- STRESS-TEST COMPARISON (2013-2019) ---")
    print(df_res.to_string())
    df_res.to_csv("stress_test_results.csv", index=False)
