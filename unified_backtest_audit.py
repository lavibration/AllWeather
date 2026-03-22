import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. CONFIGURATION
GLD = "GLD"
CASH = "CASH"

US_CONF = {
    "Index": "^GSPC",
    "Beta_Pos": {"XLE": 0.35, "XLF": 0.25, "XLB": 0.20, "DBC": 0.20},
    "Beta_Neg": {"XLK": 0.40, "XLY": 0.30, "XLU": 0.15, "XLRE": 0.15},
    "Alloc": {
        "GOLDILOCKS": {"XLK": 0.45, "XLY": 0.45, "CASH": 0.10},
        "REFLATION": {"XLE": 0.25, "XLI": 0.35, "XLK": 0.30, "CASH": 0.10},
        "STAGFLATION": {"XLU": 0.30, "DBC": 0.30, "XLV": 0.30, "CASH": 0.10},
        "DEFLATION": {"XLK": 0.50, "XLY": 0.40, "CASH": 0.10}
    }
}
EU_CONF = {
    "Index": "^STOXX50E",
    "Beta_Pos": {"EXV5.DE": 0.35, "EXV1.DE": 0.25, "EXV6.DE": 0.20, "SXRS.DE": 0.20},
    "Beta_Neg": {"EXV3.DE": 0.40, "EXV8.DE": 0.30, "EXV9.DE": 0.15, "EXI5.DE": 0.15},
    "Alloc": {
        "GOLDILOCKS": {"EXW1.DE": 0.30, "EXV3.DE": 0.30, "EXV8.DE": 0.30, "CASH": 0.10},
        "REFLATION": {"EXV1.DE": 0.35, "EXV6.DE": 0.35, "EXW1.DE": 0.20, "CASH": 0.10},
        "STAGFLATION": {"SXRS.DE": 0.40, "EXV4.DE": 0.30, "EXV7.DE": 0.20, "CASH": 0.10},
        "DEFLATION": {"EXV4.DE": 0.40, "EXV7.DE": 0.30, "EXW1.DE": 0.20, "CASH": 0.10}
    }
}
PERIODS = {
    "P0 (2000-2004)": ("2000-01-01", "2004-12-31"),
    "P1 (2005-2012)": ("2005-01-01", "2012-12-31"),
    "P2 (2013-2019)": ("2013-01-01", "2019-12-31"),
    "P3 (2020-2022)": ("2020-01-01", "2022-12-31"),
    "P4 (2023-2026)": ("2023-01-01", "2026-12-31")
}

def get_data():
    tickers = {GLD, US_CONF["Index"], EU_CONF["Index"]}
    for c in [US_CONF, EU_CONF]:
        tickers.update(c["Beta_Pos"].keys())
        tickers.update(c["Beta_Neg"].keys())
        for reg in c["Alloc"]:
            tickers.update([t for t in c["Alloc"][reg].keys() if t != CASH])
    # Download from 2000
    data = yf.download(list(tickers), start="2000-01-01")['Close'].ffill()
    return data

def calculate_signals(data, conf):
    # Growth
    idx = conf["Index"]
    price = data[idx]
    ma200 = price.rolling(200).mean()
    status_g = pd.Series("UNKNOWN", index=data.index, dtype="string")
    curr = "UNKNOWN"
    for i in range(len(data)):
        p, m = price.iloc[i], ma200.iloc[i]
        if not pd.isna(m):
            if p > 1.01 * m: curr = "UP"
            elif p < 0.99 * m: curr = "DOWN"
        status_g.iloc[i] = curr
    # Inflation
    rets = data.pct_change()
    pos = sum(rets[tk] * wt for tk, wt in conf["Beta_Pos"].items() if tk in rets.columns).fillna(0)
    neg = sum(rets[tk] * wt for tk, wt in conf["Beta_Neg"].items() if tk in rets.columns).fillna(0)
    ratio = (1+pos).cumprod() / (1+neg).cumprod()
    med200 = ratio.rolling(200).median()
    status_i = pd.Series("UNKNOWN", index=data.index, dtype="string")
    curr = "UNKNOWN"
    for i in range(len(data)):
        r, m = ratio.iloc[i], med200.iloc[i]
        if not pd.isna(m):
            if r > 1.01 * m: curr = "UP"
            elif r < 0.99 * m: curr = "DOWN"
        status_i.iloc[i] = curr
    regime = pd.Series("UNKNOWN", index=data.index, dtype="string")
    regime[(status_g=="UP") & (status_i=="DOWN")] = "GOLDILOCKS"
    regime[(status_g=="UP") & (status_i=="UP")] = "REFLATION"
    regime[(status_g=="DOWN") & (status_i=="UP")] = "STAGFLATION"
    regime[(status_g=="DOWN") & (status_i=="DOWN")] = "DEFLATION"
    return regime

def run_backtest(data, zone_name):
    conf = US_CONF if zone_name == "US" else EU_CONF
    regime = calculate_signals(data, conf)
    reg_shift = regime.shift(1).fillna("UNKNOWN")

    rets = data.pct_change().fillna(0)
    rets[CASH] = 0.0
    reb_dates = data.resample('ME').last().index

    val, current_w = 1.0, {}
    history, allocation_log, trades, changes = [], [], 0, 0
    last_reg = "UNKNOWN"

    for i in range(len(data)):
        date = data.index[i]
        if current_w:
            day_ret = sum(current_w.get(t,0) * rets.loc[date, t] for t in current_w if t in rets.columns)
            val *= (1 + day_ret)

        if date in reb_dates:
            reg = reg_shift.loc[date]
            target_w = {}
            # Handle GLD start in 2004. Before that, stay in CASH for the 10% pocket.
            if GLD in data.columns and not pd.isna(data.loc[date, GLD]):
                target_w[GLD] = 0.10
            else:
                target_w[CASH] = 0.10

            if reg != "UNKNOWN":
                for t, w in conf["Alloc"][reg].items():
                    # If ticker doesn't exist yet, stay in CASH for that specific weight
                    if t in data.columns and not pd.isna(data.loc[date, t]):
                        target_w[t] = target_w.get(t, 0) + w * 0.90
                    else:
                        target_w[CASH] = target_w.get(CASH, 0) + w * 0.90
            else: target_w[CASH] = target_w.get(CASH, 0) + 0.90

            all_t = set(list(target_w.keys()) + list(current_w.keys()))
            turnover = sum(abs(target_w.get(t,0) - current_w.get(t,0)) for t in all_t)
            if turnover > 1e-6: trades += 1; val *= (1 - turnover * 0.0010)
            if reg != last_reg: changes += 1; last_reg = reg
            current_w = target_w.copy()

        history.append(val)
        log_entry = {"Date": date}
        for t, weight in current_w.items(): log_entry[f"W_{t}"] = weight
        allocation_log.append(log_entry)

    return pd.Series(history, index=data.index, name=f"Value_{zone_name}"), regime, trades, changes, pd.DataFrame(allocation_log).set_index("Date").fillna(0)

if __name__ == "__main__":
    data = get_data()
    v_us, r_us, t_us, c_us, w_us = run_backtest(data, "US")
    v_eu, r_eu, t_eu, c_eu, w_eu = run_backtest(data, "EU")

    # 1. Backtest Results (CSV/PNG)
    res = pd.concat([v_us, v_eu], axis=1)
    res["Bench_US"] = (1 + data[US_CONF["Index"]].pct_change().fillna(0)).cumprod()
    res["Bench_EU"] = (1 + data[EU_CONF["Index"]].pct_change().fillna(0)).cumprod()
    res.to_csv("backtest_results.csv")

    plt.figure(figsize=(12, 6))
    for c in res.columns: plt.plot(res[c], label=c)
    plt.yscale('log'); plt.legend(); plt.title("Performance Strategy vs Benchmark (Since 2000)"); plt.savefig("backtest_plot.png")

    # Metrics
    stats = []
    for z, v, b in [("US", v_us, res["Bench_US"]), ("EU", v_eu, res["Bench_EU"])]:
        rets = v.pct_change().fillna(0)
        cagr = (v.iloc[-1] / v.iloc[0])**(252/len(v)) - 1
        vol = rets.std() * np.sqrt(252)
        mdd = ((v - v.cummax()) / v.cummax()).min()
        hit = (rets[rets > 0].count() / rets[rets != 0].count())
        stats.append({"Zone": z, "CAGR": cagr, "Vol": vol, "Sharpe": cagr/vol if vol != 0 else 0, "MaxDD": mdd, "HitRate": hit})
    pd.DataFrame(stats).to_csv("performance_metrics.csv", index=False)

    # 2. Transition Matrix
    trans_us = pd.DataFrame({"From": r_us[r_us != "UNKNOWN"], "To": r_us[r_us != "UNKNOWN"].shift(-1)}).dropna(); trans_eu = pd.DataFrame({"From": r_eu[r_eu != "UNKNOWN"], "To": r_eu[r_eu != "UNKNOWN"].shift(-1)}).dropna(); trans = pd.concat([trans_us, trans_eu])
    matrix = pd.crosstab(trans["From"], trans["To"], normalize='index')
    matrix.to_csv("transition_matrix.csv")
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap="YlGnBu"); plt.title("Regime Transition Matrix"); plt.savefig("transition_heatmap.png")

    # 3. Allocation Stats per Period
    alloc_stats = []
    for p_name, (start, end) in PERIODS.items():
        for z, w in [("US", w_us), ("EU", w_eu)]:
            # Filter rows within the period
            p_w_df = w[(w.index >= start) & (w.index <= end)]
            if not p_w_df.empty:
                p_w = p_w_df.mean()
                alloc_stats.append({"Period": p_name, "Zone": z, **p_w.to_dict()})
    pd.DataFrame(alloc_stats).to_csv("allocation_stats.csv", index=False)

    # 4. Daily Regime Audit
    pd.DataFrame({"US_Regime": r_us, "EU_Regime": r_eu}).to_csv("daily_regime_audit.csv")

    # 5. Trading Audit Stats
    pd.DataFrame([
        {"Zone": "US", "Trades": t_us, "AllocChanges": c_us},
        {"Zone": "EU", "Trades": t_eu, "AllocChanges": c_eu}
    ]).to_csv("trading_audit_stats.csv", index=False)

    print("Analyse complète. CSVs et PNGs générés (2000-2026).")
