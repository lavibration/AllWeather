import yfinance as yf
import pandas as pd
import numpy as np
import os

# --- 1. CONFIGURATION & PARAMETERS ---
GLD_TICKER = "GLD"
US_INDEX = "^GSPC"
EU_INDEX = "^STOXX50E"

# Beta Portfolios Components & Weights (STRICT)
BETA_POS_WEIGHTS = {"Energy": 0.35, "Financials": 0.25, "Materials": 0.20, "Commodities": 0.20}
BETA_NEG_WEIGHTS = {"GrowthTech": 0.40, "ConsDisc": 0.30, "Utilities": 0.15, "GrowthRE": 0.15}

# Ticker Mappings
TICKERS_US = {
    "Energy": "XLE", "Financials": "XLF", "Materials": "XLB", "Commodities": "DBC",
    "GrowthTech": "XLK", "ConsDisc": "XLY", "Utilities": "XLU", "GrowthRE": "XLRE",
    "Comm": "XLC"
}

TICKERS_EU = {
    "Energy": "EXV5.DE", "Financials": "EXV1.DE", "Materials": "EXV6.DE", "Commodities": "DBC",
    "GrowthTech": "EXV3.DE", "ConsDisc": "EXV8.DE", "Utilities": "EXV9.DE", "GrowthRE": "EXI5.DE",
    "Health": "EXV4.DE", "Auto": "EXV2.DE", "Telecoms": "EXV7.DE"
}

# ALLOCATION MATRIX (Macro Pocket: 90%) - STRICT 33.3% RULE
ALLOC_US = {
    "GOLDILOCKS": {"XLK": 0.333, "XLY": 0.333, "XLF": 0.334},
    "REFLATION": {"XLE": 0.333, "XLB": 0.333, "XLF": 0.334},
    "STAGFLATION": {"DBC": 0.333, "CASH": 0.667},
    "DEFLATION": {"XLY": 0.333, "XLRE": 0.333, "XLC": 0.334}
}

ALLOC_EU = {
    "GOLDILOCKS": {"EXV4.DE": 0.333, "EXV3.DE": 0.333, "EXI5.DE": 0.334},
    "REFLATION": {"EXV1.DE": 0.333, "EXV6.DE": 0.333, "DBC": 0.334},
    "STAGFLATION": {"DBC": 0.333, "CASH": 0.667},
    "DEFLATION": {"EXV4.DE": 0.333, "EXV2.DE": 0.333, "EXV7.DE": 0.334}
}

PERIODS = {
    "P1 (2005-2012)": ("2005-01-01", "2012-12-31"),
    "P2 (2013-2019)": ("2013-01-01", "2019-12-31"),
    "P3 (2020-2022)": ("2020-01-01", "2022-12-31"),
    "P4 (2023-Present)": ("2023-01-01", "2026-12-31")
}

def download_data():
    tickers = {GLD_TICKER, US_INDEX, EU_INDEX, "DBC"}
    tickers.update(TICKERS_US.values())
    tickers.update(TICKERS_EU.values())
    data = yf.download(list(tickers), start="2004-01-01")['Close']
    data = data.ffill().bfill()
    return data

def calculate_signals(data, index_ticker, zone):
    # Growth Signal
    price = data[index_ticker]
    ma200 = price.rolling(200).mean()
    growth_signal = pd.Series("UNKNOWN", index=data.index, dtype='string')
    curr = "UNKNOWN"
    for i in range(len(data)):
        p, m = price.iloc[i], ma200.iloc[i]
        if not pd.isna(m):
            if p > 1.01 * m: curr = "UP"
            elif p < 0.99 * m: curr = "DOWN"
        growth_signal.iloc[i] = curr

    # Inflation Signal
    rets = data.pct_change().fillna(0)
    t_map = TICKERS_US if zone == "US" else TICKERS_EU
    pos_ret = sum(rets[t_map[k]] * BETA_POS_WEIGHTS[k] for k in BETA_POS_WEIGHTS if t_map[k] in rets.columns)
    neg_ret = sum(rets[t_map[k]] * BETA_NEG_WEIGHTS[k] for k in BETA_NEG_WEIGHTS if t_map[k] in rets.columns)
    ratio = (1 + pos_ret).cumprod() / (1 + neg_ret).cumprod()
    median200 = ratio.rolling(200).median()
    inflation_signal = pd.Series("UNKNOWN", index=data.index, dtype='string')
    curr = "UNKNOWN"
    for i in range(len(data)):
        r, m = ratio.iloc[i], median200.iloc[i]
        if not pd.isna(m):
            if r > 1.01 * m: curr = "UP"
            elif r < 0.99 * m: curr = "DOWN"
        inflation_signal.iloc[i] = curr

    regime = pd.Series("UNKNOWN", index=data.index, dtype='string')
    regime[(growth_signal == "UP") & (inflation_signal == "DOWN")] = "GOLDILOCKS"
    regime[(growth_signal == "UP") & (inflation_signal == "UP")] = "REFLATION"
    regime[(growth_signal == "DOWN") & (inflation_signal == "UP")] = "STAGFLATION"
    regime[(growth_signal == "DOWN") & (inflation_signal == "DOWN")] = "DEFLATION"

    return {
        "regime": regime,
        "growth": growth_signal,
        "inflation": inflation_signal,
        "ma200": ma200,
        "ratio": ratio,
        "median200": median200,
        "price": price
    }

def run_backtest(data, zone_name, alloc_matrix, regime_series):
    rets = data.pct_change().fillna(0)
    rets['CASH'] = 0.0
    # Apply 1-day lag to signal to avoid look-ahead bias
    regime_delayed = regime_series.shift(1).fillna("UNKNOWN")
    reb_dates = data.resample('ME').last().index
    val, current_w = 100.0, {}
    history = []

    for date in data.index:
        if current_w:
            day_ret = sum(current_w.get(t, 0) * rets.loc[date, t] for t in current_w if t in rets.columns or t == 'CASH')
            val *= (1 + day_ret)

        if date in reb_dates:
            reg = regime_delayed.loc[date]
            target_w = {GLD_TICKER: 0.10}
            if reg in alloc_matrix:
                for t, w in alloc_matrix[reg].items():
                    target_w[t] = target_w.get(t, 0) + w * 0.90
            else:
                target_w['CASH'] = target_w.get('CASH', 0) + 0.90

            # Transaction Fees: 0.10% on turnover (value of modified lines)
            turnover = sum(abs(target_w.get(t, 0) - current_w.get(t, 0)) for t in set(target_w) | set(current_w))
            val *= (1 - turnover * 0.0010)
            current_w = target_w.copy()

        history.append(val)
    return pd.Series(history, index=data.index)

def calculate_stats(strat_val, bench_val):
    def get_metrics(v):
        r = v.pct_change().dropna()
        cagr = (v.iloc[-1]/v.iloc[0])**(252/len(v)) - 1
        vol = r.std() * np.sqrt(252)
        mdd = ((v - v.cummax())/v.cummax()).min()
        sharpe = cagr/vol if vol != 0 else 0
        hit_rate = (r > 0).mean()
        return cagr, vol, mdd, sharpe, hit_rate

    s_cagr, s_vol, s_mdd, s_sharpe, s_hit = get_metrics(strat_val)
    return {
        "CAGR": s_cagr, "Vol": s_vol, "MaxDD": s_mdd, "Sharpe": s_sharpe, "HitRate": s_hit
    }

if __name__ == "__main__":
    data = download_data()
    res_us = calculate_signals(data, US_INDEX, "US")
    res_eu = calculate_signals(data, EU_INDEX, "EU")

    v_us = run_backtest(data, "US", ALLOC_US, res_us["regime"])
    v_eu = run_backtest(data, "EU", ALLOC_EU, res_eu["regime"])

    bench_us = (1 + data[US_INDEX].pct_change().fillna(0)).cumprod() * 100
    bench_eu = (1 + data[EU_INDEX].pct_change().fillna(0)).cumprod() * 100

    # 1. backtest_results.csv
    pd.DataFrame({
        "Strategy_US": v_us, "Benchmark_US": bench_us, "Regime_US": res_us["regime"],
        "Strategy_EU": v_eu, "Benchmark_EU": bench_eu, "Regime_EU": res_eu["regime"]
    }).to_csv("backtest_results.csv")

    # 2. signals_analysis.csv
    pd.DataFrame({
        "Price_US": res_us["price"], "MA200_US": res_us["ma200"], "Ratio_US": res_us["ratio"], "Median200_US": res_us["median200"],
        "Price_EU": res_eu["price"], "MA200_EU": res_eu["ma200"], "Ratio_EU": res_eu["ratio"], "Median200_EU": res_eu["median200"]
    }).to_csv("signals_analysis.csv")

    # 3. sector_performance.csv
    all_rets = data.pct_change().fillna(0)
    perf_records = []
    for zone, reg_series, tickers in [("US", res_us["regime"], TICKERS_US.values()), ("EU", res_eu["regime"], TICKERS_EU.values())]:
        for reg in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]:
            mask = (reg_series == reg)
            if mask.any():
                for t in set(tickers):
                    if t in all_rets.columns:
                        ann_ret = all_rets.loc[mask, t].mean() * 252
                        perf_records.append({"Zone": zone, "Regime": reg, "Sector": t, "Ann_Return": ann_ret})
    pd.DataFrame(perf_records).to_csv("sector_performance.csv", index=False)

    # 4. strategy_stats.csv
    stats_us = calculate_stats(v_us, bench_us)
    stats_eu = calculate_stats(v_eu, bench_eu)
    pd.DataFrame([
        {"Zone": "US", **stats_us},
        {"Zone": "EU", **stats_eu}
    ]).to_csv("strategy_stats.csv", index=False)

    # 5. Sector Performance CSVs (Total + 4 Periods)
    import matplotlib.pyplot as plt
    import seaborn as sns

    all_periods = {"TOTAL": (data.index[0], data.index[-1])}
    for k, v in PERIODS.items():
        all_periods[k.split(" ")[0]] = v # Extract P1, P2, etc.

    for p_key, (start, end) in all_periods.items():
        p_mask = (data.index >= start) & (data.index <= end)
        if not p_mask.any(): continue
        p_perf = []
        for zone, reg_series, tickers in [("US", res_us["regime"], TICKERS_US.values()), ("EU", res_eu["regime"], TICKERS_EU.values())]:
            for reg in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]:
                mask = p_mask & (reg_series == reg)
                if mask.any():
                    for t in set(tickers):
                        if t in all_rets.columns:
                            ann_ret = all_rets.loc[mask, t].mean() * 252
                            p_perf.append({"Zone": zone, "Regime": reg, "Sector": t, "Ann_Return": ann_ret})
        pd.DataFrame(p_perf).to_csv(f"sector_perf_{p_key}.csv", index=False)

    # 6. Global Strategy vs Benchmark Plot
    plt.figure(figsize=(12, 6))
    plt.plot(v_us, label="Stratégie US", color="blue")
    plt.plot(bench_us, label="Benchmark US (S&P500)", color="lightblue", linestyle="--")
    plt.plot(v_eu, label="Stratégie EU", color="green")
    plt.plot(bench_eu, label="Benchmark EU (STX50)", color="lightgreen", linestyle="--")
    plt.yscale('log')
    plt.title("Comparaison Stratégies vs Benchmarks (Échelle Log)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("strategy_vs_benchmark.png")

    print("Data Engine Execution Successful.")
