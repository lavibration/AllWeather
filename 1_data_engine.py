import yfinance as yf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# =============================================================================
# 1. CONFIGURATION & PARAMETERS
# =============================================================================

GLD_TICKER  = "GLD"
US_INDEX    = "^GSPC"
EU_INDEX    = "^STOXX50E"

# --- Common parameters ---
SIGNAL_WINDOW       = 200
BUFFER_ASYMM_WINDOW = 60

# --- Beta Portfolios (inchangés) ---
BETA_POS_WEIGHTS = {"Energy": 0.35, "Financials": 0.25, "Materials": 0.20, "Commodities": 0.20}
BETA_NEG_WEIGHTS = {"GrowthTech": 0.40, "ConsDisc": 0.30, "Utilities": 0.15, "GrowthRE": 0.15}

# --- Ticker Mappings (inchangés) ---
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

# --- Allocation Matrix v2 (inchangée) ---
ALLOC_US = {
    "GOLDILOCKS":  {"XLK": 0.333, "XLY": 0.333, "XLF": 0.334},
    "REFLATION":   {"XLE": 0.400, "DBC": 0.300, "XLB": 0.300},
    "STAGFLATION": {"DBC": 0.400, "CASH": 0.600},
    "DEFLATION":   {"XLY": 0.400, "XLC": 0.300, "CASH": 0.300},
}

ALLOC_EU = {
    "GOLDILOCKS":  {"EXV4.DE": 0.333, "EXV3.DE": 0.333, "EXI5.DE": 0.334},
    "REFLATION":   {"EXV6.DE": 0.400, "EXV1.DE": 0.333, "DBC":     0.267},
    "STAGFLATION": {"DBC": 0.333, "CASH": 0.667},
    "DEFLATION":   {"EXV3.DE": 0.400, "EXV8.DE": 0.333, "EXV7.DE": 0.267},
}

PERIODS = {
    "P1 (2005-2012)":    ("2005-01-01", "2012-12-31"),
    "P2 (2013-2019)":    ("2013-01-01", "2019-12-31"),
    "P3 (2020-2022)":    ("2020-01-01", "2022-12-31"),
    "P4 (2023-Present)": ("2023-01-01", "2026-12-31"),
}

# =============================================================================
# 2. CORE FUNCTIONS
# =============================================================================

def download_data():
    tickers = {GLD_TICKER, US_INDEX, EU_INDEX, "DBC"}
    tickers.update(TICKERS_US.values())
    tickers.update(TICKERS_EU.values())
    data = yf.download(list(tickers), start="2004-01-01")["Close"]
    data = data.ffill().bfill()
    return data

def _apply_persistence_filter(raw_signal: pd.Series, min_days: int) -> pd.Series:
    filtered    = raw_signal.copy()
    current     = raw_signal.iloc[0]
    pending     = None
    pending_cnt = 0

    for i in range(len(raw_signal)):
        r = raw_signal.iloc[i]
        if r == current:
            pending, pending_cnt = None, 0
        else:
            if r == pending:
                pending_cnt += 1
                if pending_cnt >= min_days:
                    current, pending, pending_cnt = pending, None, 0
            else:
                pending, pending_cnt = r, 1
        filtered.iloc[i] = current
    return filtered

def _compute_asymmetric_buffer(series: pd.Series, window: int) -> pd.Series:
    vol    = series.rolling(window).std()
    buffer = (0.5 * vol / series.abs()).fillna(0.01).clip(lower=0.005)
    return buffer

def calculate_signals_parametric(data: pd.DataFrame, index_ticker: str, zone: str,
                                 buffer_mode="fixed", confirm_days=15) -> dict:
    # 1. Growth
    price = data[index_ticker]
    med200_growth = price.rolling(SIGNAL_WINDOW).median()
    if buffer_mode == "fixed":
        buf_growth = pd.Series(0.01, index=data.index)
    else:
        buf_growth = _compute_asymmetric_buffer(price, BUFFER_ASYMM_WINDOW)

    growth_raw = pd.Series("UNKNOWN", index=data.index, dtype="string")
    curr = "UNKNOWN"
    for i in range(len(data)):
        p, m, b = price.iloc[i], med200_growth.iloc[i], buf_growth.iloc[i]
        if not pd.isna(m) and not pd.isna(b):
            if   p > (1 + b) * m: curr = "UP"
            elif p < (1 - b) * m: curr = "DOWN"
        growth_raw.iloc[i] = curr
    growth_signal = _apply_persistence_filter(growth_raw, confirm_days)

    # 2. Inflation
    rets  = data.pct_change().fillna(0)
    t_map = TICKERS_US if zone == "US" else TICKERS_EU
    pos_ret = sum(rets[t_map[k]] * BETA_POS_WEIGHTS[k] for k in BETA_POS_WEIGHTS if t_map[k] in rets.columns)
    neg_ret = sum(rets[t_map[k]] * BETA_NEG_WEIGHTS[k] for k in BETA_NEG_WEIGHTS if t_map[k] in rets.columns)
    ratio = (1 + pos_ret).cumprod() / (1 + neg_ret).cumprod()
    med200_infl = ratio.rolling(SIGNAL_WINDOW).median()
    if buffer_mode == "fixed":
        buf_infl = pd.Series(0.01, index=data.index)
    else:
        buf_infl = _compute_asymmetric_buffer(ratio, BUFFER_ASYMM_WINDOW)

    inflation_raw = pd.Series("UNKNOWN", index=data.index, dtype="string")
    curr = "UNKNOWN"
    for i in range(len(data)):
        r, m, b = ratio.iloc[i], med200_infl.iloc[i], buf_infl.iloc[i]
        if not pd.isna(m) and not pd.isna(b):
            if   r > (1 + b) * m: curr = "UP"
            elif r < (1 - b) * m: curr = "DOWN"
        inflation_raw.iloc[i] = curr
    inflation_signal = _apply_persistence_filter(inflation_raw, confirm_days)

    regime = pd.Series("UNKNOWN", index=data.index, dtype="string")
    regime[(growth_signal == "UP")   & (inflation_signal == "DOWN")] = "GOLDILOCKS"
    regime[(growth_signal == "UP")   & (inflation_signal == "UP")]   = "REFLATION"
    regime[(growth_signal == "DOWN") & (inflation_signal == "UP")]   = "STAGFLATION"
    regime[(growth_signal == "DOWN") & (inflation_signal == "DOWN")] = "DEFLATION"

    return {
        "regime": regime, "growth": growth_signal, "inflation": inflation_signal,
        "med200_growth": med200_growth, "buf_growth": buf_growth,
        "med200_infl": med200_infl, "buf_infl": buf_infl,
        "ratio": ratio, "price": price
    }

def run_backtest(data: pd.DataFrame, zone_name: str, alloc_matrix: dict, regime_series: pd.Series) -> pd.Series:
    rets, rets["CASH"] = data.pct_change().fillna(0), 0.0
    regime_delayed = regime_series.shift(1).fillna("UNKNOWN")
    reb_dates, val, current_w, history = data.resample("ME").last().index, 100.0, {}, []
    for date in data.index:
        if current_w:
            day_ret = sum(current_w.get(t, 0) * rets.loc[date, t] for t in current_w if t in rets.columns or t == "CASH")
            val *= (1 + day_ret)
        if date in reb_dates:
            reg, target_w = regime_delayed.loc[date], {GLD_TICKER: 0.10}
            if reg in alloc_matrix:
                for t, w in alloc_matrix[reg].items(): target_w[t] = target_w.get(t, 0) + w * 0.90
            else: target_w["CASH"] = target_w.get("CASH", 0) + 0.90
            turnover = sum(abs(target_w.get(t, 0) - current_w.get(t, 0)) for t in set(target_w) | set(current_w))
            val *= (1 - turnover * 0.0010)
            current_w = target_w.copy()
        history.append(val)
    return pd.Series(history, index=data.index)

def calculate_stats(strat_val: pd.Series, bench_val: pd.Series) -> dict:
    def get_metrics(v):
        r = v.pct_change().dropna()
        cagr = (v.iloc[-1] / v.iloc[0]) ** (252 / len(v)) - 1
        vol = r.std() * np.sqrt(252)
        mdd = ((v - v.cummax()) / v.cummax()).min()
        sharpe = cagr / vol if vol != 0 else 0
        hit_rate = (r > 0).mean()
        return cagr, vol, mdd, sharpe, hit_rate
    s_cagr, s_vol, s_mdd, s_sharpe, s_hit = get_metrics(strat_val)
    b_cagr, *_ = get_metrics(bench_val)
    return {"CAGR": s_cagr, "Vol": s_vol, "MaxDD": s_mdd, "Sharpe": s_sharpe, "HitRate": s_hit, "Alpha": s_cagr - b_cagr}

def export_sector_performance(data: pd.DataFrame, res_us: dict, res_eu: dict):
    rets = data.pct_change()

    def get_zone_perf(zone_name, t_map, regime_series):
        records = []
        for reg in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]:
            mask = (regime_series == reg)
            if mask.sum() < 20: continue
            for sec, tk in t_map.items():
                if tk in rets.columns:
                    ann_ret = rets.loc[mask, tk].mean() * 252
                    records.append({"Zone": zone_name, "Regime": reg, "Sector": sec, "Ann_Return": ann_ret})
        return records

    # Global
    all_rec = get_zone_perf("US", TICKERS_US, res_us["regime"]) + get_zone_perf("EU", TICKERS_EU, res_eu["regime"])
    pd.DataFrame(all_rec).to_csv("sector_performance.csv", index=False)
    pd.DataFrame(all_rec).to_csv("sector_perf_TOTAL.csv", index=False)

    # By Period
    for p_name, (start, end) in PERIODS.items():
        p_code = p_name.split(" ")[0]
        p_data = data.loc[start:end]
        if p_data.empty: continue
        p_rets = p_data.pct_change()

        p_rec = []
        for zone, t_map, reg_s in [("US", TICKERS_US, res_us["regime"]), ("EU", TICKERS_EU, res_eu["regime"])]:
            for reg in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]:
                mask = (reg_s.loc[start:end] == reg)
                if mask.sum() < 5: continue
                for sec, tk in t_map.items():
                    if tk in p_rets.columns:
                        ann_ret = p_rets.loc[mask, tk].mean() * 252
                        p_rec.append({"Zone": zone, "Regime": reg, "Sector": sec, "Ann_Return": ann_ret})
        pd.DataFrame(p_rec).to_csv(f"sector_perf_{p_code}.csv", index=False)

# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Téléchargement des données...")
    data = download_data()
    bench_us = (1 + data[US_INDEX].pct_change().fillna(0)).cumprod() * 100
    bench_eu = (1 + data[EU_INDEX].pct_change().fillna(0)).cumprod() * 100

    print("Calcul des signaux et du backtest (Robustifiée par zone)...")
    # Zone-specific confirmation delay: US = 10j (réactif), EU = 15j (résistant)
    res_us = calculate_signals_parametric(data, US_INDEX, "US", buffer_mode="asymm", confirm_days=10)
    res_eu = calculate_signals_parametric(data, EU_INDEX, "EU", buffer_mode="asymm", confirm_days=15)

    v_us = run_backtest(data, "US", ALLOC_US, res_us["regime"])
    v_eu = run_backtest(data, "EU", ALLOC_EU, res_eu["regime"])

    # 1. backtest_results.csv
    pd.DataFrame({
        "Strategy_US": v_us, "Benchmark_US": bench_us, "Regime_US": res_us["regime"],
        "Strategy_EU": v_eu, "Benchmark_EU": bench_eu, "Regime_EU": res_eu["regime"]
    }).to_csv("backtest_results.csv")

    # 2. signals_analysis.csv
    pd.DataFrame({
        "Price_US": res_us["price"], "Med200_US": res_us["med200_growth"], "BufGrowth_US": res_us["buf_growth"],
        "Ratio_US": res_us["ratio"], "Med200Ratio_US": res_us["med200_infl"], "BufInfl_US": res_us["buf_infl"],
        "Price_EU": res_eu["price"], "Med200_EU": res_eu["med200_growth"], "BufGrowth_EU": res_eu["buf_growth"],
        "Ratio_EU": res_eu["ratio"], "Med200Ratio_EU": res_eu["med200_infl"], "BufInfl_EU": res_eu["buf_infl"]
    }).to_csv("signals_analysis.csv")

    # 3. strategy_stats.csv
    s_us = calculate_stats(v_us, bench_us)
    s_eu = calculate_stats(v_eu, bench_eu)
    pd.DataFrame([
        {"Zone": "US", **s_us},
        {"Zone": "EU", **s_eu}
    ]).to_csv("strategy_stats.csv", index=False)

    # 4. sector_performance.csv (and period files)
    export_sector_performance(data, res_us, res_eu)

    # Final Visual for Dashboard
    plt.figure(figsize=(12, 6))
    plt.plot(v_us, label="Strategy US (10j delay)", color="blue")
    plt.plot(bench_us, label="Bench US", color="blue", linestyle="--", alpha=0.5)
    plt.plot(v_eu, label="Strategy EU (15j delay)", color="red")
    plt.plot(bench_eu, label="Bench EU", color="red", linestyle="--", alpha=0.5)
    plt.yscale("log")
    plt.title("Performance Stratégie Robustifiée (US 10j / EU 15j)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("strategy_vs_benchmark.png")

    print("\nData Engine — Exécution réussie.")
