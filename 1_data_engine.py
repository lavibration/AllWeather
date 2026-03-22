import yfinance as yf
import pandas as pd
import numpy as np
import os

# --- 1. CONFIGURATION & PARAMETERS ---
GLD_TICKER = "GLD"
US_INDEX = "^GSPC"
EU_INDEX = "^STOXX50E"

# Beta Portfolios Components & Weights
BETA_POS_WEIGHTS = {"Energy": 0.35, "Financials": 0.25, "Materials": 0.20, "Commodities": 0.20}
BETA_NEG_WEIGHTS = {"GrowthTech": 0.40, "ConsDisc": 0.30, "Utilities": 0.15, "GrowthRE": 0.15}

# Ticker Mappings
TICKERS_US = {
    "Energy": "XLE", "Financials": "XLF", "Materials": "XLB", "Commodities": "DBC",
    "GrowthTech": "XLK", "ConsDisc": "XLY", "Utilities": "XLU", "GrowthRE": "XLRE",
    "CommServices": "XLC"
}

TICKERS_EU = {
    "Energy": "EXV5.DE", "Financials": "EXV1.DE", "Materials": "EXV6.DE", "Commodities": "DBC",
    "GrowthTech": "EXV3.DE", "ConsDisc": "EXV8.DE", "Utilities": "EXV9.DE", "GrowthRE": "EXI5.DE",
    "HealthCare": "EXV4.DE", "Automobile": "EXV2.DE", "Telecoms": "EXV7.DE"
}

# Allocation Matrix (Macro Pocket: 90%)
# Max 33.3% per sector in the 90% pocket means 0.333 * 0.9 = 29.97% of total portfolio.
# But the requirement says "Max 33.3% par secteur dans la poche de 90%", so 1/3 of the 90%.
ALLOC_US = {
    "GOLDILOCKS": {"XLK": 0.333, "XLY": 0.333, "XLF": 0.333},
    "REFLATION": {"XLE": 0.333, "XLB": 0.333, "XLF": 0.333},
    "STAGFLATION": {"DBC": 0.333, "CASH": 0.667},
    "DEFLATION": {"XLY": 0.333, "XLRE": 0.333, "XLC": 0.333}
}

ALLOC_EU = {
    "GOLDILOCKS": {"EXV4.DE": 0.333, "EXV3.DE": 0.333, "EXI5.DE": 0.333},
    "REFLATION": {"EXV1.DE": 0.333, "EXV6.DE": 0.333, "DBC": 0.333},
    "STAGFLATION": {"DBC": 0.333, "CASH": 0.667},
    "DEFLATION": {"EXV4.DE": 0.333, "EXV2.DE": 0.333, "EXV7.DE": 0.333}
}

def get_all_tickers():
    t = {GLD_TICKER, US_INDEX, EU_INDEX}
    t.update(TICKERS_US.values())
    t.update(TICKERS_EU.values())
    return sorted(list(t))

def download_data():
    tickers = get_all_tickers()
    data = yf.download(tickers, start="2000-01-01")['Close']
    # Fill missing values: ffill then bfill for start of history
    data = data.ffill().bfill()
    return data

def calculate_signals(data, index_ticker, beta_pos_map, beta_neg_map):
    # Growth Signal
    price = data[index_ticker]
    ma200 = price.rolling(200).mean()

    growth_signal = pd.Series(index=data.index, dtype='string')
    current_growth = "UNKNOWN"
    for i in range(len(data)):
        p = price.iloc[i]
        m = ma200.iloc[i]
        if pd.isna(m):
            growth_signal.iloc[i] = "UNKNOWN"
            continue
        if p > 1.01 * m:
            current_growth = "UP"
        elif p < 0.99 * m:
            current_growth = "DOWN"
        growth_signal.iloc[i] = current_growth

    # Inflation Signal
    rets = data.pct_change().fillna(0)

    # Synthetic Beta Portfolios
    pos_ret = sum(rets[beta_pos_map[k]] * BETA_POS_WEIGHTS[k] for k in BETA_POS_WEIGHTS)
    neg_ret = sum(rets[beta_neg_map[k]] * BETA_NEG_WEIGHTS[k] for k in BETA_NEG_WEIGHTS)

    ratio = (1 + pos_ret).cumprod() / (1 + neg_ret).cumprod()
    median200 = ratio.rolling(200).median()

    inflation_signal = pd.Series(index=data.index, dtype='string')
    current_inflation = "UNKNOWN"
    for i in range(len(data)):
        r = ratio.iloc[i]
        m = median200.iloc[i]
        if pd.isna(m):
            inflation_signal.iloc[i] = "UNKNOWN"
            continue
        if r > 1.01 * m:
            current_inflation = "UP"
        elif r < 0.99 * m:
            current_inflation = "DOWN"
        inflation_signal.iloc[i] = current_inflation

    regime = pd.Series(index=data.index, dtype='string')
    regime[(growth_signal == "UP") & (inflation_signal == "DOWN")] = "GOLDILOCKS"
    regime[(growth_signal == "UP") & (inflation_signal == "UP")] = "REFLATION"
    regime[(growth_signal == "DOWN") & (inflation_signal == "UP")] = "STAGFLATION"
    regime[(growth_signal == "DOWN") & (inflation_signal == "DOWN")] = "DEFLATION"
    regime = regime.fillna("UNKNOWN")

    return regime, growth_signal, inflation_signal, ma200, ratio, median200

def run_backtest(data, zone_name, alloc_matrix, regime_series):
    # 10% GOLD, 90% MACRO
    rets = data.pct_change().fillna(0)
    rets['CASH'] = 0.0

    # Shift regime by 1 day to avoid look-ahead bias
    regime_delayed = regime_series.shift(1).fillna("UNKNOWN")

    # Monthly Rebalancing
    reb_dates = data.resample('ME').last().index

    portfolio_value = 100.0
    current_weights = {} # Ticker -> weight
    history = []

    for date in data.index:
        # Update portfolio value with daily returns
        if current_weights:
            day_return = sum(current_weights.get(t, 0) * rets.loc[date, t] for t in current_weights)
            portfolio_value *= (1 + day_return)

        # Rebalance
        if date in reb_dates:
            reg = regime_delayed.loc[date]
            target_weights = {GLD_TICKER: 0.10}

            if reg != "UNKNOWN":
                macro_alloc = alloc_matrix[reg]
                for ticker, weight in macro_alloc.items():
                    target_weights[ticker] = target_weights.get(ticker, 0) + weight * 0.90
            else:
                target_weights['CASH'] = target_weights.get('CASH', 0) + 0.90

            # Transaction fees: 0.10% on turnover
            all_tickers = set(current_weights.keys()) | set(target_weights.keys())
            turnover = sum(abs(target_weights.get(t, 0) - current_weights.get(t, 0)) for t in all_tickers)
            portfolio_value *= (1 - turnover * 0.0010)

            current_weights = target_weights.copy()

        history.append(portfolio_value)

    return pd.Series(history, index=data.index)

def calculate_stats(val_series, bench_series):
    rets = val_series.pct_change().dropna()
    cagr = (val_series.iloc[-1] / val_series.iloc[0]) ** (252 / len(val_series)) - 1
    vol = rets.std() * np.sqrt(252)
    sharpe = cagr / vol if vol != 0 else 0
    mdd = ((val_series - val_series.cummax()) / val_series.cummax()).min()
    hit_rate = len(rets[rets > 0]) / len(rets) if len(rets) > 0 else 0

    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": mdd, "HitRate": hit_rate}

if __name__ == "__main__":
    print("Downloading data...")
    data = download_data()

    print("Calculating signals...")
    reg_us, growth_us, infl_us, ma200_us, ratio_us, med200_us = calculate_signals(data, US_INDEX, TICKERS_US, TICKERS_US)
    reg_eu, growth_eu, infl_eu, ma200_eu, ratio_eu, med200_eu = calculate_signals(data, EU_INDEX, TICKERS_EU, TICKERS_EU)

    print("Running backtests...")
    val_us = run_backtest(data, "US", ALLOC_US, reg_us)
    val_eu = run_backtest(data, "EU", ALLOC_EU, reg_eu)

    # Benchmarks (Base 100)
    bench_us = (1 + data[US_INDEX].pct_change().fillna(0)).cumprod() * 100
    bench_eu = (1 + data[EU_INDEX].pct_change().fillna(0)).cumprod() * 100

    # 1. backtest_results.csv
    backtest_results = pd.DataFrame({
        "Strategy_US": val_us,
        "Benchmark_US": bench_us,
        "Strategy_EU": val_eu,
        "Benchmark_EU": bench_eu,
        "Regime_US": reg_us,
        "Regime_EU": reg_eu
    })
    backtest_results.to_csv("backtest_results.csv")

    # 2. signals_analysis.csv
    signals_analysis = pd.DataFrame({
        "Price_US": data[US_INDEX],
        "MA200_US": ma200_us,
        "Ratio_US": ratio_us,
        "Median200_US": med200_us,
        "Price_EU": data[EU_INDEX],
        "MA200_EU": ma200_eu,
        "Ratio_EU": ratio_eu,
        "Median200_EU": med200_eu
    })
    signals_analysis.to_csv("signals_analysis.csv")

    # 3. sector_performance.csv
    # Matrix of performance of each sector by regime
    all_rets = data.pct_change().fillna(0)
    sector_perf = []
    for reg in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]:
        # US
        mask_us = (reg_us == reg)
        if mask_us.any():
            for s_ticker in set(TICKERS_US.values()):
                avg_ret = all_rets.loc[mask_us, s_ticker].mean() * 252
                sector_perf.append({"Regime": reg, "Zone": "US", "Sector": s_ticker, "Ann_Return": avg_ret})
        # EU
        mask_eu = (reg_eu == reg)
        if mask_eu.any():
            for s_ticker in set(TICKERS_EU.values()):
                avg_ret = all_rets.loc[mask_eu, s_ticker].mean() * 252
                sector_perf.append({"Regime": reg, "Zone": "EU", "Sector": s_ticker, "Ann_Return": avg_ret})
    pd.DataFrame(sector_perf).to_csv("sector_performance.csv", index=False)

    # 4. strategy_stats.csv
    stats_us = calculate_stats(val_us, bench_us)
    stats_eu = calculate_stats(val_eu, bench_eu)
    strategy_stats = pd.DataFrame([
        {"Zone": "US", **stats_us},
        {"Zone": "EU", **stats_eu}
    ])
    strategy_stats.to_csv("strategy_stats.csv", index=False)

    print("Engine completed successfully.")
