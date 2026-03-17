import yfinance as yf
import pandas as pd
import numpy as np
import os

# --- 1. PARAMETERS & CONFIGURATION ---
TICKERS = {
    "Indices": {"US": "^GSPC", "EU": "^STOXX50E"},
    "Gold": "GLD",
    "US_Sectors": {
        "Tech": "XLK",
        "Cons_Disc": "XLY",
        "Financials": "XLF",
        "Energy": "XLE",
        "Materials": "XLB",
        "Real_Estate": "XLRE",
        "Comm": "XLC",
        "Utilities": "XLU",
        "Commodities": "DBC"
    },
    "EU_Sectors": {
        "Sante": "EXV4.DE",
        "Tech": "EXV3.DE",
        "Real_Estate": "EXI5.DE",
        "Banks": "EXV1.DE",
        "Basic_Resources": "EXV6.DE",
        "Automobile": "EXV2.DE",
        "Telecoms": "EXV7.DE",
        "Commodities": "SXRS.DE"
    }
}

ALL_TICKERS = ["^GSPC", "^STOXX50E", "GLD",
               "XLK", "XLY", "XLF", "XLE", "XLB", "XLRE", "XLC", "XLU", "DBC",
               "EXV4.DE", "EXV3.DE", "EXI5.DE", "EXV1.DE", "EXV6.DE", "EXV2.DE", "EXV7.DE", "SXRS.DE"]

def download_data():
    print("Downloading data...")
    all_data = {}
    for t in ALL_TICKERS:
        try:
            d = yf.download(t, start="2020-01-01")
            if not d.empty:
                # If d is MultiIndex (newer yfinance), get Close
                if isinstance(d.columns, pd.MultiIndex):
                    if 'Close' in d.columns.get_level_values(0):
                        all_data[t] = d['Close'][t]
                    elif t in d.columns.get_level_values(0):
                         all_data[t] = d.xs(t, axis=1, level=0)['Close']
                else:
                    all_data[t] = d['Close']
        except Exception as e:
            print(f"Failed to download {t}: {e}")

    df = pd.DataFrame(all_data)
    df = df.ffill()
    df = df.dropna(subset=['^GSPC', '^STOXX50E'])
    return df

def calculate_growth_signals(data):
    signals = pd.DataFrame(index=data.index)

    for zone, ticker in TICKERS["Indices"].items():
        price = data[ticker]
        ma200 = price.rolling(window=200).mean()

        signals[f"Price_{zone}"] = price
        signals[f"MA200_{zone}"] = ma200

        # Hysteresis
        # UP: Price > 1.01 * MA200
        # DOWN: Price < 0.99 * MA200
        status = pd.Series(index=data.index, dtype='string')
        current_status = None

        for i in range(len(data)):
            p = price.iloc[i]
            m = ma200.iloc[i]

            if pd.isna(m):
                status.iloc[i] = "UNKNOWN"
                continue

            if p > 1.01 * m:
                current_status = "UP"
            elif p < 0.99 * m:
                current_status = "DOWN"
            # else keep current_status

            status.iloc[i] = current_status if current_status else "UNKNOWN"

        signals[f"Growth_{zone}"] = status

    return signals

def calculate_inflation_signals(data):
    # Beta Portfolios (using US ETFs as requested for the global inflation signal)
    # Bêta Positif : Energy (35%), Financials/Banks (25%), Materials (20%), Commodities (20%)
    # Bêta Négatif : Growth Tech (40%), Cons. Discretionary (30%), Utilities (15%), Growth Real Estate (15%)

    pos_tickers = ["XLE", "XLF", "XLB", "DBC"]
    pos_weights = [0.35, 0.25, 0.20, 0.20]

    neg_tickers = ["XLK", "XLY", "XLU", "XLRE"]
    neg_weights = [0.40, 0.30, 0.15, 0.15]

    # Calculate daily returns for these sectors
    returns = data.pct_change()

    pos_returns = (returns[pos_tickers] * pos_weights).sum(axis=1)
    neg_returns = (returns[neg_tickers] * neg_weights).sum(axis=1)

    # Growth of 1 unit
    pos_price = (1 + pos_returns).cumprod()
    neg_price = (1 + neg_returns).cumprod()

    ratio = pos_price / neg_price
    median_200 = ratio.rolling(window=200).median()

    signals = pd.DataFrame(index=data.index)
    signals["Inflation_Ratio"] = ratio
    signals["Inflation_Median"] = median_200

    # Hysteresis
    # UP: Ratio > 1.01 * Median
    # DOWN: Ratio < 0.99 * Median
    status = pd.Series(index=data.index, dtype='string')
    current_status = None

    for i in range(len(data)):
        r = ratio.iloc[i]
        m = median_200.iloc[i]

        if pd.isna(m):
            status.iloc[i] = "UNKNOWN"
            continue

        if r > 1.01 * m:
            current_status = "UP"
        elif r < 0.99 * m:
            current_status = "DOWN"

        status.iloc[i] = current_status if current_status else "UNKNOWN"

    signals["Inflation"] = status
    return signals

def determine_regimes(growth_signals, inflation_signals):
    regimes = pd.DataFrame(index=growth_signals.index)

    for zone in ["US", "EU"]:
        g = growth_signals[f"Growth_{zone}"]
        i = inflation_signals["Inflation"]

        regime = pd.Series(index=growth_signals.index, dtype='string')

        # GOLDILOCKS (Growth UP / Inflation DOWN)
        # REFLATION (Growth UP / Inflation UP)
        # STAGFLATION (Growth DOWN / Inflation UP)
        # DEFLATION (Growth DOWN / Inflation DOWN)

        mask_goldi = (g == "UP") & (i == "DOWN")
        mask_refl = (g == "UP") & (i == "UP")
        mask_stag = (g == "DOWN") & (i == "UP")
        mask_defl = (g == "DOWN") & (i == "DOWN")

        regime[mask_goldi] = "GOLDILOCKS"
        regime[mask_refl] = "REFLATION"
        regime[mask_stag] = "STAGFLATION"
        regime[mask_defl] = "DEFLATION"
        regime[regime.isna()] = "UNKNOWN"

        regimes[f"Regime_{zone}"] = regime

    return regimes

def calculate_zone_allocation(data):
    # Dynamic split of the 90% Macro Pocket between US and EU
    # We use 1-year (252 days) rolling return as a proxy for relative strength
    # This is a common interpretation of "pilotée dynamiquement" in this context

    returns_us = data["^GSPC"].pct_change(252)
    returns_eu = data["^STOXX50E"].pct_change(252)

    # We use a simple momentum-based weight
    # If one zone is negative and the other positive, it gets more weight
    # Let's use a softmax or a simple proportion of positive returns
    # Actually, a common way is 50/50 baseline adjusted by momentum

    # Simple approach: Rank-based or just normalized positive returns
    # To keep it robust, let's use 60/40 or 40/60 based on which one has higher 1y return

    weights = pd.DataFrame(index=data.index)

    # Default 50/50 split of the 90% pocket (45% each)
    us_share = pd.Series(0.45, index=data.index)

    for i in range(252, len(data)):
        ret_us = returns_us.iloc[i]
        ret_eu = returns_eu.iloc[i]

        if ret_us > ret_eu:
            us_share.iloc[i] = 0.55 # Favor US
        else:
            us_share.iloc[i] = 0.35 # Favor EU

    weights["Macro_US"] = us_share
    weights["Macro_EU"] = 0.90 - us_share
    weights["Gold"] = 0.10

    return weights

def backtest_strategy(data, regimes, zone_alloc):
    # Allocation Matrix
    ALLOC_MATRIX = {
        "US": {
            "GOLDILOCKS": {"XLK": 0.333, "XLY": 0.333, "XLF": 0.333},
            "REFLATION": {"XLE": 0.333, "XLB": 0.333, "XLF": 0.333},
            "STAGFLATION": {"DBC": 0.333, "CASH": 0.667},
            "DEFLATION": {"XLY": 0.333, "XLRE": 0.333, "XLC": 0.333}
        },
        "EU": {
            "GOLDILOCKS": {"EXV4.DE": 0.333, "EXV3.DE": 0.333, "EXI5.DE": 0.333},
            "REFLATION": {"EXV1.DE": 0.333, "EXV6.DE": 0.333, "SXRS.DE": 0.333},
            "STAGFLATION": {"SXRS.DE": 0.333, "CASH": 0.667},
            "DEFLATION": {"EXV4.DE": 0.333, "EXV2.DE": 0.333, "EXV7.DE": 0.333}
        }
    }

    returns = data.pct_change().fillna(0)
    # Add CASH return (0)
    returns["CASH"] = 0.0

    portfolio_value = 1000.0
    values = [portfolio_value]

    current_weights = {} # ticker -> weight

    history = []

    for i in range(len(data)):
        date = data.index[i]
        daily_ret = returns.iloc[i]

        # Calculate Target Weights
        target_weights = {"GLD": 0.10}

        regime_us = regimes.iloc[i]["Regime_US"]
        regime_eu = regimes.iloc[i]["Regime_EU"]

        alloc_us = zone_alloc.iloc[i]["Macro_US"]
        alloc_eu = zone_alloc.iloc[i]["Macro_EU"]

        if regime_us != "UNKNOWN":
            for ticker, weight in ALLOC_MATRIX["US"][regime_us].items():
                target_weights[ticker] = target_weights.get(ticker, 0) + weight * alloc_us
        else:
            target_weights["CASH"] = target_weights.get("CASH", 0) + alloc_us

        if regime_eu != "UNKNOWN":
            for ticker, weight in ALLOC_MATRIX["EU"][regime_eu].items():
                target_weights[ticker] = target_weights.get(ticker, 0) + weight * alloc_eu
        else:
            target_weights["CASH"] = target_weights.get("CASH", 0) + alloc_eu

        # Ensure sum is 1.0 (approx)
        total_w = sum(target_weights.values())
        if total_w < 0.999:
            target_weights["CASH"] = target_weights.get("CASH", 0) + (1.0 - total_w)

        # Transaction Fees (0.10% on turnover)
        turnover = 0
        all_tickers = set(list(current_weights.keys()) + list(target_weights.keys()))
        for t in all_tickers:
            turnover += abs(target_weights.get(t, 0) - current_weights.get(t, 0))

        fee = turnover * portfolio_value * 0.0010
        portfolio_value -= fee

        # Daily Return
        p_ret = 0
        for t, w in target_weights.items():
            p_ret += w * daily_ret.get(t, 0)

        portfolio_value *= (1 + p_ret)
        values.append(portfolio_value)
        current_weights = target_weights.copy()

        history.append({
            "Date": date,
            "Portfolio_Value": portfolio_value,
            "Daily_Return": p_ret,
            "Regime_US": regime_us,
            "Regime_EU": regime_eu,
            "Turnover": turnover,
            "Fee": fee
        })

    results = pd.DataFrame(history).set_index("Date")
    return results

def calculate_stats(results, benchmark_data):
    # Strategy vs Benchmark
    # Using S&P 500 as benchmark
    bench_returns = benchmark_data["^GSPC"].pct_change().fillna(0)
    bench_values = (1 + bench_returns).cumprod() * 1000.0

    results["Benchmark_Value"] = bench_values

    # Stats
    returns = results["Daily_Return"]

    days = len(results)
    years = days / 252.0

    total_return = (results["Portfolio_Value"].iloc[-1] / 1000.0) - 1
    cagr = (1 + total_return) ** (1/years) - 1

    vol = returns.std() * np.sqrt(252)
    sharpe = cagr / vol if vol != 0 else 0

    # MDD
    peak = results["Portfolio_Value"].cummax()
    drawdown = (results["Portfolio_Value"] - peak) / peak
    mdd = drawdown.min()

    # Hit Rate
    hit_rate = len(returns[returns > 0]) / len(returns[returns != 0])

    stats = pd.DataFrame([{
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max_Drawdown": mdd,
        "Hit_Rate": hit_rate
    }])

    return stats, results

def export_sector_performance(data, regimes):
    # Performance of sectors per regime
    returns = data.pct_change().fillna(0)
    perf_list = []

    for zone in ["US", "EU"]:
        regime_col = f"Regime_{zone}"
        for regime in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]:
            mask = regimes[regime_col] == regime
            if mask.any():
                regime_rets = returns[mask].mean() * 252 # Annualized avg return
                for sector, ret in regime_rets.items():
                    perf_list.append({
                        "Zone": zone,
                        "Regime": regime,
                        "Sector": sector,
                        "Avg_Annual_Return": ret
                    })

    return pd.DataFrame(perf_list)

if __name__ == "__main__":
    data = download_data()
    print(f"Data downloaded: {len(data)} rows.")

    growth_signals = calculate_growth_signals(data)
    print("Growth signals calculated.")

    inflation_signals = calculate_inflation_signals(data)
    print("Inflation signals calculated.")

    regimes = determine_regimes(growth_signals, inflation_signals)
    print("Regimes determined.")

    zone_alloc = calculate_zone_allocation(data)
    print("Zone allocation calculated.")

    backtest_results = backtest_strategy(data, regimes, zone_alloc)
    print("Backtest completed.")

    stats, backtest_results = calculate_stats(backtest_results, data)
    sector_perf = export_sector_performance(data, regimes)

    # Prepare signals_analysis.csv
    signals_analysis = pd.concat([growth_signals, inflation_signals], axis=1)

    # Export CSVs
    backtest_results.to_csv("backtest_results.csv")
    signals_analysis.to_csv("signals_analysis.csv")
    sector_perf.to_csv("sector_performance.csv", index=False)
    stats.to_csv("strategy_stats.csv", index=False)

    print("CSVs exported successfully.")
