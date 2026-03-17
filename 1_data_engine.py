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
        "Commodities": "SXRS.DE" # STOXX 600 Resources as proxy
    }
}

TICKER_TO_NAME = {
    "XLK": "Tech US",
    "XLY": "Cons. Disc. US",
    "XLF": "Finance US",
    "XLE": "Energie US",
    "XLB": "Matériaux US",
    "XLRE": "Immobilier US",
    "XLC": "Comm. US",
    "XLU": "Utilities US",
    "DBC": "Commodities US",
    "EXV4.DE": "Santé EU",
    "EXV3.DE": "Tech EU",
    "EXI5.DE": "Immobilier EU",
    "EXV1.DE": "Banques EU",
    "EXV6.DE": "Ressources de Base EU",
    "EXV2.DE": "Automobile EU",
    "EXV7.DE": "Télécoms EU",
    "SXRS.DE": "Commodities EU",
    "GLD": "Or",
    "CASH": "CASH",
    "^GSPC": "S&P 500",
    "^STOXX50E": "Euro Stoxx 50"
}

ALL_TICKERS = list(TICKER_TO_NAME.keys())
if "CASH" in ALL_TICKERS: ALL_TICKERS.remove("CASH")

def download_data():
    print("Downloading data...")
    all_data = {}
    for t in ALL_TICKERS:
        try:
            # We need a bit more data for the 200d rolling median/ma
            d = yf.download(t, start="2019-01-01")
            if not d.empty:
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
    # Ensure indices are present
    df = df.dropna(subset=['^GSPC', '^STOXX50E'])
    return df

def calculate_growth_signals(data):
    signals = pd.DataFrame(index=data.index)
    for zone, ticker in TICKERS["Indices"].items():
        price = data[ticker]
        ma200 = price.rolling(window=200).mean()
        signals[f"Price_{zone}"] = price
        signals[f"MA200_{zone}"] = ma200

        status = pd.Series(index=data.index, dtype='string')
        current_status = None
        for i in range(len(data)):
            p = price.iloc[i]
            m = ma200.iloc[i]
            if pd.isna(m):
                status.iloc[i] = "UNKNOWN"
                continue
            if p > 1.01 * m: current_status = "UP"
            elif p < 0.99 * m: current_status = "DOWN"
            status.iloc[i] = current_status if current_status else "UNKNOWN"
        signals[f"Growth_{zone}"] = status
    return signals

def calculate_inflation_signals(data):
    pos_tickers = ["XLE", "XLF", "XLB", "DBC"]
    pos_weights = [0.35, 0.25, 0.20, 0.20]
    neg_tickers = ["XLK", "XLY", "XLU", "XLRE"]
    neg_weights = [0.40, 0.30, 0.15, 0.15]

    returns = data.pct_change()
    pos_returns = (returns[pos_tickers] * pos_weights).sum(axis=1)
    neg_returns = (returns[neg_tickers] * neg_weights).sum(axis=1)

    pos_price = (1 + pos_returns).cumprod()
    neg_price = (1 + neg_returns).cumprod()

    ratio = pos_price / neg_price
    median_200 = ratio.rolling(window=200).median()

    signals = pd.DataFrame(index=data.index)
    signals["Inflation_Ratio"] = ratio
    signals["Inflation_Median"] = median_200

    status = pd.Series(index=data.index, dtype='string')
    current_status = None
    for i in range(len(data)):
        r = ratio.iloc[i]
        m = median_200.iloc[i]
        if pd.isna(m):
            status.iloc[i] = "UNKNOWN"
            continue
        if r > 1.01 * m: current_status = "UP"
        elif r < 0.99 * m: current_status = "DOWN"
        status.iloc[i] = current_status if current_status else "UNKNOWN"
    signals["Inflation"] = status
    return signals

def determine_regimes(growth_signals, inflation_signals):
    regimes = pd.DataFrame(index=growth_signals.index)
    for zone in ["US", "EU"]:
        g = growth_signals[f"Growth_{zone}"]
        i = inflation_signals["Inflation"]
        regime = pd.Series(index=growth_signals.index, dtype='string')
        mask_goldi = (g == "UP") & (i == "DOWN")
        mask_refl = (g == "UP") & (i == "UP")
        mask_stag = (g == "DOWN") & (i == "UP")
        mask_defl = (g == "DOWN") & (i == "DOWN")
        regime[mask_goldi], regime[mask_refl], regime[mask_stag], regime[mask_defl] = "GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"
        regime[regime.isna()] = "UNKNOWN"
        regimes[f"Regime_{zone}"] = regime
    return regimes

def calculate_zone_allocation(data):
    returns_us = data["^GSPC"].pct_change(252)
    returns_eu = data["^STOXX50E"].pct_change(252)
    weights = pd.DataFrame(index=data.index)
    us_share = pd.Series(0.45, index=data.index)
    for i in range(252, len(data)):
        if returns_us.iloc[i] > returns_eu.iloc[i]: us_share.iloc[i] = 0.55
        else: us_share.iloc[i] = 0.35
    weights["Macro_US"], weights["Macro_EU"], weights["Gold"] = us_share, 0.90 - us_share, 0.10
    return weights

def backtest_strategy(data, regimes, zone_alloc):
    # ALLOC_MATRIX remains the same
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
    returns["CASH"] = 0.0

    # --- NO LOOK-AHEAD BIAS: Shift signals ---
    regimes_shifted = regimes.shift(1).fillna("UNKNOWN")
    zone_alloc_shifted = zone_alloc.shift(1).fillna(0)
    # For first day, set default
    zone_alloc_shifted.iloc[0] = [0.45, 0.45, 0.10]

    val_global, val_us, val_eu = 1000.0, 1000.0, 1000.0
    w_global, w_us, w_eu = {}, {}, {}
    history = []

    for i in range(len(data)):
        date = data.index[i]
        daily_ret = returns.iloc[i]

        # 1. Target Weights
        # Global
        t_global = {"GLD": 0.10}
        reg_us = regimes_shifted.iloc[i]["Regime_US"]
        reg_eu = regimes_shifted.iloc[i]["Regime_EU"]
        a_us = zone_alloc_shifted.iloc[i]["Macro_US"]
        a_eu = zone_alloc_shifted.iloc[i]["Macro_EU"]

        # US Pocket target
        t_us = {}
        if reg_us != "UNKNOWN":
            for tk, wt in ALLOC_MATRIX["US"][reg_us].items():
                t_us[tk] = wt
                t_global[tk] = t_global.get(tk, 0) + wt * a_us
        else:
            t_us["CASH"] = 1.0
            t_global["CASH"] = t_global.get("CASH", 0) + a_us

        # EU Pocket target
        t_eu = {}
        if reg_eu != "UNKNOWN":
            for tk, wt in ALLOC_MATRIX["EU"][reg_eu].items():
                t_eu[tk] = wt
                t_global[tk] = t_global.get(tk, 0) + wt * a_eu
        else:
            t_eu["CASH"] = 1.0
            t_global["CASH"] = t_global.get("CASH", 0) + a_eu

        # Re-normalize to ensure sum=1
        for d, t_dict in zip([val_global, val_us, val_eu], [t_global, t_us, t_eu]):
            tw = sum(t_dict.values())
            if tw < 0.999: t_dict["CASH"] = t_dict.get("CASH", 0) + (1.0 - tw)

        # 2. Fees & Returns
        # We calculate turnover for each to apply fees
        def apply_step(val, cur_w, tar_w):
            to = sum(abs(tar_w.get(tk, 0) - cur_w.get(tk, 0)) for tk in set(list(cur_w.keys()) + list(tar_w.keys())))
            fee = to * val * 0.0010
            val -= fee
            ret = sum(wt * daily_ret.get(tk, 0) for tk, wt in tar_w.items())
            val *= (1 + ret)
            return val, ret, to, fee

        val_global, r_global, to_g, f_g = apply_step(val_global, w_global, t_global)
        val_us, r_us, to_us, f_us = apply_step(val_us, w_us, t_us)
        val_eu, r_eu, to_eu, f_eu = apply_step(val_eu, w_eu, t_eu)

        w_global, w_us, w_eu = t_global.copy(), t_us.copy(), t_eu.copy()

        history.append({
            "Date": date,
            "Value_Global": val_global, "Return_Global": r_global,
            "Value_US": val_us, "Return_US": r_us,
            "Value_EU": val_eu, "Return_EU": r_eu,
            "Regime_US": reg_us, "Regime_EU": reg_eu
        })

    return pd.DataFrame(history).set_index("Date")

def calculate_stats(results, benchmark_data):
    bench_returns = benchmark_data["^GSPC"].pct_change().fillna(0)
    bench_values = (1 + bench_returns).cumprod() * 1000.0
    results["Benchmark_Value"] = bench_values

    stats_list = []
    for port in ["Global", "US", "EU"]:
        rets = results[f"Return_{port}"]
        vals = results[f"Value_{port}"]
        years = len(results) / 252.0
        cagr = (vals.iloc[-1] / 1000.0) ** (1/years) - 1
        vol = rets.std() * np.sqrt(252)
        sharpe = cagr / vol if vol != 0 else 0
        mdd = ((vals - vals.cummax()) / vals.cummax()).min()
        hit = len(rets[rets > 0]) / len(rets[rets != 0])
        stats_list.append({"Portfolio": port, "CAGR": cagr, "Volatility": vol, "Sharpe": sharpe, "Max_Drawdown": mdd, "Hit_Rate": hit})

    return pd.DataFrame(stats_list), results

def export_sector_performance(data, regimes):
    returns = data.pct_change().fillna(0)
    perf_list = []
    for zone in ["US", "EU"]:
        regime_col = f"Regime_{zone}"
        # Filter data to only include dates where we have regime info
        valid_regimes = regimes[regimes[regime_col] != "UNKNOWN"].index
        for regime in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]:
            mask = (regimes[regime_col] == regime)
            if mask.any():
                regime_rets = returns.loc[mask].mean() * 252
                for ticker, ret in regime_rets.items():
                    if ticker in TICKER_TO_NAME:
                        name = TICKER_TO_NAME[ticker]
                        # Only add if it belongs to the zone or is Gold/Indices
                        if (zone == "US" and "US" in name) or (zone == "EU" and "EU" in name) or name in ["Or", "S&P 500", "Euro Stoxx 50"]:
                            perf_list.append({"Zone": zone, "Regime": regime, "Sector": name, "Avg_Annual_Return": ret})
    return pd.DataFrame(perf_list)

if __name__ == "__main__":
    data = download_data()
    growth_signals = calculate_growth_signals(data)
    inflation_signals = calculate_inflation_signals(data)
    regimes = determine_regimes(growth_signals, inflation_signals)
    zone_alloc = calculate_zone_allocation(data)
    backtest_results = backtest_strategy(data, regimes, zone_alloc)
    stats, backtest_results = calculate_stats(backtest_results, data)
    sector_perf = export_sector_performance(data, regimes)
    signals_analysis = pd.concat([growth_signals, inflation_signals], axis=1)

    backtest_results.to_csv("backtest_results.csv")
    signals_analysis.to_csv("signals_analysis.csv")
    sector_perf.to_csv("sector_performance.csv", index=False)
    stats.to_csv("strategy_stats.csv", index=False)
    print("CSVs exported successfully.")
