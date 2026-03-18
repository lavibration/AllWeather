import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. UNIVERSES & CONFIGURATION ---
UNIVERSE = {
    "US": {
        "Benchmark": "^GSPC",
        "Sectors": {
            "XLK": "Tech", "XLY": "Conso", "XLF": "Finance", "XLE": "Énergie",
            "XLB": "Matériaux", "XLV": "Santé", "XLI": "Industrie", "XLU": "Utilities",
            "XLRE": "Immo", "XLC": "Telecoms", "DBC": "Commodities"
        },
        "Beta_Pos": {"XLE": 0.35, "XLF": 0.25, "XLB": 0.20, "DBC": 0.20},
        "Beta_Neg": {"XLK": 0.40, "XLY": 0.30, "XLU": 0.15, "XLRE": 0.15}
    },
    "EU": {
        "Benchmark": "^STOXX50E",
        "Sectors": {
            "EXV3.DE": "Tech", "EXV8.DE": "Conso", "EXV1.DE": "Banques", "EXV5.DE": "Énergie",
            "EXV6.DE": "Ressources", "EXV4.DE": "Santé", "EXV2.DE": "Auto", "EXW1.DE": "Assurance",
            "EXI5.DE": "Immo", "EXV9.DE": "Utilities", "EXV7.DE": "Telecoms"
        },
        "Beta_Pos": {"EXV5.DE": 0.35, "EXV1.DE": 0.25, "EXV6.DE": 0.20, "SXRS.DE": 0.20},
        "Beta_Neg": {"EXV3.DE": 0.40, "EXV8.DE": 0.30, "EXV9.DE": 0.15, "EXI5.DE": 0.15}
    }
}

PERIODS = {
    "P1": ("2005-01-01", "2012-12-31"),
    "P2": ("2013-01-01", "2019-12-31"),
    "P3": ("2020-01-01", "2022-12-31"),
    "P4": ("2023-01-01", "2026-12-31")
}

def get_full_data():
    all_tk = set()
    for z in UNIVERSE.values():
        all_tk.add(z["Benchmark"])
        all_tk.update(z["Sectors"].keys())
        all_tk.update(z["Beta_Pos"].keys())
        all_tk.update(z["Beta_Neg"].keys())
    if "SXRS.DE" in all_tk: pass # Ensure we have commodity proxy for EU too if used
    data = yf.download(list(all_tk), start="2005-01-01")['Close']
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
    pos_rets = sum(rets[tk] * wt for tk, wt in conf["Beta_Pos"].items() if tk in rets.columns).fillna(0)
    neg_rets = sum(rets[tk] * wt for tk, wt in conf["Beta_Neg"].items() if tk in rets.columns).fillna(0)
    ratio = (1+pos_rets).cumprod() / (1+neg_rets).cumprod()
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

def analyze_sector_performance(data, regime_series, zone):
    conf = UNIVERSE[zone]
    idx_tk = conf["Benchmark"]
    results = []

    rets = data.pct_change().loc[regime_series.index]

    for p_name, (start, end) in PERIODS.items():
        p_mask = (regime_series.index >= start) & (regime_series.index <= end)
        p_regimes = regime_series[p_mask]
        p_rets = rets[p_mask]

        if p_regimes.empty: continue

        for reg in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]:
            reg_mask = p_regimes == reg
            if not reg_mask.any(): continue

            r_rets = p_rets[reg_mask]
            idx_bench = r_rets[idx_tk].mean()

            for s_tk in conf["Sectors"].keys():
                if s_tk not in r_rets.columns: continue
                s_series = r_rets[s_tk].dropna()
                if s_series.empty: continue

                alpha = (s_series.mean() - idx_bench) * 252
                vol = s_series.std() * np.sqrt(252)
                sharpe = alpha / vol if vol > 0 else 0

                results.append({
                    "Zone": zone, "Period": p_name, "Regime": reg, "Ticker": s_tk,
                    "Sector": conf["Sectors"][s_tk], "Alpha": alpha, "Sharpe": sharpe
                })
    return pd.DataFrame(results)

def optimize_selection(stats):
    optimized = []
    for zone in ["US", "EU"]:
        for reg in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]:
            subset = stats[(stats["Zone"] == zone) & (stats["Regime"] == reg)]
            if subset.empty: continue

            # Criteria: Max 2 negative alpha periods across P1-P4
            constancy = subset.groupby("Ticker")["Alpha"].apply(lambda x: (x > 0).sum())
            valid_tickers = constancy[constancy >= 2].index # Relaxed from "max 2 neg" to "min 2 pos" for safety

            # Priority: Alpha P4
            p4_stats = subset[(subset["Period"] == "P4") & (subset["Ticker"].isin(valid_tickers))]
            winners = p4_stats.sort_values("Alpha", ascending=False).head(3)

            if winners.empty:
                optimized.append({"Zone": zone, "Regime": reg, "Sectors": "CASH", "Weights": "100%", "Simulated_Sharpe_P4": 0})
                continue

            # Allocation Logic
            # 1. Max Alpha P4 gets 33.3%
            # 2. Others pro-rata
            # 3. Residual to CASH

            # Ensure total alpha > 0 for pro-rata
            alphas = winners["Alpha"].values
            tickers = winners["Ticker"].values

            weights = np.zeros(len(tickers))
            weights[0] = 0.3333
            if len(alphas) > 1:
                remaining = 0.90 - 0.3333 # Macro budget is 90% in final strategy, but here we work on the 100% Macro pocket
                # Let's assume the user wants 100% of the Macro pocket allocated.
                # Requirement: "Max 33,3% par secteur".
                # Propose "Max Alpha P4": 33.3% to leader, pro-rata for others.

                rem_alphas = alphas[1:]
                if rem_alphas.sum() > 0:
                    w_others = (rem_alphas / rem_alphas.sum()) * (1.0 - 0.3333)
                    # Cap others at 33.3% too
                    w_others = np.minimum(w_others, 0.3333)
                    weights[1:] = w_others
                else:
                    weights[1:] = 0

            # Total allocated
            total_w = weights.sum()
            cash_w = 1.0 - total_w

            sim_sharpe = winners["Sharpe"].iloc[0] # Simplification: Sharpe of the leader

            sector_str = ", ".join([f"{t} ({w:.1%})" for t, w in zip(tickers, weights) if w > 0])
            if cash_w > 0.001: sector_str += f", CASH ({cash_w:.1%})"

            optimized.append({
                "Zone": zone, "Regime": reg, "Recommended_Sectors": sector_str,
                "Leader_Alpha_P4": alphas[0], "Simulated_Sharpe_P4": winners["Sharpe"].mean()
            })

    return pd.DataFrame(optimized)

if __name__ == "__main__":
    data = get_full_data()
    all_stats = []
    for zone in ["US", "EU"]:
        regimes = calc_regimes(data, zone)
        stats = analyze_sector_performance(data, regimes, zone)
        all_stats.append(stats)

    full_stats = pd.concat(all_stats)
    matrix = optimize_selection(full_stats)
    matrix.to_csv("master_optimization_matrix.csv", index=False)
    print("Optimization Matrix generated.")
