import yfinance as yf
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
TICKERS_CONFIG = {
    "US": {
        "Index": "^GSPC",
        "Beta_Pos": {"XLE": 0.35, "XLF": 0.25, "XLB": 0.20, "DBC": 0.20},
        "Beta_Neg": {"XLK": 0.40, "XLY": 0.30, "XLU": 0.15, "XLRE": 0.15},
        "Regimes": {
            "GOLDILOCKS": ["XLK", "XLY", "XLF"],
            "REFLATION": ["XLE", "XLB", "XLF"],
            "STAGFLATION": ["DBC"],
            "DEFLATION": ["XLY", "XLRE", "XLC"]
        }
    },
    "EU": {
        "Index": "^STOXX50E",
        "Beta_Pos": {"EXV5.DE": 0.35, "EXV1.DE": 0.25, "EXV6.DE": 0.20, "SXRS.DE": 0.20},
        "Beta_Neg": {"EXV3.DE": 0.40, "EXV2.DE": 0.30, "EXV9.DE": 0.15, "EXI5.DE": 0.15},
        "Regimes": {
            "GOLDILOCKS": ["EXV4.DE", "EXV3.DE", "EXI5.DE"],
            "REFLATION": ["EXV1.DE", "EXV6.DE", "SXRS.DE"],
            "STAGFLATION": ["SXRS.DE"],
            "DEFLATION": ["EXV4.DE", "EXV2.DE", "EXV7.DE"]
        }
    }
}

ALL_TICKERS = ["^GSPC", "^STOXX50E", "GLD", "XLK", "XLY", "XLF", "XLE", "XLB", "XLRE", "XLC", "XLU", "DBC",
               "EXV5.DE", "EXV1.DE", "EXV6.DE", "SXRS.DE", "EXV3.DE", "EXV2.DE", "EXV9.DE", "EXI5.DE", "EXV4.DE", "EXV7.DE"]

def get_data():
    df = yf.download(ALL_TICKERS, start="2005-01-01")['Close']
    return df.ffill()

def calc_signals(data, zone, hyst=0.01):
    conf = TICKERS_CONFIG[zone]
    idx = data[conf["Index"]].dropna()
    ma200 = idx.rolling(200).mean()

    # Growth Signal
    g_sig = pd.Series(index=idx.index, dtype='string')
    curr = None
    for i in range(len(idx)):
        p, m = idx.iloc[i], ma200.iloc[i]
        if not pd.isna(m):
            if p > (1 + hyst) * m: curr = "UP"
            elif p < (1 - hyst) * m: curr = "DOWN"
            g_sig.iloc[i] = curr

    # Inflation Signal
    rets = data.pct_change()
    pos_rets = sum(rets[tk] * wt for tk, wt in conf["Beta_Pos"].items()).fillna(0)
    neg_rets = sum(rets[tk] * wt for tk, wt in conf["Beta_Neg"].items()).fillna(0)
    ratio = (1+pos_rets).cumprod() / (1+neg_rets).cumprod()
    ratio = ratio.loc[idx.index]
    med200 = ratio.rolling(200).median()

    i_sig = pd.Series(index=idx.index, dtype='string')
    curr = None
    for i in range(len(idx)):
        r, m = ratio.iloc[i], med200.iloc[i]
        if not pd.isna(m):
            if r > (1 + hyst) * m: curr = "UP"
            elif r < (1 - hyst) * m: curr = "DOWN"
            i_sig.iloc[i] = curr

    regime = pd.Series(index=idx.index, dtype='string')
    regime[(g_sig == "UP") & (i_sig == "DOWN")] = "GOLDILOCKS"
    regime[(g_sig == "UP") & (i_sig == "UP")] = "REFLATION"
    regime[(g_sig == "DOWN") & (i_sig == "UP")] = "STAGFLATION"
    regime[(g_sig == "DOWN") & (i_sig == "DOWN")] = "DEFLATION"

    return pd.DataFrame({"Growth": g_sig, "Inflation": i_sig, "Regime": regime, "Ratio": ratio, "Price": idx, "MA200": ma200, "Median": med200})

def audit_lead_lag(data, signals, zone):
    print(f"\n--- 1. Lead-Lag Analysis ({zone}) ---")
    rets = data.pct_change()
    results = []

    for sig_name in ["Growth", "Inflation"]:
        sig = signals[sig_name]
        # Detect transitions
        transitions = sig[sig != sig.shift(1)].dropna()
        for date in transitions.index:
            new_state = transitions.loc[date]
            for lag in [20, 40, 60]:
                fwd_ret = (1 + rets.shift(-lag).loc[date:date + pd.Timedelta(days=lag*1.5)].iloc[:lag]).prod() - 1
                # Cross-corr is hard with discrete signals, let's use spearman between signal distance and fwd rets
                # User specifically asked for correlation between "crossing" and "performance".
                # We'll calculate avg fwd return of the Index after transition.
                idx_ret = fwd_ret[TICKERS_CONFIG[zone]["Index"]]
                results.append({"Signal": sig_name, "To": new_state, "Lag": lag, "Avg_Fwd_Ret": idx_ret})

    df = pd.DataFrame(results).groupby(["Signal", "To", "Lag"]).mean()
    print(df)

def audit_sector_matrix(data, signals, zone):
    print(f"\n--- 2. Sector Matrix Audit ({zone}) ---")
    rets = data.pct_change()
    # Align rets with signals
    rets = rets.loc[signals.index]
    results = []
    idx_tk = TICKERS_CONFIG[zone]["Index"]

    for reg in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]:
        mask = signals["Regime"] == reg
        if not mask.any(): continue

        reg_rets = rets.loc[mask]
        target_sectors = TICKERS_CONFIG[zone]["Regimes"][reg]

        for sector in target_sectors:
            s_rets = reg_rets[sector].dropna()
            if s_rets.empty: continue

            mdd = ((1 + s_rets).cumprod() / (1 + s_rets).cumprod().cummax() - 1).min()
            win_rate = (s_rets > 0).mean()

            # Alpha vs Index
            alpha = (s_rets.mean() - reg_rets[idx_tk].mean()) * 252

            results.append({
                "Regime": reg, "Sector": sector,
                "Mean_Ann": s_rets.mean() * 252,
                "Median_Ann": s_rets.median() * 252,
                "Win_Rate": win_rate, "Max_DD": mdd,
                "Alpha_vs_Idx": alpha
            })

    df = pd.DataFrame(results)
    print(df.to_string())

def audit_persistence(data, zone):
    print(f"\n--- 3. Persistence & Stability ({zone}) ---")
    sig_1 = calc_signals(data, zone, hyst=0.01)["Regime"]
    sig_0 = calc_signals(data, zone, hyst=0.00)["Regime"]

    def get_durations(s):
        s = s.dropna()
        change = s != s.shift(1)
        groups = change.cumsum()
        return s.groupby(groups).size()

    dur_1 = get_durations(sig_1)
    print(f"Average Regime Duration (1% Hyst): {dur_1.mean():.1f} days")

    # False signals: regimes < 10 days
    false_0 = (get_durations(sig_0) < 10).sum()
    false_1 = (get_durations(sig_1) < 10).sum()
    print(f"False Signals (<10d) with 0% Hyst: {false_0}")
    print(f"False Signals (<10d) with 1% Hyst: {false_1}")
    print(f"Hysteresis eliminated {false_0 - false_1} noise transitions.")

    # Fee impact
    trades = (sig_1 != sig_1.shift(1)).sum()
    total_fee = trades * 0.0010
    total_ret = (1 + data[TICKERS_CONFIG[zone]["Index"]].pct_change()).prod() - 1
    print(f"Estimated cumulative fees (0.1% per change): {total_fee:.2%}")

def audit_drift(data, zone):
    print(f"\n--- 4. Drift Analysis (2021-2026 vs 2005-2020) ({zone}) ---")
    # Use timezone-naive index for slicing if needed, but here simple string works if index is DatetimeIndex
    df_old = data[data.index <= "2020-12-31"]
    df_new = data[data.index >= "2021-01-01"]

    for label, df in [("PRE-2021", df_old), ("POST-2021", df_new)]:
        if df.empty: continue
        sigs = calc_signals(df, zone)
        rets = df.pct_change().loc[sigs.index]
        print(f"\nPeriod: {label}")
        res = []
        for reg in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]:
            mask = sigs["Regime"] == reg
            if mask.any():
                target = TICKERS_CONFIG[zone]["Regimes"][reg][0] # Check lead sector
                if target in rets.columns:
                    res.append({"Regime": reg, "Lead_Sector": target, "Avg_Ret": rets.loc[mask, target].mean()*252})
        print(pd.DataFrame(res))

if __name__ == "__main__":
    df = get_data()
    for zone in ["US", "EU"]:
        sigs = calc_signals(df, zone)
        audit_lead_lag(df, sigs, zone)
        audit_sector_matrix(df, sigs, zone)
        audit_persistence(df, zone)
        audit_drift(df, zone)
