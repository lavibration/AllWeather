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

PERIODS = {
    "P1 (2005-2012)": ("2005-01-01", "2012-12-31"),
    "P2 (2013-2019)": ("2013-01-01", "2019-12-31"),
    "P3 (2020-2022)": ("2020-01-01", "2022-12-31"),
    "P4 (2023-2026)": ("2023-01-01", "2026-12-31")
}

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

def audit_period(data, zone, period_name, start, end):
    print(f"\n=========================================")
    print(f"   AUDIT: {zone} | {period_name}")
    print(f"=========================================")

    # Slice data
    p_data = data[(data.index >= start) & (data.index <= end)]
    if p_data.empty: return None

    sigs = calc_signals(p_data, zone)
    rets = p_data.pct_change().loc[sigs.index]
    idx_tk = TICKERS_CONFIG[zone]["Index"]

    # 1. Lead-Lag
    print(f"\n--- 1. Lead-Lag Analysis ---")
    ll_res = []
    for sig_name in ["Growth", "Inflation"]:
        sig = sigs[sig_name]
        transitions = sig[sig != sig.shift(1)].dropna()
        for date in transitions.index:
            new_state = transitions.loc[date]
            for lag in [20, 40, 60]:
                fwd = (1 + rets.shift(-lag).loc[date:date + pd.Timedelta(days=lag*1.5)].iloc[:lag]).prod() - 1
                if idx_tk in fwd.index: ll_res.append({"Signal": sig_name, "To": new_state, "Lag": lag, "Avg_Fwd_Ret": fwd[idx_tk]})
    if ll_res:
        print(pd.DataFrame(ll_res).groupby(["Signal", "To", "Lag"]).mean())
    else: print("No transitions found in period.")

    # 2. Sector Matrix
    print(f"\n--- 2. Sector Matrix Audit ---")
    sm_res = []
    for reg in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]:
        mask = sigs["Regime"] == reg
        if not mask.any(): continue
        reg_rets = rets.loc[mask]
        for sector in TICKERS_CONFIG[zone]["Regimes"][reg]:
            if sector not in reg_rets.columns: continue
            s_rets = reg_rets[sector].dropna()
            if s_rets.empty: continue
            mdd = ((1 + s_rets).cumprod() / (1 + s_rets).cumprod().cummax() - 1).min()
            alpha = (s_rets.mean() - reg_rets[idx_tk].mean()) * 252
            sm_res.append({"Regime": reg, "Sector": sector, "Mean_Ann": s_rets.mean()*252, "Win_Rate": (s_rets>0).mean(), "Max_DD": mdd, "Alpha_vs_Idx": alpha})
    sm_df = pd.DataFrame(sm_res)
    print(sm_df.to_string())

    # 3. Persistence
    print(f"\n--- 3. Persistence & Stability ---")
    sig_1 = sigs["Regime"]
    sig_0 = calc_signals(p_data, zone, hyst=0.00)["Regime"]
    def get_dur(s): return s.groupby((s != s.shift(1)).cumsum()).size()
    print(f"Avg Regime Duration (1% Hyst): {get_dur(sig_1).mean():.1f} days")
    print(f"Faux signaux (<10d) 0% Hyst: {(get_dur(sig_0)<10).sum()}")
    print(f"Faux signaux (<10d) 1% Hyst: {(get_dur(sig_1)<10).sum()}")

    return sm_df

def get_transitions(data, zone):
    sigs = calc_signals(data, zone)
    t = sigs[sigs["Regime"] != sigs["Regime"].shift(1)].dropna(subset=["Regime"])
    log = []
    for idx, row in t.iterrows():
        log.append({"Date": idx.strftime('%Y-%m-%d'), "Regime": row["Regime"], "Price": row["Price"], "Inflation_Ratio": row["Ratio"]})
    return pd.DataFrame(log)

if __name__ == "__main__":
    df = get_data()
    all_sm = {}

    for zone in ["US", "EU"]:
        zone_sm = {}
        for p_name, (start, end) in PERIODS.items():
            res = audit_period(df, zone, p_name, start, end)
            if res is not None: zone_sm[p_name] = res
        all_sm[zone] = zone_sm

    print(f"\n=========================================")
    print(f"   ANALYSE DE DÉVIATION (DRIFT)")
    print(f"=========================================")
    for zone in ["US", "EU"]:
        print(f"\n--- Drift analysis ({zone}) ---")
        sm2 = all_sm[zone].get("P2 (2013-2019)")
        sm4 = all_sm[zone].get("P4 (2023-2026)")
        if sm2 is not None and sm4 is not None:
            merged = sm2[["Regime", "Sector", "Alpha_vs_Idx"]].merge(sm4[["Regime", "Sector", "Alpha_vs_Idx"]], on=["Regime", "Sector"], suffixes=("_P2", "_P4"))
            print(merged.to_string())
            neg_alpha = merged[merged["Alpha_vs_Idx_P4"] < 0]
            if not neg_alpha.empty:
                print(f"\nSecteurs avec Alpha négatif en P4:")
                print(neg_alpha[["Regime", "Sector", "Alpha_vs_Idx_P4"]])
            else: print("\nAucun secteur avec Alpha négatif en P4 parmi les actifs historiques.")
        else: print("Period 2 or 4 data missing for drift analysis.")

    print(f"\n=========================================")
    print(f"   REGIME TRANSITION LOG")
    print(f"=========================================")
    for zone in ["US", "EU"]:
        print(f"\n--- Transition Log ({zone}) ---")
        print(get_transitions(df, zone).to_string())
