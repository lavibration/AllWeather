import yfinance as yf
import pandas as pd
import numpy as np
import os

# CONFIGURATION
US_CONF = {
    "Index": "^GSPC",
    "Beta_Pos": {"XLE": 0.35, "XLF": 0.25, "XLB": 0.20, "DBC": 0.20},
    "Beta_Neg": {"XLK": 0.40, "XLY": 0.30, "XLU": 0.15, "XLRE": 0.15},
    "Sectors": ["XLK", "XLY", "XLF", "XLE", "XLB", "XLRE", "XLC", "XLU", "XLV", "XLI", "DBC"]
}
EU_CONF = {
    "Index": "^STOXX50E",
    "Beta_Pos": {"EXV5.DE": 0.35, "EXV1.DE": 0.25, "EXV6.DE": 0.20, "SXRS.DE": 0.20},
    "Beta_Neg": {"EXV3.DE": 0.40, "EXV8.DE": 0.30, "EXV9.DE": 0.15, "EXI5.DE": 0.15},
    "Sectors": ["EXV3.DE", "EXV8.DE", "EXV1.DE", "EXV5.DE", "EXV6.DE", "EXV4.DE", "EXV2.DE", "EXW1.DE", "EXI5.DE", "EXV9.DE", "EXV7.DE", "SXRS.DE"]
}

PERIODS = {
    "P1 (2005-2012)": ("2005-01-01", "2012-12-31"),
    "P2 (2013-2019)": ("2013-01-01", "2019-12-31"),
    "P3 (2020-2022)": ("2020-01-01", "2022-12-31"),
    "P4 (2023-2026)": ("2023-01-01", "2026-12-31")
}

def get_all_data():
    all_t = set([US_CONF["Index"], EU_CONF["Index"]])
    for c in [US_CONF, EU_CONF]:
        all_t.update(c["Beta_Pos"].keys())
        all_t.update(c["Beta_Neg"].keys())
        all_t.update(c["Sectors"])
    data = yf.download(list(all_t), start="2005-01-01")['Close'].ffill()
    return data

def calculate_signals(data, conf):
    price = data[conf["Index"]]
    ma200 = price.rolling(200).mean()
    status_g = pd.Series("UNKNOWN", index=data.index, dtype="string")
    curr = "UNKNOWN"
    for i in range(len(data)):
        p, m = price.iloc[i], ma200.iloc[i]
        if not pd.isna(m):
            if p > 1.01 * m: curr = "UP"
            elif p < 0.99 * m: curr = "DOWN"
        status_g.iloc[i] = curr

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
    return regime, status_g, status_i

def run_granular_stats(data, zone_name, conf):
    regime, _, _ = calculate_signals(data, conf)
    returns = data.pct_change().fillna(0)
    idx_ret = returns[conf["Index"]]

    alpha_rows = []
    corr_rows = []

    for p_name, (start, end) in PERIODS.items():
        p_mask = (data.index >= start) & (data.index <= end)
        if not p_mask.any(): continue
        p_rets = returns[p_mask]
        p_reg = regime[p_mask]

        for reg in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]:
            r_mask = p_reg == reg
            if not r_mask.any(): continue

            r_rets = p_rets[r_mask]
            r_idx_ret = r_rets[conf["Index"]]

            # Alpha
            for s in conf["Sectors"]:
                s_ret = r_rets[s].mean() * 252
                i_ret = r_idx_ret.mean() * 252
                alpha_rows.append({"Zone": zone_name, "Period": p_name, "Regime": reg, "Sector": s, "Alpha": s_ret - i_ret})

            # Full Correlation Matrix for this Regime/Period (Melted for CSV)
            c_mat = r_rets[conf["Sectors"]].corr()
            c_melted = c_mat.rename_axis("Sector_A").reset_index().melt(id_vars="Sector_A", var_name="Sector_B", value_name="Correlation")
            c_melted["Zone"] = zone_name
            c_melted["Period"] = p_name
            c_melted["Regime"] = reg
            corr_rows.append(c_melted)

    return pd.DataFrame(alpha_rows), pd.concat(corr_rows) if corr_rows else pd.DataFrame()

def run_stress_test(data, zone_name, conf):
    regime, sg, si = calculate_signals(data, conf)
    price = data[conf["Index"]]

    crash_periods = []
    if zone_name == "US":
        crash_periods = [("2007-10-01", "2009-03-09"), ("2020-02-19", "2020-03-23"), ("2022-01-03", "2022-10-12")]
    else:
        crash_periods = [("2007-07-13", "2009-03-09"), ("2020-02-19", "2020-03-18"), ("2022-01-05", "2022-09-29")]

    stress_rows = []
    for start, end in crash_periods:
        c_mask = (data.index >= start) & (data.index <= end)
        if not c_mask.any(): continue

        c_reg = regime[c_mask]
        start_reg = c_reg.iloc[0]
        mid_reg = c_reg.iloc[len(c_reg)//2]

        c_rets = data[c_mask].pct_change().fillna(0)
        idx_ret = c_rets[conf["Index"]].mean() * 252

        for s in conf["Sectors"]:
            s_ret = c_rets[s].mean() * 252
            alpha = s_ret - idx_ret
            if alpha > 0:
                stress_rows.append({
                    "Zone": zone_name, "Crash": f"{start} to {end}",
                    "Regime_Start": start_reg, "Regime_Mid": mid_reg,
                    "Resilient_Sector": s, "Alpha": alpha
                })
    return pd.DataFrame(stress_rows)

def run_efficiency_analysis(data, zone_name, conf):
    regime, _, _ = calculate_signals(data, conf)
    valid_regs = regime[regime != "UNKNOWN"]

    # Transition Matrix
    transitions = pd.DataFrame({"From": valid_regs, "To": valid_regs.shift(-1)}).dropna()
    matrix = pd.crosstab(transitions["From"], transitions["To"], normalize='index')
    matrix_csv = matrix.reset_index().melt(id_vars="From", var_name="To", value_name="Prob")
    matrix_csv["Zone"] = zone_name

    # Improved Lead-Lag Analysis: Cross-correlation between Signal and Returns
    rets = data[conf["Index"]].pct_change().fillna(0)
    # Convert regime to numeric for correlation
    reg_map = {"GOLDILOCKS": 1, "REFLATION": 2, "STAGFLATION": 3, "DEFLATION": 4}
    reg_num = valid_regs.map(reg_map).astype(float)

    lead_lag_rows = []
    shared_idx = reg_num.index.intersection(rets.index)
    rn = reg_num.loc[shared_idx]
    rt = rets.loc[shared_idx]

    for lag in range(-10, 11):
        corr = rn.corr(rt.shift(-lag))
        lead_lag_rows.append({"Zone": zone_name, "Lag_Days": lag, "Cross_Corr": corr})

    return matrix_csv, pd.DataFrame(lead_lag_rows)

if __name__ == "__main__":
    data = get_all_data()

    results_alpha = []
    results_corr = []
    results_stress = []
    results_trans = []
    results_ll = []

    for zone, conf in [("US", US_CONF), ("EU", EU_CONF)]:
        a, c = run_granular_stats(data, zone, conf)
        s = run_stress_test(data, zone, conf)
        m, l = run_efficiency_analysis(data, zone, conf)

        results_alpha.append(a)
        results_corr.append(c)
        results_stress.append(s)
        results_trans.append(m)
        results_ll.append(l)

    pd.concat(results_alpha).to_csv("granular_alpha.csv", index=False)
    pd.concat(results_corr).to_csv("granular_correlation.csv", index=False)
    pd.concat(results_stress).to_csv("crash_stress_test.csv", index=False)
    pd.concat(results_trans).to_csv("transition_matrix.csv", index=False)
    pd.concat(results_ll).to_csv("lead_lag_analysis.csv", index=False)

    print("Advanced Audit Complete. All outputs are CSV.")
