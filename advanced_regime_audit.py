import pandas as pd
import numpy as np
from strategy_config import US_CONF, EU_CONF, PERIODS, get_all_data, calculate_signals

def run_granular_stats(data, zone_name, conf):
    regime = calculate_signals(data, conf)
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

            for s in conf["Sectors"]:
                s_ret = r_rets[s].mean() * 252
                i_ret = r_idx_ret.mean() * 252
                alpha_rows.append({"Zone": zone_name, "Period": p_name, "Regime": reg, "Sector": s, "Alpha": s_ret - i_ret})

            c_mat = r_rets[conf["Sectors"]].corr()
            c_melted = c_mat.rename_axis("Sector_A").reset_index().melt(id_vars="Sector_A", var_name="Sector_B", value_name="Correlation")
            c_melted["Zone"] = zone_name
            c_melted["Period"] = p_name
            c_melted["Regime"] = reg
            corr_rows.append(c_melted)

    return pd.DataFrame(alpha_rows), pd.concat(corr_rows) if corr_rows else pd.DataFrame()

def run_stress_test(data, zone_name, conf):
    regime = calculate_signals(data, conf)
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
    regime = calculate_signals(data, conf)
    valid_regs = regime[regime != "UNKNOWN"]
    transitions = pd.DataFrame({"From": valid_regs, "To": valid_regs.shift(-1)}).dropna()
    matrix = pd.crosstab(transitions["From"], transitions["To"], normalize='index')
    matrix_csv = matrix.reset_index().melt(id_vars="From", var_name="To", value_name="Prob")
    matrix_csv["Zone"] = zone_name

    rets = data[conf["Index"]].pct_change().fillna(0)
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
    results_alpha, results_corr, results_stress, results_trans, results_ll = [], [], [], [], []
    for zone, conf in [("US", US_CONF), ("EU", EU_CONF)]:
        a, c = run_granular_stats(data, zone, conf)
        s = run_stress_test(data, zone, conf)
        m, l = run_efficiency_analysis(data, zone, conf)
        results_alpha.append(a); results_corr.append(c); results_stress.append(s); results_trans.append(m); results_ll.append(l)

    pd.concat(results_alpha).to_csv("granular_alpha.csv", index=False)
    pd.concat(results_corr).to_csv("granular_correlation.csv", index=False)
    pd.concat(results_stress).to_csv("crash_stress_test.csv", index=False)
    pd.concat(results_trans).to_csv("transition_matrix.csv", index=False)
    pd.concat(results_ll).to_csv("lead_lag_analysis.csv", index=False)
    print("Advanced Audit Complete.")
