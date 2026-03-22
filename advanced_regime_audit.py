import pandas as pd
import numpy as np
from strategy_config import US_CONF, EU_CONF, PERIODS, get_all_data, calculate_signals

def run_granular_stats(data, zone_name, conf):
    regime = calculate_signals(data, conf)
    returns = data.pct_change().fillna(0)
    idx_ret = returns[conf["Index"]]
    alpha_rows, corr_rows = [], []
    for p_name, (start, end) in PERIODS.items():
        p_mask = (data.index >= start) & (data.index <= end)
        if not p_mask.any(): continue
        p_rets, p_reg = returns[p_mask], regime[p_mask]
        for reg in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]:
            r_mask = p_reg == reg
            if not r_mask.any(): continue
            r_rets = p_rets[r_mask]
            for s in conf["Sectors"]:
                alpha_rows.append({"Zone": zone_name, "Period": p_name, "Regime": reg, "Sector": s, "Alpha": r_rets[s].mean() * 252 - r_rets[conf["Index"]].mean() * 252})
            c_mat = r_rets[conf["Sectors"]].corr()
            c_melted = c_mat.rename_axis("Sector_A").reset_index().melt(id_vars="Sector_A", var_name="Sector_B", value_name="Correlation")
            c_melted["Zone"], c_melted["Period"], c_melted["Regime"] = zone_name, p_name, reg
            corr_rows.append(c_melted)
    return pd.DataFrame(alpha_rows), pd.concat(corr_rows) if corr_rows else pd.DataFrame()

def run_stress_test(data, zone_name, conf):
    regime = calculate_signals(data, conf)
    crash_periods = [("2007-10-01", "2009-03-09"), ("2020-02-19", "2020-03-23"), ("2022-01-03", "2022-10-12")] if zone_name == "US" else [("2007-07-13", "2009-03-09"), ("2020-02-19", "2020-03-18"), ("2022-01-05", "2022-09-29")]
    stress_rows = []
    for start, end in crash_periods:
        c_mask = (data.index >= start) & (data.index <= end)
        if not c_mask.any(): continue
        c_reg = regime[c_mask]
        c_rets = data[c_mask].pct_change().fillna(0)
        idx_ret = c_rets[conf["Index"]].mean() * 252
        for s in conf["Sectors"]:
            alpha = c_rets[s].mean() * 252 - idx_ret
            if alpha > 0:
                stress_rows.append({"Zone": zone_name, "Crash": f"{start} to {end}", "Regime_Start": c_reg.iloc[0], "Regime_Mid": c_reg.iloc[len(c_reg)//2], "Resilient_Sector": s, "Alpha": alpha})
    return pd.DataFrame(stress_rows)

if __name__ == "__main__":
    data = get_all_data()
    results_alpha, results_corr, results_stress = [], [], []
    for zone, conf in [("US", US_CONF), ("EU", EU_CONF)]:
        a, c = run_granular_stats(data, zone, conf)
        s = run_stress_test(data, zone, conf)
        results_alpha.append(a); results_corr.append(c); results_stress.append(s)
    pd.concat(results_alpha).to_csv("granular_alpha.csv", index=False)
    pd.concat(results_corr).to_csv("granular_correlation.csv", index=False)
    pd.concat(results_stress).to_csv("crash_stress_test.csv", index=False)
    print("Advanced Audit Complete.")
