import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from strategy_config import US_CONF, EU_CONF, get_all_data, calculate_signals

def run_risk_analysis(data, zone_name, conf):
    regime = calculate_signals(data, conf)
    rets = data.pct_change().fillna(0)
    idx_ret = rets[conf["Index"]]
    risk_rows, corr_results = [], []
    for reg in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]:
        mask = regime == reg
        if not mask.any(): continue
        reg_rets, reg_idx_ret = rets[mask], idx_ret[mask]
        m_rets = (1 + reg_rets).resample('ME').prod() - 1
        days_per_month = mask.resample('ME').sum()
        active_months = days_per_month[days_per_month > 0].index
        m_rets = m_rets.loc[active_months]
        for s in conf["Sectors"]:
            alpha_daily = reg_rets[s] - reg_idx_ret
            ir = alpha_daily.mean() / alpha_daily.std() * np.sqrt(252) if alpha_daily.std() != 0 else 0
            win_rate = ((m_rets[s] - m_rets[conf["Index"]]) > 0).mean()
            risk_rows.append({"Zone": zone_name, "Regime": reg, "Sector": s, "IR": ir, "Win_Rate": win_rate, "Skewness": skew(reg_rets[s]), "Kurtosis": kurtosis(reg_rets[s]), "CVaR_95": reg_rets[s][reg_rets[s] <= reg_rets[s].quantile(0.05)].mean()})
        c_mat = reg_rets[conf["Sectors"]].corr()
        for i in range(len(c_mat.columns)):
            for j in range(i+1, len(c_mat.columns)):
                if c_mat.iloc[i,j] > 0.8:
                    corr_results.append({"Zone": zone_name, "Regime": reg, "Sector_A": c_mat.columns[i], "Sector_B": c_mat.columns[j], "Correlation": c_mat.iloc[i,j]})
    return pd.DataFrame(risk_rows), pd.DataFrame(corr_results)

if __name__ == "__main__":
    data = get_all_data()
    r_us, c_us = run_risk_analysis(data, "US", US_CONF)
    r_eu, c_eu = run_risk_analysis(data, "EU", EU_CONF)
    final_risk = pd.concat([r_us, r_eu])
    final_risk.to_csv("portfolio_risk_metrics.csv", index=False)
    pd.concat([c_us, c_eu]).to_csv("regime_correlation_clusters.csv", index=False)
    plt.figure(figsize=(15, 10))
    sns.scatterplot(data=final_risk, x="CVaR_95", y="IR", hue="Regime", style="Zone", s=100)
    plt.axhline(0, color='gray', linestyle='--'); plt.grid(True); plt.savefig("portfolio_risk_analysis.png")
    print("Risk analysis complete.")
