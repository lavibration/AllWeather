import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from strategy_config import US_CONF, EU_CONF, PERIODS, get_all_data, calculate_signals, GLD, CASH

def backtest(data, zone_name):
    conf = US_CONF if zone_name == "US" else EU_CONF
    regime = calculate_signals(data, conf)
    regime_shifted = regime.shift(1).fillna("UNKNOWN")

    alloc_map = conf["Alloc"]
    rets = data.pct_change().fillna(0)
    rets[CASH] = 0.0

    rebalance_dates = data.resample('ME').last().index
    current_val = 1.0
    val_history = []
    current_weights = {}

    for i in range(len(data)):
        date = data.index[i]
        if current_weights:
            day_ret = sum(current_weights[t] * rets.loc[date, t] for t in current_weights)
            current_val *= (1 + day_ret)

        if date in rebalance_dates:
            reg = regime_shifted.loc[date]
            target_weights = {GLD: 0.10}
            if reg != "UNKNOWN":
                macro_alloc = alloc_map[reg]
                for ticker, weight in macro_alloc.items():
                    target_weights[ticker] = target_weights.get(ticker, 0) + weight * 0.90
            else:
                target_weights[CASH] = target_weights.get(CASH, 0) + 0.90

            all_t = set(list(target_weights.keys()) + list(current_weights.keys()))
            turnover = sum(abs(target_weights.get(t,0) - current_weights.get(t,0)) for t in all_t)
            current_val *= (1 - turnover * 0.0010)
            current_weights = target_weights.copy()

        val_history.append(current_val)
    return pd.Series(val_history, index=data.index, name=f"Value_{zone_name}")

def calculate_audit_stats(values, data, zone_name):
    conf = US_CONF if zone_name == "US" else EU_CONF
    rets = values.pct_change().fillna(0)
    idx_rets = data[conf["Index"]].pct_change().fillna(0)

    stats_rows = []
    for p_name, (start, end) in PERIODS.items():
        mask = (values.index >= start) & (values.index <= end)
        if not mask.any(): continue
        p_rets, p_idx, p_vals = rets[mask], idx_rets[mask], values[mask]

        years = (p_vals.index[-1] - p_vals.index[0]).days / 365.25
        cagr = (p_vals.iloc[-1] / p_vals.iloc[0])**(1/max(years, 0.001)) - 1
        vol = p_rets.std() * np.sqrt(252)
        sharpe = cagr / vol if vol != 0 else 0
        alpha = p_rets - p_idx
        ir = alpha.mean() / alpha.std() * np.sqrt(252) if alpha.std() != 0 else 0
        s_skew = skew(p_rets)
        cvar_95 = p_rets[p_rets <= p_rets.quantile(0.05)].mean()
        roll_max = p_vals.cummax()
        mdd = ((p_vals - roll_max) / roll_max).min()

        stats_rows.append({"Zone": zone_name, "Period": p_name, "CAGR": cagr, "Sharpe": sharpe, "IR": ir, "Skewness": s_skew, "CVaR_95": cvar_95, "MDD": mdd})
    return pd.DataFrame(stats_rows)

if __name__ == "__main__":
    data = get_all_data()
    v_us, v_eu = backtest(data, "US"), backtest(data, "EU")
    s_us, s_eu = calculate_audit_stats(v_us, data, "US"), calculate_audit_stats(v_eu, data, "EU")
    pd.concat([s_us, s_eu]).to_csv("backtest_performance_audit.csv", index=False)
    pd.concat([v_us, v_eu], axis=1).to_csv("backtest_timeseries.csv")

    plt.figure(figsize=(12, 6))
    plt.plot(v_us, label="Stratégie US")
    plt.plot(v_eu, label="Stratégie EU")
    plt.plot((1 + data[US_CONF["Index"]].pct_change().fillna(0)).cumprod(), label="Benchmark US", linestyle='--')
    plt.plot((1 + data[EU_CONF["Index"]].pct_change().fillna(0)).cumprod(), label="Benchmark EU", linestyle='--')
    plt.yscale('log'); plt.title("Performance Cumulative (Log)"); plt.legend(); plt.grid(True)
    plt.savefig("backtest_visual_audit.png")
    print("Backtest terminé.")
