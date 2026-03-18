import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def run_deep_decomposition():
    # 1. PEAK TO TROUGH WINDOW
    # Based on previous run: Peak 2020-02-13, Trough 2020-03-18
    peak_date = "2020-02-13"
    trough_date = "2020-03-18"

    # Download detailed data for the period
    tickers_list = ["^STOXX50E", "GLD", "EXV1.DE", "EXV9.DE", "EXV8.DE", "EXV3.DE", "EXV4.DE", "EXV2.DE", "EXV7.DE", "SXRS.DE", "EXV6.DE", "EXW1.DE", "EXI5.DE"]

    data = yf.download(tickers_list, start="2018-01-01", end="2021-04-01")
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']

    data = data.ffill().dropna(subset=["^STOXX50E"])
    available_tickers = data.columns.tolist()
    print(f"Tickers téléchargeables: {available_tickers}")

    # Calculate returns on period
    period_data = data.loc[peak_date:trough_date]
    perf = (period_data.iloc[-1] / period_data.iloc[0]) - 1

    # 2. ATTRIBUTION
    print("--- RAPPORT DÉCOMPOSITION DU RISQUE (CRASH COVID 2020) ---")
    print(f"Période: {peak_date} au {trough_date}")
    print(f"Baisse Euro Stoxx 50: {perf['^STOXX50E']*100:.2f}%")
    print(f"Baisse Poche OR (10%): {perf['GLD']*100:.2f}%")

    # Sectors held in "Optimized" GOLDILOCKS (Feb 13 - Feb 27)
    # Banks (EXV1.DE), Utilities (EXV9.DE), Conso (EXV8.DE)
    print("\nPerformance des secteurs 'Optimisés' (détenus au début du crash):")
    print(f"Banques (EXV1.DE): {perf['EXV1.DE']*100:.2f}%")
    print(f"Utilities (EXV9.DE): {perf['EXV9.DE']*100:.2f}%")
    print(f"Conso (EXV8.DE): {perf['EXV8.DE']*100:.2f}%")

    # Sectors in "Strict" DEFLATION (requested by user)
    # Santé (EXV4.DE), Automobile (EXV2.DE), Telecoms (EXV7.DE)
    print("\nPerformance des secteurs 'Strict' (Paramètres initiaux):")
    print(f"Santé (EXV4.DE): {perf['EXV4.DE']*100:.2f}%")
    print(f"Auto (EXV2.DE): {perf['EXV2.DE']*100:.2f}%")
    print(f"Telecoms (EXV7.DE): {perf['EXV7.DE']*100:.2f}%")

    # 3. SIGNAL ANALYSIS
    # MA200 exit signal
    price = data['^STOXX50E']
    ma200 = price.rolling(200).mean()
    h_up, h_down = 1.01 * ma200, 0.99 * ma200

    # Debug
    print(f"Price at Peak: {price.loc[peak_date]}")
    print(f"MA200 at Peak: {ma200.loc[peak_date]}")

    # Trigger date
    mask = (price < h_down) & (price.index >= peak_date)
    if not mask.any():
        print("Hysteresis trigger not found after peak.")
        trigger_date = price.index[-1]
    else:
        trigger_date = price[mask].index[0]
    days_to_exit = (trigger_date - pd.to_datetime(peak_date)).days
    loss_at_exit = (price.loc[trigger_date] / price.loc[peak_date]) - 1

    print(f"\n--- ANALYSE DU SIGNAL ---")
    print(f"Signal de sortie (Hystérésis 1.0%): {trigger_date.date()}")
    print(f"Latence (Lag): {days_to_exit} jours")
    print(f"Perte subie AVANT la sortie du régime de croissance: {loss_at_exit*100:.2f}%")

    # Test 0% Hysteresis
    mask_0 = (price < ma200) & (price.index >= peak_date)
    if not mask_0.any():
        trigger_0 = price.index[-1]
    else:
        trigger_0 = price[mask_0].index[0]

    loss_0 = (price.loc[trigger_0] / price.loc[peak_date]) - 1
    print(f"Signal de sortie (Hystérésis 0.0%): {trigger_0.date()} ({ (trigger_0 - pd.to_datetime(peak_date)).days } jours)")
    print(f"Perte subie (0.0% Hysteresis): {loss_0*100:.2f}%")

    # 4. OPTIMIZED VS HISTORICAL MDD
    # Let's calculate the portfolio value for both
    # 10% Gold + 90% (Sectors)

    # Optimized (Feb 13 - Feb 27: Goldilocks | Feb 27 - Mar 18: Deflation)
    # Note: Both regimes in my "optimized" version had Banks.

    # Weight Simulation
    def sim_pocket(regime_weights, start_date, end_date):
        p_data = data.loc[start_date:end_date]
        rets = p_data.pct_change().fillna(0)
        val = 1000.0
        vals = [val]
        for i in range(1, len(rets)):
            r = 0.10 * rets.iloc[i]['GLD']
            for tk, wt in regime_weights.items():
                if tk != "CASH": r += 0.90 * wt * rets.iloc[i][tk]
            val *= (1 + r)
            vals.append(val)
        return pd.Series(vals, index=p_data.index)

    # Simplified simulation (assuming regime switch at trigger_date)
    opt_weights_1 = {"EXV1.DE": 0.333, "EXV9.DE": 0.333, "EXV8.DE": 0.286} # Goldilocks Opt
    opt_weights_2 = {"EXV9.DE": 0.333, "EXV3.DE": 0.333, "EXV1.DE": 0.324} # Deflation Opt

    strict_weights_1 = {"EXV4.DE": 0.333, "EXV3.DE": 0.333, "EXI5.DE": 0.333} # Goldilocks Strict (Proxy: Santé, Tech, Real Estate)
    strict_weights_2 = {"EXV4.DE": 0.333, "EXV2.DE": 0.333, "EXV7.DE": 0.333} # Deflation Strict (Santé, Auto, Telecoms)

    # Simulation
    def complex_sim(w1, w2, trigger):
        v = 1000.0
        hist = []
        rets = data.pct_change().fillna(0)
        for d in data.loc[peak_date:trough_date].index:
            weights = w1 if d < trigger else w2
            r = 0.10 * rets.loc[d, 'GLD'] if 'GLD' in rets.columns else 0
            for tk, wt in weights.items():
                if tk in rets.columns:
                    r += 0.90 * wt * rets.loc[d, tk]
                elif tk == "CASH":
                    pass # 0 return
            v *= (1 + r)
            hist.append(v)
        return pd.Series(hist, index=data.loc[peak_date:trough_date].index)

    v_opt = complex_sim(opt_weights_1, opt_weights_2, trigger_date)
    v_strict = complex_sim(strict_weights_1, strict_weights_2, trigger_date)

    print(f"\n--- COMPARAISON MDD SUR LA PÉRIODE ---")
    print(f"Max Drawdown (Matrice OPTIMISÉE): {((v_opt - v_opt.cummax())/v_opt.cummax()).min()*100:.2f}%")
    print(f"Max Drawdown (Matrice STRICT): {((v_strict - v_strict.cummax())/v_strict.cummax()).min()*100:.2f}%")

    # UNDERWATER PLOT
    plt.figure(figsize=(12, 6))
    dd_opt = (v_opt - v_opt.cummax()) / v_opt.cummax()
    dd_strict = (v_strict - v_strict.cummax()) / v_strict.cummax()
    plt.fill_between(dd_opt.index, dd_opt, 0, color='red', alpha=0.3, label="Optimisée (Alpha-Focused)")
    plt.plot(dd_strict, color='blue', label="Strict (Paramètres d'origine)")
    plt.title("Underwater Plot: Comparaison des Matrices pendant le Crash 2020")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)
    plt.savefig("underwater_crash_comparison.png")
    print("\nGraphique 'underwater_crash_comparison.png' généré.")

if __name__ == "__main__":
    run_deep_decomposition()
