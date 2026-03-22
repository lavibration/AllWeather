import yfinance as yf
import pandas as pd
import numpy as np
import os

# =============================================================================
# 1. CONFIGURATION & PARAMETERS
# =============================================================================

GLD_TICKER  = "GLD"
US_INDEX    = "^GSPC"
EU_INDEX    = "^STOXX50E"

# --- Signal parameters ---
SIGNAL_WINDOW       = 200   # fenêtre médiane pour les deux signaux
CONFIRM_DAYS        = 15    # filtre de persistance : N jours consécutifs avant confirmation
BUFFER_ASYMM_WINDOW = 60    # fenêtre de vol pour buffer asymétrique (demi-écart type)

# --- Beta Portfolios (inchangés) ---
BETA_POS_WEIGHTS = {"Energy": 0.35, "Financials": 0.25, "Materials": 0.20, "Commodities": 0.20}
BETA_NEG_WEIGHTS = {"GrowthTech": 0.40, "ConsDisc": 0.30, "Utilities": 0.15, "GrowthRE": 0.15}

# --- Ticker Mappings (inchangés) ---
TICKERS_US = {
    "Energy": "XLE", "Financials": "XLF", "Materials": "XLB", "Commodities": "DBC",
    "GrowthTech": "XLK", "ConsDisc": "XLY", "Utilities": "XLU", "GrowthRE": "XLRE",
    "Comm": "XLC"
}

TICKERS_EU = {
    "Energy": "EXV5.DE", "Financials": "EXV1.DE", "Materials": "EXV6.DE", "Commodities": "DBC",
    "GrowthTech": "EXV3.DE", "ConsDisc": "EXV8.DE", "Utilities": "EXV9.DE", "GrowthRE": "EXI5.DE",
    "Health": "EXV4.DE", "Auto": "EXV2.DE", "Telecoms": "EXV7.DE"
}

# =============================================================================
# 2. ALLOCATION MATRIX  — révisée d'après analyse sectorielle par période
# =============================================================================
#
# Règle : poids dans la poche macro (90%). GLD occupe 10% fixe.
# Les poids ci-dessous s'appliquent sur les 90% restants.
#
# Changements vs v1 :
#   US GOLDILOCKS  : inchangé (XLK/XLY/XLF validés sur 4 périodes, XLF std=4.3)
#   US REFLATION   : XLF retiré (P4 : -18.5%), DBC ajouté (stable sur 4 périodes)
#   US STAGFLATION : inchangé (XLE incohérent entre périodes : P2=-53%, P4=-12%)
#   US DEFLATION   : XLRE retiré (P3=-54%), cash défensif ajouté
#   EU GOLDILOCKS  : inchangé (EXV1 trop volatile std=23.2, EXI5 stable std=5.3 gardé)
#   EU REFLATION   : légère surpondération EXV6 (meilleur actif, stable sur 4 périodes)
#   EU STAGFLATION : inchangé (DBC peut perdre -25% en P4 → cash 67% justifié)
#   EU DEFLATION   : EXV4 retiré (dégradation P3/P4), EXV2 retiré (incohérent),
#                    EXV3 ajouté (positif sur 4 périodes), EXV8 ajouté (stable std=6.2)

ALLOC_US = {
    "GOLDILOCKS":  {"XLK": 0.333, "XLY": 0.333, "XLF": 0.334},          # inchangé
    "REFLATION":   {"XLE": 0.400, "DBC": 0.300, "XLB": 0.300},           # XLF → DBC
    "STAGFLATION": {"DBC": 0.400, "CASH": 0.600},                         # inchangé
    "DEFLATION":   {"XLY": 0.400, "XLC": 0.300, "CASH": 0.300},          # XLRE retiré, cash ajouté
}

ALLOC_EU = {
    "GOLDILOCKS":  {"EXV4.DE": 0.333, "EXV3.DE": 0.333, "EXI5.DE": 0.334},   # inchangé
    "REFLATION":   {"EXV6.DE": 0.400, "EXV1.DE": 0.333, "DBC":     0.267},   # EXV6 ↑
    "STAGFLATION": {"DBC": 0.333, "CASH": 0.667},                              # inchangé
    "DEFLATION":   {"EXV3.DE": 0.400, "EXV8.DE": 0.333, "EXV7.DE": 0.267},   # EXV4/EXV2 → EXV3/EXV8
}

# =============================================================================
# 3. PERIODS
# =============================================================================

PERIODS = {
    "P1 (2005-2012)":    ("2005-01-01", "2012-12-31"),
    "P2 (2013-2019)":    ("2013-01-01", "2019-12-31"),
    "P3 (2020-2022)":    ("2020-01-01", "2022-12-31"),
    "P4 (2023-Present)": ("2023-01-01", "2026-12-31"),
}

# =============================================================================
# 4. DATA DOWNLOAD
# =============================================================================

def download_data():
    tickers = {GLD_TICKER, US_INDEX, EU_INDEX, "DBC"}
    tickers.update(TICKERS_US.values())
    tickers.update(TICKERS_EU.values())
    data = yf.download(list(tickers), start="2004-01-01")["Close"]
    data = data.ffill().bfill()
    return data

# =============================================================================
# 5. SIGNAL HELPERS
# =============================================================================

def _apply_persistence_filter(raw_signal: pd.Series, min_days: int) -> pd.Series:
    """
    Filtre de confirmation : un changement de régime n'est validé que si le
    nouveau signal est stable pendant `min_days` jours consécutifs.

    Fonctionnement :
      - On mémorise l'état confirmé courant (`current`).
      - On compte les jours consécutifs sur le même état candidat (`pending`).
      - Quand le compteur atteint min_days, l'état candidat devient confirmé.
      - Tout signal différent du candidat en cours remet le compteur à 1.
    """
    filtered    = raw_signal.copy()
    current     = raw_signal.iloc[0]
    pending     = None
    pending_cnt = 0

    for i in range(len(raw_signal)):
        r = raw_signal.iloc[i]

        if r == current:
            pending     = None
            pending_cnt = 0
        else:
            if r == pending:
                pending_cnt += 1
                if pending_cnt >= min_days:
                    current     = pending
                    pending     = None
                    pending_cnt = 0
            else:
                pending     = r
                pending_cnt = 1

        filtered.iloc[i] = current

    return filtered


def _compute_asymmetric_buffer(series: pd.Series, window: int) -> pd.Series:
    """
    Buffer adaptatif = 0.5 × écart-type glissant sur `window` jours,
    exprimé en proportion de la série (ratio vol/niveau).

    En période calme    → buffer petit  → signal plus réactif.
    En période volatile → buffer grand  → résistance aux faux croisements.

    Plancher à 0.005 (0.5%) pour éviter un buffer nul en début de série.
    """
    vol    = series.rolling(window).std()
    buffer = (0.5 * vol / series.abs()).fillna(0.01).clip(lower=0.005)
    return buffer

# =============================================================================
# 6. SIGNAL CALCULATION  — avec les 3 améliorations
# =============================================================================

def calculate_signals(data: pd.DataFrame, index_ticker: str, zone: str) -> dict:
    """
    Calcule les signaux Growth et Inflation, puis le régime combiné.

    Améliorations v2 vs v1 :
      1. Growth signal   : médiane200 remplace MA200 (robustesse aux outliers)
      2. Buffer adaptatif: ±0.5*vol/niveau remplace ±1% fixe sur les deux signaux
      3. Filtre 15 jours : appliqué sur chaque sous-signal brut avant combinaison
    """

    # ------------------------------------------------------------------
    # 6.1  GROWTH SIGNAL
    # ------------------------------------------------------------------
    price = data[index_ticker]

    # v2 : médiane200 — plus robuste aux krachs brefs que la MA200
    med200_growth = price.rolling(SIGNAL_WINDOW).median()

    # v2 : buffer asymétrique sur le prix
    buf_growth = _compute_asymmetric_buffer(price, BUFFER_ASYMM_WINDOW)

    growth_raw = pd.Series("UNKNOWN", index=data.index, dtype="string")
    curr = "UNKNOWN"
    for i in range(len(data)):
        p, m, b = price.iloc[i], med200_growth.iloc[i], buf_growth.iloc[i]
        if not pd.isna(m) and not pd.isna(b):
            if   p > (1 + b) * m: curr = "UP"
            elif p < (1 - b) * m: curr = "DOWN"
        growth_raw.iloc[i] = curr

    # v2 : filtre de persistance 15 jours
    growth_signal = _apply_persistence_filter(growth_raw, CONFIRM_DAYS)

    # ------------------------------------------------------------------
    # 6.2  INFLATION SIGNAL
    # ------------------------------------------------------------------
    rets  = data.pct_change().fillna(0)
    t_map = TICKERS_US if zone == "US" else TICKERS_EU

    pos_ret = sum(
        rets[t_map[k]] * BETA_POS_WEIGHTS[k]
        for k in BETA_POS_WEIGHTS if t_map[k] in rets.columns
    )
    neg_ret = sum(
        rets[t_map[k]] * BETA_NEG_WEIGHTS[k]
        for k in BETA_NEG_WEIGHTS if t_map[k] in rets.columns
    )

    ratio = (1 + pos_ret).cumprod() / (1 + neg_ret).cumprod()

    # v2 : médiane200 sur le ratio (cohérence avec growth)
    med200_infl = ratio.rolling(SIGNAL_WINDOW).median()

    # v2 : buffer asymétrique sur le ratio
    buf_infl = _compute_asymmetric_buffer(ratio, BUFFER_ASYMM_WINDOW)

    inflation_raw = pd.Series("UNKNOWN", index=data.index, dtype="string")
    curr = "UNKNOWN"
    for i in range(len(data)):
        r, m, b = ratio.iloc[i], med200_infl.iloc[i], buf_infl.iloc[i]
        if not pd.isna(m) and not pd.isna(b):
            if   r > (1 + b) * m: curr = "UP"
            elif r < (1 - b) * m: curr = "DOWN"
        inflation_raw.iloc[i] = curr

    # v2 : filtre de persistance 15 jours
    inflation_signal = _apply_persistence_filter(inflation_raw, CONFIRM_DAYS)

    # ------------------------------------------------------------------
    # 6.3  REGIME COMBINÉ  (matrice 2×2 inchangée)
    # ------------------------------------------------------------------
    regime = pd.Series("UNKNOWN", index=data.index, dtype="string")
    regime[(growth_signal == "UP")   & (inflation_signal == "DOWN")] = "GOLDILOCKS"
    regime[(growth_signal == "UP")   & (inflation_signal == "UP")]   = "REFLATION"
    regime[(growth_signal == "DOWN") & (inflation_signal == "UP")]   = "STAGFLATION"
    regime[(growth_signal == "DOWN") & (inflation_signal == "DOWN")] = "DEFLATION"

    return {
        "regime":        regime,
        "growth":        growth_signal,
        "growth_raw":    growth_raw,        # diagnostic : avant filtre
        "inflation":     inflation_signal,
        "inflation_raw": inflation_raw,     # diagnostic : avant filtre
        "med200_growth": med200_growth,     # v2 : remplace ma200
        "buf_growth":    buf_growth,        # v2 : buffer adaptatif
        "med200_infl":   med200_infl,       # v2 : médiane ratio
        "buf_infl":      buf_infl,          # v2 : buffer adaptatif ratio
        "ratio":         ratio,
        "price":         price,
    }

# =============================================================================
# 7. BACKTEST  (inchangé vs v1)
# =============================================================================

def run_backtest(data: pd.DataFrame, zone_name: str,
                 alloc_matrix: dict, regime_series: pd.Series) -> pd.Series:
    rets         = data.pct_change().fillna(0)
    rets["CASH"] = 0.0

    regime_delayed = regime_series.shift(1).fillna("UNKNOWN")
    reb_dates      = data.resample("ME").last().index
    val, current_w = 100.0, {}
    history        = []

    for date in data.index:
        if current_w:
            day_ret = sum(
                current_w.get(t, 0) * rets.loc[date, t]
                for t in current_w
                if t in rets.columns or t == "CASH"
            )
            val *= (1 + day_ret)

        if date in reb_dates:
            reg      = regime_delayed.loc[date]
            target_w = {GLD_TICKER: 0.10}

            if reg in alloc_matrix:
                for t, w in alloc_matrix[reg].items():
                    target_w[t] = target_w.get(t, 0) + w * 0.90
            else:
                target_w["CASH"] = target_w.get("CASH", 0) + 0.90

            turnover = sum(
                abs(target_w.get(t, 0) - current_w.get(t, 0))
                for t in set(target_w) | set(current_w)
            )
            val      *= (1 - turnover * 0.0010)
            current_w = target_w.copy()

        history.append(val)

    return pd.Series(history, index=data.index)

# =============================================================================
# 8. STATISTICS  — alpha ajouté vs v1
# =============================================================================

def calculate_stats(strat_val: pd.Series, bench_val: pd.Series) -> dict:
    def get_metrics(v):
        r        = v.pct_change().dropna()
        cagr     = (v.iloc[-1] / v.iloc[0]) ** (252 / len(v)) - 1
        vol      = r.std() * np.sqrt(252)
        mdd      = ((v - v.cummax()) / v.cummax()).min()
        sharpe   = cagr / vol if vol != 0 else 0
        hit_rate = (r > 0).mean()
        return cagr, vol, mdd, sharpe, hit_rate

    s_cagr, s_vol, s_mdd, s_sharpe, s_hit = get_metrics(strat_val)
    b_cagr, *_                             = get_metrics(bench_val)
    return {
        "CAGR": s_cagr, "Vol": s_vol, "MaxDD": s_mdd,
        "Sharpe": s_sharpe, "HitRate": s_hit,
        "Alpha": s_cagr - b_cagr,
    }

# =============================================================================
# 9. MAIN
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Téléchargement des données...")
    data = download_data()

    print("Calcul des signaux v2 (médiane200 / buffer asymétrique / filtre 15j)...")
    res_us = calculate_signals(data, US_INDEX, "US")
    res_eu = calculate_signals(data, EU_INDEX, "EU")

    print("Backtests...")
    v_us = run_backtest(data, "US", ALLOC_US, res_us["regime"])
    v_eu = run_backtest(data, "EU", ALLOC_EU, res_eu["regime"])

    bench_us = (1 + data[US_INDEX].pct_change().fillna(0)).cumprod() * 100
    bench_eu = (1 + data[EU_INDEX].pct_change().fillna(0)).cumprod() * 100

    # ------------------------------------------------------------------
    # 9.1  backtest_results.csv
    # ------------------------------------------------------------------
    pd.DataFrame({
        "Strategy_US":  v_us,  "Benchmark_US": bench_us, "Regime_US": res_us["regime"],
        "Strategy_EU":  v_eu,  "Benchmark_EU": bench_eu, "Regime_EU": res_eu["regime"],
    }).to_csv("backtest_results.csv")

    # ------------------------------------------------------------------
    # 9.2  signals_analysis.csv  — colonnes enrichies v2
    # ------------------------------------------------------------------
    pd.DataFrame({
        "Price_US":       res_us["price"],        "Med200_US":      res_us["med200_growth"],
        "BufGrowth_US":   res_us["buf_growth"],   "Growth_Raw_US":  res_us["growth_raw"],
        "Growth_Filt_US": res_us["growth"],       "Ratio_US":       res_us["ratio"],
        "Med200Ratio_US": res_us["med200_infl"],  "BufInfl_US":     res_us["buf_infl"],
        "Infl_Raw_US":    res_us["inflation_raw"],"Infl_Filt_US":   res_us["inflation"],
        "Price_EU":       res_eu["price"],        "Med200_EU":      res_eu["med200_growth"],
        "BufGrowth_EU":   res_eu["buf_growth"],   "Growth_Raw_EU":  res_eu["growth_raw"],
        "Growth_Filt_EU": res_eu["growth"],       "Ratio_EU":       res_eu["ratio"],
        "Med200Ratio_EU": res_eu["med200_infl"],  "BufInfl_EU":     res_eu["buf_infl"],
        "Infl_Raw_EU":    res_eu["inflation_raw"],"Infl_Filt_EU":   res_eu["inflation"],
    }).to_csv("signals_analysis.csv")

    # ------------------------------------------------------------------
    # 9.3  sector_performance.csv
    # ------------------------------------------------------------------
    all_rets     = data.pct_change().fillna(0)
    perf_records = []
    for zone, reg_series, tickers in [
        ("US", res_us["regime"], TICKERS_US.values()),
        ("EU", res_eu["regime"], TICKERS_EU.values()),
    ]:
        for reg in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]:
            mask = (reg_series == reg)
            if mask.any():
                for t in set(tickers):
                    if t in all_rets.columns:
                        ann_ret = all_rets.loc[mask, t].mean() * 252
                        perf_records.append({"Zone": zone, "Regime": reg,
                                             "Sector": t, "Ann_Return": ann_ret})
    pd.DataFrame(perf_records).to_csv("sector_performance.csv", index=False)

    # ------------------------------------------------------------------
    # 9.4  strategy_stats.csv
    # ------------------------------------------------------------------
    stats_us = calculate_stats(v_us, bench_us)
    stats_eu = calculate_stats(v_eu, bench_eu)
    pd.DataFrame([
        {"Zone": "US", **stats_us},
        {"Zone": "EU", **stats_eu},
    ]).to_csv("strategy_stats.csv", index=False)

    # ------------------------------------------------------------------
    # 9.5  Sector performance par période
    # ------------------------------------------------------------------
    all_periods = {"TOTAL": (str(data.index[0].date()), str(data.index[-1].date()))}
    for k, v in PERIODS.items():
        all_periods[k.split(" ")[0]] = v

    for p_key, (start, end) in all_periods.items():
        p_mask = (data.index >= start) & (data.index <= end)
        if not p_mask.any():
            continue
        p_perf = []
        for zone, reg_series, tickers in [
            ("US", res_us["regime"], TICKERS_US.values()),
            ("EU", res_eu["regime"], TICKERS_EU.values()),
        ]:
            for reg in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]:
                mask = p_mask & (reg_series == reg)
                if mask.any():
                    for t in set(tickers):
                        if t in all_rets.columns:
                            ann_ret = all_rets.loc[mask, t].mean() * 252
                            p_perf.append({"Zone": zone, "Regime": reg,
                                           "Sector": t, "Ann_Return": ann_ret})
        pd.DataFrame(p_perf).to_csv(f"sector_perf_{p_key}.csv", index=False)

    # ------------------------------------------------------------------
    # 9.6  Diagnostic : impact du filtre de persistance
    # ------------------------------------------------------------------
    def count_transitions(s):
        s = s[s != "UNKNOWN"]
        return int((s != s.shift()).sum())

    def build_raw_regime(g_raw, i_raw):
        cond = [
            (g_raw == "UP")   & (i_raw == "DOWN"),
            (g_raw == "UP")   & (i_raw == "UP"),
            (g_raw == "DOWN") & (i_raw == "UP"),
            (g_raw == "DOWN") & (i_raw == "DOWN"),
        ]
        choices = ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION"]
        return pd.Series(np.select(cond, choices, default="UNKNOWN"), index=g_raw.index)

    raw_regime_us = build_raw_regime(res_us["growth_raw"], res_us["inflation_raw"])
    raw_regime_eu = build_raw_regime(res_eu["growth_raw"], res_eu["inflation_raw"])

    diag = pd.DataFrame([
        {
            "Zone": "US",
            "Transitions_Raw":      count_transitions(raw_regime_us),
            "Transitions_Filtered": count_transitions(res_us["regime"]),
        },
        {
            "Zone": "EU",
            "Transitions_Raw":      count_transitions(raw_regime_eu),
            "Transitions_Filtered": count_transitions(res_eu["regime"]),
        },
    ])
    diag["Reduction_pct"] = (
        (diag["Transitions_Raw"] - diag["Transitions_Filtered"])
        / diag["Transitions_Raw"] * 100
    ).round(1)
    diag.to_csv("signal_diagnostics.csv", index=False)

    print("\n--- Diagnostic filtre de persistance ---")
    print(diag.to_string(index=False))

    # ------------------------------------------------------------------
    # 9.7  Plot comparatif
    # ------------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(v_us,     label="Stratégie US v2",      color="blue")
    plt.plot(bench_us, label="Benchmark US (S&P500)", color="lightblue",  linestyle="--")
    plt.plot(v_eu,     label="Stratégie EU v2",      color="green")
    plt.plot(bench_eu, label="Benchmark EU (STX50)",  color="lightgreen", linestyle="--")
    plt.yscale("log")
    plt.title("Stratégies v2 vs Benchmarks — médiane200, buffer asymétrique, filtre 15j")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("strategy_vs_benchmark.png")

    print("\nData Engine v2 — Exécution réussie.")
    print(f"\nStats US : CAGR={stats_us['CAGR']:.1%}  Sharpe={stats_us['Sharpe']:.2f}  "
          f"MaxDD={stats_us['MaxDD']:.1%}  Alpha={stats_us['Alpha']:.1%}")
    print(f"Stats EU : CAGR={stats_eu['CAGR']:.1%}  Sharpe={stats_eu['Sharpe']:.2f}  "
          f"MaxDD={stats_eu['MaxDD']:.1%}  Alpha={stats_eu['Alpha']:.1%}")
