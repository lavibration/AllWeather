import yfinance as yf
import pandas as pd
import numpy as np

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
    return yf.download(list(all_t), start="2005-01-01")['Close'].ffill()

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
    return regime
