import yfinance as yf
tickers = ["EXV11.DE", "EXW1.DE", "IINS.L", "EXW2.DE", "SXIP.EX"]
for t in tickers:
    d = yf.download(t, period="5d")
    if not d.empty: print(f"FOUND: {t}")
