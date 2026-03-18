import yfinance as yf
import pandas as pd

us_tickers = ["XLK", "XLY", "XLF", "XLE", "XLB", "XLV", "XLI", "XLU", "XLRE", "XLC", "DBC", "^GSPC"]
eu_tickers = ["SX8P.EX", "SX3P.EX", "SX7P.EX", "SXEP.EX", "SXPP.EX", "SXDP.EX", "SXAP.EX", "SXIP.EX", "SX86P.EX", "SX6P.EX", "SXKP.EX", "^STOXX50E"]

def check_tickers(tickers):
    for t in tickers:
        d = yf.download(t, start="2005-01-01", end="2006-01-01")
        print(f"{t}: {len(d)} rows")

if __name__ == "__main__":
    print("Checking US...")
    check_tickers(us_tickers)
    print("\nChecking EU...")
    check_tickers(eu_tickers)
