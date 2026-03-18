import yfinance as yf
import pandas as pd

def check(t):
    try:
        d = yf.download(t, period="5d")
        return not d.empty
    except:
        return False

eu_sector_bases = ["SX8P", "SX3P", "SX7P", "SXEP", "SXPP", "SXDP", "SXAP", "SXIP", "SX86P", "SX6P", "SXKP"]
suffixes = ["", ".EX", ".DE", ".L", ".PA"]
prefixes = ["", "^"]

results = []
for base in eu_sector_bases:
    found = False
    for p in prefixes:
        for s in suffixes:
            ticker = f"{p}{base}{s}"
            if check(ticker):
                print(f"FOUND: {ticker}")
                results.append(ticker)
                found = True
                break
        if found: break
    if not found:
        print(f"NOT FOUND: {base}")

print("\nFinal List:", results)
