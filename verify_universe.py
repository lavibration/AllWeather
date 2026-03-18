import yfinance as yf
import pandas as pd

us_map = {
    "Tech": "XLK", "Conso": "XLY", "Finance": "XLF", "Énergie": "XLE",
    "Matériaux": "XLB", "Santé": "XLV", "Industrie": "XLI", "Utilities": "XLU",
    "Immo": "XLRE", "Telecoms": "XLC", "Commodities": "DBC"
}

eu_map = {
    "Tech": "EXV3.DE", "Conso": "EXV8.DE", "Banques": "EXV1.DE", "Énergie": "EXV5.DE",
    "Ressources": "EXV6.DE", "Santé": "EXV4.DE", "Auto": "EXV2.DE", "Assurance": "EXV11.DE",
    "Immo": "EXI5.DE", "Utilities": "EXV9.DE", "Telecoms": "EXV7.DE"
}

def check(name, ticker):
    try:
        d = yf.download(ticker, start="2005-01-01")
        if d.empty: return {"Name": name, "Ticker": ticker, "Start": "N/A", "Rows": 0}
        return {"Name": name, "Ticker": ticker, "Start": d.index.min().strftime('%Y-%m-%d'), "Rows": len(d)}
    except:
        return {"Name": name, "Ticker": ticker, "Start": "Error", "Rows": 0}

print("Checking US...")
us_res = [check(k, v) for k, v in us_map.items()]
print(pd.DataFrame(us_res))

print("\nChecking EU...")
eu_res = [check(k, v) for k, v in eu_map.items()]
print(pd.DataFrame(eu_res))

print("\nChecking Indices...")
indices = {"S&P 500": "^GSPC", "Euro Stoxx 50": "^STOXX50E", "STOXX 600": "^STOXX"}
idx_res = [check(k, v) for k, v in indices.items()]
print(pd.DataFrame(idx_res))
