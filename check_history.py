import yfinance as yf
import pandas as pd

TICKERS = [
    "^GSPC", "^STOXX50E", "GLD", "XLK", "XLY", "XLF", "XLE", "XLB", "XLRE", "XLC", "XLU", "DBC",
    "EXV5.DE", "EXV1.DE", "EXV6.DE", "SXRS.DE", "EXV3.DE", "EXV2.DE", "EXV9.DE", "EXI5.DE", "EXV4.DE", "EXV7.DE"
]

def check_data():
    results = []
    for t in TICKERS:
        try:
            d = yf.download(t, start="2005-01-01")
            if not d.empty:
                results.append({
                    "Ticker": t,
                    "Start": d.index.min().strftime('%Y-%m-%d'),
                    "End": d.index.max().strftime('%Y-%m-%d'),
                    "Count": len(d)
                })
            else:
                results.append({"Ticker": t, "Start": "N/A", "End": "N/A", "Count": 0})
        except Exception as e:
            results.append({"Ticker": t, "Error": str(e)})

    df = pd.DataFrame(results)
    print(df.to_string())

if __name__ == "__main__":
    check_data()
