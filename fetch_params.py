import yfinance as yf
import numpy as np
import sys

def get_market_data(ticker_symbol):
 
    # fetch historical data for the last year to calculate volatility
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="1y")

    if hist.empty:
        raise ValueError(f"Could not fetch historical data for {ticker_symbol}. Check the ticker.")

    
    log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
    sigma = log_returns.std() * np.sqrt(252) 

    S = hist['Close'].iloc[-1]

    treasury_ticker = yf.Ticker("^TNX")
    tnx_hist = treasury_ticker.history(period="5d") # recent data
    
    if tnx_hist.empty:
        r = 0.05 # default r value
    else:
        r = tnx_hist['Close'].iloc[-1] / 100.0

    print(f"{S},{r},{sigma}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_params.py <TICKER>", file=sys.stderr)
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    try:
        get_market_data(ticker)
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}", file=sys.stderr)
        sys.exit(1)
