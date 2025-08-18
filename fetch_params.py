import yfinance as yf
import numpy as np
import sys

def get_market_data(ticker_symbol):
    """
    Fetches market data and prints it to stdout in a comma-separated format.
    Format: CurrentPrice,RiskFreeRate,Volatility
    """
    # Fetch historical data for the last year to calculate volatility
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="1y")

    if hist.empty:
        # If no data is found, raise an error
        raise ValueError(f"Could not fetch historical data for {ticker_symbol}. Check the ticker.")

    # Calculate historical volatility (annualized standard deviation of log returns)
    log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
    sigma = log_returns.std() * np.sqrt(252) # 252 trading days in a year

    # Get the most recent closing price for S
    S = hist['Close'].iloc[-1]

    # Fetch 10-Year Treasury yield as a proxy for the risk-free rate
    treasury_ticker = yf.Ticker("^TNX")
    tnx_hist = treasury_ticker.history(period="5d") # Get recent data
    
    if tnx_hist.empty:
        # If treasury data fails, use a reasonable default and warn the user via stderr
        print("Warning: Could not fetch 10-Year Treasury yield. Defaulting r to 5%.", file=sys.stderr)
        r = 0.05
    else:
        # Convert from percentage to decimal
        r = tnx_hist['Close'].iloc[-1] / 100.0

    # Print to standard output for the C++ application to capture
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