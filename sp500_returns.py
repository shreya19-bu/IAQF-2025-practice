import pandas as pd
import yfinance as yf
import numpy as np

def get_sp500_tickers():
    """
    Scrapes Wikipedia for the current list of S&P 500 constituents.
    Returns a list of tickers.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(url, header=0)[0]
        # The ticker symbol is in the 'Symbol' column
        tickers = table['Symbol'].tolist()
        # Wikipedia's table sometimes has incorrect ticker formats (e.g., 'BRK.B' vs 'BRK-B').
        # yfinance expects a hyphen for different classes of stock.
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        print(f"Successfully fetched {len(tickers)} S&P 500 tickers from Wikipedia.")
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        # Fallback to a predefined list if scraping fails
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ', 'XOM', 'UNH', 'PG', 'HD', 'CVX', 'MRK', 'KO', 'PEP', 'WMT', 'BAC', 'PFE', 'MCD', 'COST', 'DIS', 'CSCO', 'TMO', 'ABT', 'NEE', 'LIN', 'LLY']

def download_and_process_data(tickers, start_date, end_date):
    """
    Downloads adjusted closing prices, calculates returns, and groups stocks.

    Args:
        tickers (list): List of stock tickers.
        start_date (str): Start date for data download (YYYY-MM-DD).
        end_date (str): End date for data download (YYYY-MM-DD).

    Returns:
        pd.DataFrame: A DataFrame with Date, Ticker, Return, Group, and Close price.
    """
    mag_7 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']
    all_data_frames = []
    failed_tickers = []

    # Download all data in one go for efficiency
    print(f"Downloading daily data for {len(tickers)} tickers from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, group_by='ticker')

    print("Processing data and calculating returns...")
    for ticker in tickers:
        try:
            # yfinance multi-ticker download results in a multi-level column index
            if len(tickers) > 1:
                stock_data = data[ticker]
            else: # If only one ticker, the structure is different
                stock_data = data

            if stock_data.empty or stock_data['Close'].isnull().all():
                raise ValueError("No data found or all 'Close' values are NaN.")

            # Calculate daily total returns from adjusted closing prices
            stock_data['Return'] = stock_data['Close'].pct_change()
            stock_data['Ticker'] = ticker
            stock_data['Group'] = 'Mag7' if ticker in mag_7 else 'Other'
            
            all_data_frames.append(stock_data)

        except Exception as e:
            failed_tickers.append(ticker)
            print(f"Could not process data for {ticker}. Reason: {e}")

    if not all_data_frames:
        print("No data could be processed. Exiting.")
        return pd.DataFrame()

    # Combine all individual DataFrames
    final_df = pd.concat(all_data_frames)

    # Clean up the DataFrame
    final_df.reset_index(inplace=True)
    final_df.dropna(subset=['Return', 'Close'], inplace=True)
    final_df = final_df[['Date', 'Ticker', 'Group', 'Close', 'Return']]
    final_df.sort_values(by=['Date', 'Ticker'], inplace=True)

    if failed_tickers:
        print("\nFailed to download or process the following tickers (may be delisted or have changed symbols):")
        print(", ".join(failed_tickers))
        
    return final_df

def main():
    """
    Main function to execute the script.
    """
    # --- Configuration ---
    START_DATE = '2018-01-01'
    END_DATE = '2024-12-31'
    OUTPUT_CSV = '..\\sp500_daily_data.csv'

    # --- Execution ---
    sp500_tickers = get_sp500_tickers()
    
    # Ensure Mag7 are included, as Wikipedia list might be out of date
    mag_7_to_add = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']
    for ticker in mag_7_to_add:
        if ticker not in sp500_tickers:
            sp500_tickers.append(ticker)

    returns_df = download_and_process_data(sp500_tickers, START_DATE, END_DATE)

    if not returns_df.empty:
        # --- Output ---
        print(f"\nSuccessfully processed {returns_df['Ticker'].nunique()} tickers.")
        print("Sample of the final DataFrame:")
        print(returns_df.head())
        print(f"\nDataFrame shape: {returns_df.shape}")

        # Save to CSV
        returns_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nData saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
