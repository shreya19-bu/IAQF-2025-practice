import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

def analyze_tracking_error():
    """
    Loads reconstructed S&P 500 returns, compares them to the official ^GSPC index,
    calculates tracking error, and plots the cumulative returns.
    """
    # --- Configuration ---
    CONSTITUENT_RETURNS_CSV = 'sp500_daily_returns.csv'
    START_DATE = '2018-01-01'
    END_DATE = '2024-12-31'
    BENCHMARK_TICKER = '^GSPC'

    # --- 1. Load and Process Reconstructed Returns ---
    try:
        constituent_returns_df = pd.read_csv(CONSTITUENT_RETURNS_CSV, parse_dates=['Date'])
        print(f"Successfully loaded '{CONSTITUENT_RETURNS_CSV}'.")
    except FileNotFoundError:
        print(f"Error: The file '{CONSTITUENT_RETURNS_CSV}' was not found.")
        print("Please run the 'sp500_returns.py' script first to generate the data.")
        return

    # Calculate the equal-weighted daily return for the reconstructed portfolio
    # This assumes the returns in the CSV are daily returns for each constituent
    reconstructed_returns = constituent_returns_df.groupby('Date')['Return'].mean()

    # --- 2. Download Benchmark (^GSPC) Returns ---
    print(f"Downloading benchmark data for {BENCHMARK_TICKER}...")
    gspc_data = yf.download(BENCHMARK_TICKER, start=START_DATE, end=END_DATE, auto_adjust=True)
    
    if gspc_data.empty:
        print(f"Error: Could not download data for benchmark ticker {BENCHMARK_TICKER}.")
        return

    gspc_returns = gspc_data['Close'].pct_change()

    # --- 3. Align Data and Calculate Tracking Error ---
    # Create DataFrames for reconstructed and benchmark returns, then merge them.
    recon_df = reconstructed_returns.to_frame(name='Reconstructed')
    
    gspc_df = pd.DataFrame()
    gspc_df['GSPC'] = gspc_returns

    comparison_df = pd.merge(recon_df, gspc_df, left_index=True, right_index=True).dropna()

    # Calculate the difference in daily returns
    comparison_df['Difference'] = comparison_df['Reconstructed'] - comparison_df['GSPC']

    # Calculate tracking error (standard deviation of the difference in returns)
    daily_tracking_error = comparison_df['Difference'].std()
    annual_tracking_error = daily_tracking_error * np.sqrt(252) # Annualize

    print("\n--- Tracking Error Analysis ---")
    print(f"Daily Tracking Error: {daily_tracking_error:.4%}")
    print(f"Annualized Tracking Error: {annual_tracking_error:.4%}")
    print("\nNote: A non-zero tracking error is expected because this analysis compares an")
    print("equal-weighted reconstructed portfolio with the market-cap-weighted ^GSPC index.")

    # --- 4. Plot Cumulative Returns ---
    # Calculate cumulative returns for both series
    comparison_df['Cumulative_Reconstructed'] = (1 + comparison_df['Reconstructed']).cumprod() - 1
    comparison_df['Cumulative_GSPC'] = (1 + comparison_df['GSPC']).cumprod() - 1

    # Plotting
    style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(14, 7))

    plt.plot(comparison_df.index, comparison_df['Cumulative_Reconstructed'], label='Reconstructed (Equal-Weighted) S&P 500')
    plt.plot(comparison_df.index, comparison_df['Cumulative_GSPC'], label='Official ^GSPC (Market-Cap Weighted)')

    plt.title('Reconstructed S&P 500 vs. Official ^GSPC Index (Cumulative Returns)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Format the y-axis as a percentage
    vals = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    
    print("\nDisplaying plot. Close the plot window to finish the script.")
    plt.show()

if __name__ == '__main__':
    analyze_tracking_error()
