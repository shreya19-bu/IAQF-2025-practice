import pandas as pd
import numpy as np
import statsmodels.api as sm
import random

# --- Configuration ---
DATA_CSV = 'sp500_daily_data.csv' 
SHOCK_STOCK = 'NVDA'
SHOCK_PERCENTAGE = -0.20
NUM_SAMPLES = 10
ROLLING_WINDOW = 126  # Approx. 6 months of trading days

def load_data(data_csv):
    """Loads and prepares the returns and price data."""
    try:
        df = pd.read_csv(data_csv, parse_dates=['Date'])
        print(f"Successfully loaded '{data_csv}'.")
        # Ensure we have the necessary columns
        if not {'Date', 'Ticker', 'Group', 'Close', 'Return'}.issubset(df.columns):
            print("Error: The CSV file is missing required columns.")
            print("Please ensure it contains: Date, Ticker, Group, Close, Return")
            return None
        return df
    except FileNotFoundError:
        print(f"Error: The file '{data_csv}' was not found.")
        print("Please run the 'sp500_returns.py' script to generate the data file.")
        return None

def calculate_value_weighted_market_return(df):
    """
    Calculates the daily value-weighted market return.
    Uses 'Close' price as a proxy for market capitalization.
    """
    # Calculate total market 'cap' for each day
    daily_market_cap = df.groupby('Date')['Close'].transform('sum')
    
    # Calculate each stock's weight for each day
    df['Weight'] = df['Close'] / daily_market_cap
    
    # Calculate the weighted return for each stock
    df['Weighted_Return'] = df['Weight'] * df['Return']
    
    # Sum the weighted returns for each day to get the market return
    market_return = df.groupby('Date')['Weighted_Return'].sum()
    
    return market_return

def calculate_beta(stock_returns, market_returns):
    """Calculates CAPM beta for a single stock."""
    X = sm.add_constant(market_returns, prepend=False)
    y = stock_returns
    model = sm.OLS(y, X, missing='drop').fit()
    return model.params.iloc[0] if len(model.params) > 1 else np.nan

def run_simulation(full_df):
    """
    Runs the full shock simulation and analysis.
    """
    if full_df is None:
        return

    # --- 1. Setup the analysis window and data ---
    end_date = full_df['Date'].max()
    start_date = end_date - pd.DateOffset(days=ROLLING_WINDOW * 1.8) # Extend buffer for non-trading days
    
    analysis_df = full_df[(full_df['Date'] >= start_date) & (full_df['Date'] <= end_date)].copy()
    
    # Ensure we have exactly ROLLING_WINDOW days of data for consistent beta calcs
    valid_dates = sorted(analysis_df['Date'].unique())
    if len(valid_dates) < ROLLING_WINDOW:
        print(f"Warning: Not enough data for a full {ROLLING_WINDOW}-day window. Found {len(valid_dates)} days.")
    else:
        final_start_date = valid_dates[-ROLLING_WINDOW]
        analysis_df = analysis_df[analysis_df['Date'] >= final_start_date]

    shock_date = analysis_df['Date'].max()

    # Get a sample of non-Mag7 stocks
    non_mag7_stocks = analysis_df[analysis_df['Group'] == 'Other']['Ticker'].unique()
    sample_tickers = random.sample(list(non_mag7_stocks), min(len(non_mag7_stocks), NUM_SAMPLES))

    print(f"\n--- Shock Simulation Scenario ---")
    print(f"Shock Stock: {SHOCK_STOCK}")
    print(f"Shock: {SHOCK_PERCENTAGE:.0%} return on {shock_date.date()}")
    print(f"Analysis Window: {analysis_df['Date'].min().date()} to {shock_date.date()} ({analysis_df['Date'].nunique()} days)")
    print(f"Sample Stocks: {', '.join(sample_tickers)}")

    # --- 2. Pre-Shock Analysis ---
    print("\nCalculating pre-shock betas using value-weighted market...")
    pre_shock_market_returns = calculate_value_weighted_market_return(analysis_df)
    
    pre_shock_betas = {}
    for ticker in sample_tickers:
        stock_returns = analysis_df[analysis_df['Ticker'] == ticker].set_index('Date')['Return']
        # Align data for regression
        aligned_stock, aligned_market = stock_returns.align(pre_shock_market_returns, join='inner')
        pre_shock_betas[ticker] = calculate_beta(aligned_stock, aligned_market)

    # --- 3. Apply Shock and Post-Shock Analysis ---
    print("Applying shock and calculating post-shock betas...")
    post_shock_df = analysis_df.copy()

    # Identify the row to shock
    shock_idx = post_shock_df[(post_shock_df['Ticker'] == SHOCK_STOCK) & (post_shock_df['Date'] == shock_date)].index

    if not shock_idx.empty:
        # Get pre-shock price to calculate post-shock price
        pre_shock_price = post_shock_df.loc[shock_idx, 'Close'].iloc[0]
        # Update Return and Close price
        post_shock_df.loc[shock_idx, 'Return'] = SHOCK_PERCENTAGE
        post_shock_df.loc[shock_idx, 'Close'] = pre_shock_price * (1 + SHOCK_PERCENTAGE)
    else:
        print(f"Warning: {SHOCK_STOCK} not found on {shock_date.date()}. No shock applied.")

    post_shock_market_returns = calculate_value_weighted_market_return(post_shock_df)
    
    post_shock_betas = {}
    for ticker in sample_tickers:
        stock_returns = post_shock_df[post_shock_df['Ticker'] == ticker].set_index('Date')['Return']
        aligned_stock, aligned_market = stock_returns.align(post_shock_market_returns, join='inner')
        post_shock_betas[ticker] = calculate_beta(aligned_stock, aligned_market)
        
    # --- 4. Compare and Report ---
    comparison_data = [{
        'Ticker': ticker,
        'Pre-Shock Beta': pre_shock_betas.get(ticker),
        'Post-Shock Beta': post_shock_betas.get(ticker),
        'Beta Shift': post_shock_betas.get(ticker) - pre_shock_betas.get(ticker)
    } for ticker in sample_tickers]
        
    comparison_df = pd.DataFrame(comparison_data).set_index('Ticker')
    
    print("\n--- Value-Weighted Beta Impact Analysis ---")
    print(comparison_df.to_string(formatters={'Pre-Shock Beta': '{:,.4f}'.format, 'Post-Shock Beta': '{:,.4f}'.format, 'Beta Shift': '{:+.4f}'.format}))
    
    avg_shift = comparison_df['Beta Shift'].mean()
    
    print("\n--- Summary ---")
    print(f"Average Beta Shift for non-Mag7 stocks: {avg_shift:+.4f}")
    print("\nInterpretation: The shock to a large-cap stock like NVDA significantly alters the 'market' return for that day.")
    print("This makes other, un-shocked stocks appear to move against the market, artificially depressing their calculated beta.")
    print("This demonstrates how concentration in major indices can distort systematic risk (beta) measurements during single-stock shocks.")

if __name__ == "__main__":
    full_data = load_data(DATA_CSV)
    run_simulation(full_data)
