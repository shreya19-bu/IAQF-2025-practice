import pandas as pd
import numpy as np
import statsmodels.api as sm
import random

# --- Configuration ---
RETURNS_CSV = 'sp500_daily_returns.csv'
BETA_CONTRIBUTIONS_CSV = 'beta_contribution_by_group.csv'
SHOCK_STOCK = 'NVDA'
SHOCK_PERCENTAGE = -0.20
NUM_SAMPLES = 10
ROLLING_WINDOW = 126  # Approx. 6 months of trading days

def load_data(returns_csv):
    """Loads and prepares the returns data."""
    try:
        df = pd.read_csv(returns_csv, parse_dates=['Date'])
        print(f"Successfully loaded '{returns_csv}'.")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{returns_csv}' was not found.")
        print("Please run the precursor scripts to generate this file.")
        return None

def calculate_beta(stock_returns, market_returns):
    """Calculates CAPM beta for a single stock."""
    # Add a constant (intercept) to the independent variable (market returns)
    X = sm.add_constant(market_returns)
    y = stock_returns
    
    # Fit the OLS model
    model = sm.OLS(y, X, missing='drop').fit()
    
    # The beta is the coefficient of the market return
    if model.params.shape[0] > 1:
        return model.params.iloc[1]
    return np.nan


def simulate_shock_scenario(returns_df):
    """
    Simulates a shock on a single stock and analyzes the impact on other stocks' betas.
    """
    if returns_df is None:
        return

    # --- 1. Setup the scenario ---
    # Use a fixed date to ensure data completeness for all tickers in the sample
    shock_date = pd.to_datetime('2023-12-29')
    if shock_date > returns_df['Date'].max() or shock_date < returns_df['Date'].min():
        print(f"Error: Hardcoded shock_date {shock_date.date()} is outside the available data range.")
        # Fallback to the latest date if the hardcoded date is bad.
        shock_date = returns_df['Date'].max()
        
    # Define the 6-month analysis period ending on the shock date
    start_date = shock_date - pd.Timedelta(days=ROLLING_WINDOW * 1.7) # Estimate to get enough trading days
    analysis_period_df = returns_df[(returns_df['Date'] >= start_date) & (returns_df['Date'] <= shock_date)].copy()
    
    # Take the most recent ROLLING_WINDOW days from the filtered data
    valid_dates = analysis_period_df['Date'].unique()
    if len(valid_dates) < ROLLING_WINDOW:
        print(f"Warning: Not enough data for a full {ROLLING_WINDOW}-day window. Using {len(valid_dates)} days.")
        # Adjust start date to include all available data if less than window
        start_date = valid_dates[0]
    else:
        start_date = valid_dates[-ROLLING_WINDOW]
    
    analysis_period_df = analysis_period_df[analysis_period_df['Date'] >= start_date]

    # Get a sample of 10 non-Mag7 stocks
    non_mag7_stocks = analysis_period_df[analysis_period_df['Group'] == 'Other']['Ticker'].unique()
    if len(non_mag7_stocks) < NUM_SAMPLES:
        print(f"Warning: Found only {len(non_mag7_stocks)} non-Mag7 stocks, which is less than the desired {NUM_SAMPLES} samples. Using all available.")
        sample_tickers = list(non_mag7_stocks) # Use all available stocks
        if not sample_tickers:
            print("Error: No non-Mag7 stocks found in the analysis window. Cannot proceed.")
            return
    else:
        sample_tickers = random.sample(list(non_mag7_stocks), NUM_SAMPLES)
    
    print(f"\n--- Shock Simulation Scenario ---")
    print(f"Shock Stock: {SHOCK_STOCK}")
    print(f"Shock: {SHOCK_PERCENTAGE:.0%} return on {shock_date.date()}")
    print(f"Analysis Window: {start_date.date()} to {shock_date.date()} ({ROLLING_WINDOW} days)")
    print(f"Sample Non-Mag7 Stocks: {', '.join(sample_tickers)}")
    
    # --- 2. Pre-Shock Analysis ---
    print("\nCalculating pre-shock betas...")
    # NOTE: Using equal-weighted market return for consistency with previous scripts.
    # A true value-weighted return would require daily market cap data.
    pre_shock_market_returns = analysis_period_df.groupby('Date')['Return'].mean()
    
    pre_shock_betas = {}
    for ticker in sample_tickers:
        stock_returns = analysis_period_df[analysis_period_df['Ticker'] == ticker].set_index('Date')['Return']
        beta = calculate_beta(stock_returns, pre_shock_market_returns)
        pre_shock_betas[ticker] = beta

    # --- 3. Post-Shock Analysis ---
    print("Applying shock and calculating post-shock betas...")
    post_shock_df = analysis_period_df.copy()
    
    # Apply the shock
    shock_stock_idx = post_shock_df[(post_shock_df['Ticker'] == SHOCK_STOCK) & (post_shock_df['Date'] == shock_date)].index
    if not shock_stock_idx.empty:
        post_shock_df.loc[shock_stock_idx, 'Return'] = SHOCK_PERCENTAGE
    else:
        print(f"Warning: {SHOCK_STOCK} not found on {shock_date.date()}. No shock applied.")

    # Recalculate market return with the shocked value
    post_shock_market_returns = post_shock_df.groupby('Date')['Return'].mean()
    
    post_shock_betas = {}
    for ticker in sample_tickers:
        stock_returns = post_shock_df[post_shock_df['Ticker'] == ticker].set_index('Date')['Return']
        beta = calculate_beta(stock_returns, post_shock_market_returns)
        post_shock_betas[ticker] = beta
        
    # --- 4. Compare and Report ---
    comparison_data = []
    for ticker in sample_tickers:
        beta_pre = pre_shock_betas.get(ticker, np.nan)
        beta_post = post_shock_betas.get(ticker, np.nan)
        beta_shift = beta_post - beta_pre
        comparison_data.append({
            'Ticker': ticker,
            'Pre-Shock Beta': beta_pre,
            'Post-Shock Beta': beta_post,
            'Beta Shift': beta_shift
        })
        
    comparison_df = pd.DataFrame(comparison_data).set_index('Ticker')
    
    print("\n--- Beta Impact Analysis ---")
    print(comparison_df)
    
    avg_shift = comparison_df['Beta Shift'].mean()
    median_shift = comparison_df['Beta Shift'].median()
    
    print("\n--- Summary ---")
    print(f"Average Beta Shift for non-Mag7 stocks: {avg_shift:+.4f}")
    print(f"Median Beta Shift for non-Mag7 stocks: {median_shift:+.4f}")
    
    if avg_shift < 0:
        print("\nInterpretation: A large negative shock to a Mag7 stock caused the calculated 'market' return to drop significantly on that day.")
        print("For a typical non-Mag7 stock, whose price was relatively unchanged, this event makes it appear to have moved *against* the market.")
        print("This pulls the regression-based beta *down*, making the stock seem less sensitive to market movements.")
        print("This highlights how index concentration can make systematic risk measures (beta) unstable and misleading.")
    else:
        print("\nInterpretation: The shock did not produce a consistent downward shift in non-Mag7 betas.")
        print("This could be due to the specific daily returns of the sampled stocks on the shock day.")


if __name__ == "__main__":
    returns_data = load_data(RETURNS_CSV)
    simulate_shock_scenario(returns_data)
