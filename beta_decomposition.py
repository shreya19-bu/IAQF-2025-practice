import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import numpy as np
import warnings

# Suppress warnings from statsmodels, which can be noisy with rolling regressions
warnings.filterwarnings("ignore", category=FutureWarning)

def get_market_caps(tickers, start_date, end_date):
    """
    Downloads closing prices and shares outstanding to calculate daily market caps.
    This is a proxy for true historical market cap but is effective for this analysis.

    Returns a tuple of (market_caps_df, prices_df).
    """
    print("Fetching shares outstanding for all tickers... (This may take a few minutes)")
    shares_outstanding = {}
    failed_tickers_shares = []
    for ticker in tickers:
        try:
            stock_info = yf.Ticker(ticker).info
            if 'sharesOutstanding' in stock_info and stock_info['sharesOutstanding'] is not None:
                shares_outstanding[ticker] = stock_info['sharesOutstanding']
            else:
                raise ValueError("sharesOutstanding not available")
        except Exception as e:
            failed_tickers_shares.append(ticker)
            # print(f"Could not get shares for {ticker}. Reason: {e}")

    if failed_tickers_shares:
        print(f"\nWarning: Could not retrieve shares outstanding for {len(failed_tickers_shares)} tickers. They will be excluded from the value-weighted index.")
    
    valid_tickers = list(shares_outstanding.keys())
    shares_series = pd.Series(shares_outstanding)

    print(f"\nDownloading daily prices for {len(valid_tickers)} tickers to calculate market caps...")
    prices = yf.download(valid_tickers, start=start_date, end=end_date, auto_adjust=False)['Close']
    
    # Forward-fill and back-fill to handle missing price data for robustness
    prices = prices.ffill().bfill()
    
    # Calculate daily market caps
    market_caps = prices * shares_series
    
    # Drop any columns that are all NaN (if a ticker had no price data at all)
    market_caps.dropna(axis=1, how='all', inplace=True)
    
    print(f"Successfully calculated daily market caps for {market_caps.shape[1]} tickers.")
    return market_caps, prices

def calculate_beta_contribution(start_date, end_date, returns_csv_path):
    """
    Performs the full beta decomposition analysis.
    """
    # --- 1. Load Constituent Returns ---
    try:
        returns_df = pd.read_csv(returns_csv_path, parse_dates=['Date'])
        returns_df.set_index('Date', inplace=True)
    except FileNotFoundError:
        print(f"Error: The file '{returns_csv_path}' was not found.")
        print("Please run the 'sp500_returns.py' script first.")
        return

    # Pivot the returns data to have tickers as columns
    returns_pivot = returns_df.pivot(columns='Ticker', values='Return')
    
    unique_tickers = returns_df['Ticker'].unique().tolist()

    # --- 2. Calculate Value-Weighted Market Return ---
    market_caps, prices = get_market_caps(unique_tickers, start_date, end_date)
    
    # Ensure data aligns by dropping tickers not present in both market_caps and returns
    common_tickers = list(set(market_caps.columns) & set(returns_pivot.columns))
    market_caps = market_caps[common_tickers]
    returns_pivot = returns_pivot[common_tickers]

    total_market_cap = market_caps.sum(axis=1)
    weights = market_caps.div(total_market_cap, axis=0)

    # Calculate the value-weighted market return
    market_return = (weights * returns_pivot).sum(axis=1).rename("market_return")
    market_return = market_return.asfreq('B').ffill() # Ensure business day frequency

    # --- 3. Calculate Rolling Betas ---
    print("\nCalculating rolling 6-month (126-day) CAPM betas for each stock...")
    rolling_betas = {}
    endog = market_return.dropna()
    exog = sm.add_constant(endog) # Add intercept for regression

    for ticker in returns_pivot.columns:
        try:
            # Align stock returns with market returns
            y = returns_pivot[ticker].dropna()
            df_aligned = pd.concat([y, exog], axis=1).dropna()
            
            if df_aligned.shape[0] < 126: # Not enough data to run regression
                continue

            # Run rolling OLS
            rols = RollingOLS(
                endog=df_aligned[ticker], 
                exog=df_aligned[['const', 'market_return']], 
                window=126
            )
            rres = rols.fit()
            rolling_betas[ticker] = rres.params['market_return']
        except Exception as e:
            # print(f"Could not calculate beta for {ticker}. Reason: {e}")
            pass
            
    betas_df = pd.DataFrame(rolling_betas).dropna(how='all')
    print(f"Successfully calculated rolling betas for {betas_df.shape[1]} tickers.")

    # --- 4. Aggregate Beta Contribution ---
    print("\nAggregating beta contributions by group (Mag7 vs. Other)...")
    
    # Align the weights and betas dataframes
    aligned_weights, aligned_betas = weights.align(betas_df, join='inner', axis=0)

    # Calculate weighted beta for each stock on each day
    weighted_betas = aligned_weights * aligned_betas

    # Get group info
    groups = returns_df[['Ticker', 'Group']].drop_duplicates().set_index('Ticker')['Group']
    mag7_tickers = groups[groups == 'Mag7'].index.tolist()
    other_tickers = groups[groups == 'Other'].index.tolist()
    
    # Filter for tickers that are in our weighted_betas dataframe
    mag7_tickers_in_data = [t for t in mag7_tickers if t in weighted_betas.columns]
    other_tickers_in_data = [t for t in other_tickers if t in weighted_betas.columns]

    # Calculate contribution for each group
    mag7_contribution = weighted_betas[mag7_tickers_in_data].sum(axis=1)
    other_contribution = weighted_betas[other_tickers_in_data].sum(axis=1)

    # Combine into a final DataFrame
    contribution_df = pd.DataFrame({
        'Mag7_Beta_Contribution': mag7_contribution,
        'Other_Beta_Contribution': other_contribution,
    })
    contribution_df['Total_Market_Beta'] = contribution_df.sum(axis=1)

    return contribution_df

def main():
    """
    Main execution function.
    """
    # --- Configuration ---
    START_DATE = '2018-01-01'
    END_DATE = '2024-12-31'
    RETURNS_CSV = 'sp500_daily_returns.csv'
    OUTPUT_CSV = 'beta_contribution_by_group.csv'

    # --- Run Analysis ---
    contribution_timeseries = calculate_beta_contribution(START_DATE, END_DATE, RETURNS_CSV)

    if contribution_timeseries is not None and not contribution_timeseries.empty:
        # --- Output Results ---
        contribution_timeseries.to_csv(OUTPUT_CSV)
        print(f"\nSuccessfully calculated beta contributions. Data saved to '{OUTPUT_CSV}'.")
        print("\nSample of the final data:")
        print(contribution_timeseries.tail())

        # --- Plotting ---
        try:
            import matplotlib.pyplot as plt
            import matplotlib.style as style
            style.use('seaborn-v0_8-darkgrid')
            
            ax = contribution_timeseries[['Mag7_Beta_Contribution', 'Other_Beta_Contribution']].plot(
                kind='area',
                stacked=True,
                figsize=(14, 7),
                title='S&P 500 Beta Contribution: Mag7 vs. Other Constituents (6-Month Rolling)',
                alpha=0.8
            )
            ax.set_ylabel("Beta Contribution (Sum = 1.0)")
            ax.set_xlabel("Date")
            ax.legend(title='Group')
            ax.axhline(1, color='black', linestyle='--', linewidth=1, label='Total Market Beta (1.0)')
            ax.set_ylim(0, max(1.1, contribution_timeseries['Total_Market_Beta'].max() * 1.05))

            plt.savefig('beta_contribution_chart.png')
            print("\nPlot saved to 'beta_contribution_chart.png'.")

        except ImportError:
            print("\nPlotting skipped: `matplotlib` is not installed. Please install it with `pip install matplotlib`.")


if __name__ == "__main__":
    main()
