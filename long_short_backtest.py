import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def run_backtest_analysis():
    """
    Performs a backtest comparison between a standard and modified long/short strategy.
    """
    # --- 1. Configuration & Data Loading ---
    START_DATE = '2020-01-01'
    END_DATE = '2024-12-31'
    CONSTITUENT_RETURNS_CSV = 'sp500_daily_returns.csv'
    SMALL_CAP_TICKER = 'IWM'  # Russell 2000 ETF
    LARGE_CAP_TICKER = 'SPY'  # S&P 500 ETF

    print("--- Starting Long/Short Backtest Analysis ---")
    print(f"Analysis Period: {START_DATE} to {END_DATE}")

    # --- 2. Load Constituent Data & S&P 493 Construction ---
    try:
        returns_df = pd.read_csv(CONSTITUENT_RETURNS_CSV, parse_dates=['Date'])
    except FileNotFoundError:
        print(f"Error: The file '{CONSTITUENT_RETURNS_CSV}' was not found.")
        print("Please run 'sp500_returns.py' first.")
        return

    # Filter for the analysis period
    returns_df = returns_df[(returns_df['Date'] >= START_DATE) & (returns_df['Date'] <= END_DATE)]
    
    # Create the "S&P 493" portfolio (equal-weighted returns of non-Mag7 stocks)
    sp493_returns = returns_df[returns_df['Group'] == 'Other'].groupby('Date')['Return'].mean()
    sp493_returns.name = "SP493_Return"
    
    print(f"Successfully calculated 'S&P 493' returns for {sp493_returns.shape[0]} days.")

    # --- 3. Download ETF Data ---
    print(f"Downloading ETF data for {SMALL_CAP_TICKER} and {LARGE_CAP_TICKER}...")
    etf_data = yf.download([SMALL_CAP_TICKER, LARGE_CAP_TICKER], start=START_DATE, end=END_DATE, auto_adjust=True)['Close']
    etf_returns = etf_data.pct_change().dropna()
    etf_returns.rename(columns={SMALL_CAP_TICKER: 'SmallCap_Return', LARGE_CAP_TICKER: 'LargeCap_Return'}, inplace=True)

    # --- 4. Combine Data & Build Strategies ---
    backtest_df = pd.concat([etf_returns, sp493_returns], axis=1).dropna()
    
    # Standard Strategy: Long Small-Cap, Short Large-Cap
    backtest_df['Standard_Strategy'] = backtest_df['SmallCap_Return'] - backtest_df['LargeCap_Return']
    
    # Modified Strategy: Long Small-Cap, Short S&P 493
    backtest_df['Modified_Strategy'] = backtest_df['SmallCap_Return'] - backtest_df['SP493_Return']
    
    print("Successfully built daily returns for both strategies.")

    # --- 5. Calculate Performance Metrics ---
    metrics = []
    for strategy in ['Standard_Strategy', 'Modified_Strategy']:
        daily_returns = backtest_df[strategy]
        
        # Cumulative Return
        cumulative_return = (1 + daily_returns).prod() - 1
        
        # Annualized Volatility
        annualized_vol = daily_returns.std() * np.sqrt(252)
        
        # Sharpe Ratio (assuming risk-free rate of 0)
        sharpe_ratio = (daily_returns.mean() * 252) / annualized_vol
        
        metrics.append({
            'Strategy': strategy,
            'Cumulative Return': f"{cumulative_return:.2%}",
            'Annualized Volatility': f"{annualized_vol:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}"
        })
        
    metrics_df = pd.DataFrame(metrics).set_index('Strategy')
    
    print("\n--- Backtest Performance Metrics ---")
    print(metrics_df.to_string())

    # --- 6. Plot Cumulative Performance ---
    backtest_df['Cumulative_Standard'] = (1 + backtest_df['Standard_Strategy']).cumprod()
    backtest_df['Cumulative_Modified'] = (1 + backtest_df['Modified_Strategy']).cumprod()

    try:
        style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(14, 7))

        plt.plot(backtest_df.index, backtest_df['Cumulative_Standard'], label='Standard Strategy (Long IWM / Short SPY)')
        plt.plot(backtest_df.index, backtest_df['Cumulative_Modified'], label='Modified Strategy (Long IWM / Short S&P 493)')

        plt.title('Backtest: Standard vs. Modified Long/Short Strategy', fontsize=16)
        plt.ylabel('Cumulative Growth of $1')
        plt.xlabel('Date')
        plt.legend(fontsize=12)
        plt.grid(True)
        
        # Add a horizontal line at 1.0 for reference
        plt.axhline(1.0, color='black', linestyle='--', linewidth=0.8)

        # Save the plot
        plot_filename = 'long_short_backtest_chart.png'
        plt.savefig(plot_filename)
        print(f"\nPlot saved successfully to '{plot_filename}'.")

    except ImportError:
        print("\nPlotting skipped: `matplotlib` could not be imported.")
    except Exception as e:
        print(f"\nAn error occurred during plotting: {e}")


if __name__ == '__main__':
    run_backtest_analysis()
