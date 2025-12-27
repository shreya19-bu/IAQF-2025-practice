import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings

# It's common for the pymannkendall library to be missing, so we'll handle its import.
try:
    import pymannkendall as mk
    _Pymannkendall_INSTALLED = True
except ImportError:
    _Pymannkendall_INSTALLED = False

warnings.filterwarnings("ignore", category=FutureWarning)

def test_beta_contribution_trend(contribution_csv_path):
    """
    Performs statistical tests to check for a trend in the Mag7 beta contribution.

    Args:
        contribution_csv_path (str): Path to the beta contribution CSV file.
    """
    # --- 1. Load the Data ---
    try:
        df = pd.read_csv(contribution_csv_path, parse_dates=['Date'], index_col='Date')
        # Drop any rows with NaN values that might have resulted from the rolling calculations
        df.dropna(inplace=True)
        if df.empty:
            print("Error: The data file is empty after dropping missing values.")
            return
        y = df['Mag7_Beta_Contribution']
    except FileNotFoundError:
        print(f"Error: The file '{contribution_csv_path}' was not found.")
        print("Please run 'beta_decomposition.py' first to generate the data.")
        return
    except KeyError:
        print(f"Error: The file '{contribution_csv_path}' does not contain the required 'Mag7_Beta_Contribution' column.")
        return
        
    print(f"Loaded data from {df.index.min().date()} to {df.index.max().date()}.")

    # --- 2. OLS Regression Test for Linear Trend ---
    print("\n--- Test 1: Ordinary Least Squares (OLS) Regression ---")
    
    # Create the independent variable: time trend (0, 1, 2, ...)
    X = np.arange(len(y))
    # Add a constant (intercept) to the model
    X = sm.add_constant(X)
    
    # Fit the linear regression model
    model = sm.OLS(y, X).fit()
    
    # Extract results
    intercept, slope = model.params
    p_value_slope = model.pvalues[1]

    # Print the model summary
    print(model.summary())

    # Interpretation
    print("\n--- OLS Interpretation ---")
    print(f"The slope of the trend is {slope:.6f}.")
    print(f"The p-value for the slope is {p_value_slope:.4f}.")

    if p_value_slope < 0.05 and slope > 0:
        print("\nConclusion: The p-value is less than 0.05 and the slope is positive.")
        print("We can conclude there is a STATISTICALLY SIGNIFICANT INCREASING linear trend in the Mag7 beta contribution over this period.")
    elif slope > 0:
        print("\nConclusion: The slope is positive, but the p-value is greater than 0.05.")
        print("This suggests an upward trend, but it is NOT STATISTICALLY SIGNIFICANT at the 5% level.")
    else:
        print("\nConclusion: The slope is not positive or the p-value is high.")
        print("There is no evidence of a statistically significant increasing linear trend.")

    # --- 3. Mann-Kendall Test for Monotonic Trend ---
    print("\n\n--- Test 2: Mann-Kendall (MK) Non-Parametric Test ---")
    if not _Pymannkendall_INSTALLED:
        print("Mann-Kendall test skipped because 'pymannkendall' is not installed.")
        print("Please install it to run this test: py -m pip install pymannkendall")
        return

    # Perform the Mann-Kendall test
    mk_result = mk.original_test(y)
    
    # Interpretation
    print("\n--- MK Interpretation ---")
    print(f"Trend identified: {mk_result.trend}")
    print(f"Test statistic (Kendall's tau): {mk_result.Tau:.4f}")
    print(f"P-value: {mk_result.p:.4f}")

    if mk_result.trend == 'increasing' and mk_result.p < 0.05:
        print("\nConclusion: The test finds a STATISTICALLY SIGNIFICANT INCREASING monotonic trend.")
    elif mk_result.trend == 'increasing':
        print("\nConclusion: The test finds an increasing trend, but it is NOT STATISTICALLY SIGNIFICANT at the 5% level.")
    else:
        print("\nConclusion: The test does not find evidence of a significant increasing monotonic trend.")


def main():
    """
    Main execution function.
    """
    CONTRIBUTION_CSV = 'beta_contribution_by_group.csv'
    test_beta_contribution_trend(CONTRIBUTION_CSV)

if __name__ == "__main__":
    main()
