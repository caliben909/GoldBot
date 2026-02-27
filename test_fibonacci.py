import pandas as pd
import numpy as np
from utils.indicators import TechnicalIndicators

def test_fibonacci_retracement():
    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    prices = np.random.uniform(1900, 2100, size=30)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.uniform(0, 20, size=30),
        'low': prices - np.random.uniform(0, 20, size=30),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, size=30)
    }, index=dates)
    
    print("DataFrame columns:", list(df.columns))
    
    indicators = TechnicalIndicators()
    fib_levels = indicators.calculate_fibonacci_retracement(df, lookback_period=20)
    
    print("\nFibonacci levels returned:")
    for col in fib_levels.columns:
        print(f"  {col}")
    
    # Check if 'fib_0.0' exists
    if 'fib_0.0' in fib_levels.columns:
        print("\n'fib_0.0' column found!")
        print(fib_levels['fib_0.0'].head())
    
    # Test nearest_fib function
    print("\nNearest Fib levels:")
    print(fib_levels[['nearest_fib', 'distance_to_fib']].head())
    
    return df, fib_levels

if __name__ == "__main__":
    print("Testing Fibonacci Retracement")
    print("=" * 50)
    df, fib_levels = test_fibonacci_retracement()
    print("\n" + "=" * 50)
    print("Test completed")
