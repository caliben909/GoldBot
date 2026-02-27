import pandas as pd
from institutional.quant_framework import GoldInstitutionalFramework

def main():
    # Create dummy data
    dates = pd.date_range('2024-01-01', periods=100, freq='15T')
    
    # Create downward trending data
    df_down = pd.DataFrame({
        'open': [1.20 - i * 0.001 for i in range(100)],
        'high': [1.21 - i * 0.001 for i in range(100)],
        'low': [1.19 - i * 0.001 for i in range(100)],
        'close': [1.20 - i * 0.001 for i in range(100)],
        'volume': [10000 + i * 100 for i in range(100)]
    }, index=dates)
    
    # Create upward trending data
    df_up = pd.DataFrame({
        'open': [1.10 + i * 0.001 for i in range(100)],
        'high': [1.11 + i * 0.001 for i in range(100)],
        'low': [1.09 + i * 0.001 for i in range(100)],
        'close': [1.10 + i * 0.001 for i in range(100)],
        'volume': [10000 + i * 100 for i in range(100)]
    }, index=dates)
    
    dxy_data = pd.DataFrame({
        'close': [103 + i * 0.1 for i in range(100)]  # Upward trend
    }, index=dates)
    
    yield_data = pd.DataFrame({
        'close': [4.2 + i * 0.01 for i in range(100)]  # Upward trend
    }, index=dates)
    
    framework = GoldInstitutionalFramework()
    
    # Test downward trend
    print("=== Downward Trend ===")
    if len(df_down) > 10:
        symbol_structure = "bullish" if df_down['close'].iloc[-1] > df_down['close'].iloc[-10] else "bearish"
    else:
        symbol_structure = "neutral"
    
    if len(dxy_data) > 5:
        dxy_trend = "up" if dxy_data['close'].iloc[-1] > dxy_data['close'].iloc[-5] else "down"
    else:
        dxy_trend = "neutral"
        
    if len(yield_data) > 5:
        yield_trend = "up" if yield_data['close'].iloc[-1] > yield_data['close'].iloc[-5] else "down"
    else:
        yield_trend = "neutral"
    
    macro_score = framework.calculate_macro_score(dxy_trend, yield_trend, symbol_structure, symbol="EURUSD")
    macro_bias = framework.determine_macro_bias(macro_score, symbol_structure)
    
    print(f"Symbol Structure: {symbol_structure}")
    print(f"DXY Trend: {dxy_trend}")
    print(f"Yield Trend: {yield_trend}")
    print(f"Macro Score: {macro_score}")
    print(f"Macro Bias: {macro_bias}")
    
    # Test upward trend
    print("\n=== Upward Trend ===")
    if len(df_up) > 10:
        symbol_structure = "bullish" if df_up['close'].iloc[-1] > df_up['close'].iloc[-10] else "bearish"
    else:
        symbol_structure = "neutral"
    
    macro_score = framework.calculate_macro_score(dxy_trend, yield_trend, symbol_structure, symbol="EURUSD")
    macro_bias = framework.determine_macro_bias(macro_score, symbol_structure)
    
    print(f"Symbol Structure: {symbol_structure}")
    print(f"DXY Trend: {dxy_trend}")
    print(f"Yield Trend: {yield_trend}")
    print(f"Macro Score: {macro_score}")
    print(f"Macro Bias: {macro_bias}")

if __name__ == "__main__":
    main()
