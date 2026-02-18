#!/usr/bin/env python3
"""Test script to verify all modules are importable"""

def test_imports():
    print("Testing module imports...")
    
    try:
        from core import (
            DataEngine,
            ExecutionEngine,
            RiskEngine,
            StrategyEngine,
            SessionEngine,
            AIEngine,
            LiquidityEngine
        )
        print("✅ Core modules imported successfully")
    except Exception as e:
        print(f"❌ Core modules import failed: {e}")
    
    try:
        from backtest import BacktestEngine
        print("✅ Backtest module imported successfully")
    except Exception as e:
        print(f"❌ Backtest module import failed: {e}")
    
    try:
        from utils import (
            setup_logging,
            calculate_rsi,
            calculate_atr,
            load_csv_data,
            save_csv_data,
            format_currency,
            format_percentage,
            calculate_pips
        )
        print("✅ Utils module imported successfully")
    except Exception as e:
        print(f"❌ Utils module import failed: {e}")
    
    try:
        from live import LiveEngine
        print("✅ Live engine imported successfully")
    except Exception as e:
        print(f"❌ Live engine import failed: {e}")
    
    print("\nAll imports tested!")


if __name__ == "__main__":
    test_imports()
