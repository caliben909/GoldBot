#!/usr/bin/env python3
"""Simple backtest test script for debugging purposes"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_engine import DataEngine
from core.strategy_engine import StrategyEngine
from core.risk_engine import RiskEngine
from backtest.backtest_engine import BacktestEngine
from utils.logger import setup_logger
import yaml
from datetime import datetime


async def main():
    """Main test function"""
    logger = setup_logger(__name__, {'level': 'INFO'})
    
    # Load configuration
    with open('config/config_optimized.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("="*60)
    logger.info("SIMPLE BACKTEST DEBUG")
    logger.info("="*60)
    
    # Override config for testing
    config['assets']['forex']['symbols'] = ['XAUUSD']
    config['backtesting']['timeframes'] = ['1D']
    
    # Create engines
    data_engine = DataEngine(config)
    strategy_engine = StrategyEngine(config)
    risk_engine = RiskEngine(config)
    
    strategy_engine.set_risk_engine(risk_engine)
    backtest_engine = BacktestEngine(config)
    
    try:
        await data_engine.initialize()
        
        # Run backtest on 1D timeframe
        df = await data_engine.get_historical_data(
            symbol='XAUUSD',
            timeframe='1D',
            start=datetime(2023, 1, 1),
            end=datetime(2023, 12, 31)
        )
        
        data = {'XAUUSD': df}
        
        logger.info(f"Loaded {len(df)} days of data")
        
        # Debug: Print the first 10 days of data
        print("\nFirst 10 days of data:")
        print(df[['open', 'high', 'low', 'close']].head(10))
        
        # Run backtest
        metrics = await backtest_engine.run(strategy_engine, data)
        
        logger.info("Backtest completed")
        logger.info(f"Total trades: {metrics.total_trades}")
        logger.info(f"Win rate: {metrics.win_rate:.2f}%")
        logger.info(f"Total return: {metrics.total_return:.2f}%")
        logger.info(f"Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        
        # Print all trades
        print("\nAll Trades:")
        for i, trade in enumerate(backtest_engine.trades[:10], 1):  # Print first 10 trades
            print(f"{i}. Symbol: {trade.symbol}")
            print(f"   Direction: {trade.direction}")
            print(f"   Entry Price: {trade.entry_price:.2f}")
            print(f"   Exit Price: {trade.exit_price:.2f}")
            print(f"   Profit: ${trade.profit:.2f}")
            print(f"   Signal Type: {trade.signal_type}")
            print()
            
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        
    finally:
        await data_engine.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
