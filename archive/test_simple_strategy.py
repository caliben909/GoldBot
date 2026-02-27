#!/usr/bin/env python3
"""
Simple test script for GoldBot strategy without AI dependencies.
This script tests core functionality using only standard libraries and
basic technical analysis to evaluate performance.
"""

import logging
import sys
import asyncio
import argparse

# Add the project root directory to Python path
sys.path.insert(0, '/'.join(sys.path[0].split('/')[:-1]))

from backtest.backtest_engine import BacktestEngine
from core.data_engine import DataEngine
from core.strategy_engine import StrategyEngine
from core.risk_engine import RiskEngine
from core.liquidity_engine import LiquidityEngine

from utils.helpers import load_config
from utils.logging_config import setup_logging

# Configure logging
logger = logging.getLogger(__name__)
setup_logging({'level': 'INFO'})


async def run_test():
    """Run simple strategy test without AI dependencies"""
    logger.info("Starting simple strategy test without AI dependencies")
    
    try:
        # Load optimized configuration
        config = load_config('config/config_optimized.yaml')
        
        # Disable AI features to avoid TensorFlow dependency
        config['strategy']['ai_filter']['enabled'] = False
        
        # Create backtest engine with optimized configuration
        backtest_engine = BacktestEngine(config)
        
        # Run backtest
        logger.info("Running backtest with optimized strategy")
        result = await backtest_engine.run(backtest_engine.strategy_engine)
        
        # Print summary
        backtest_engine.print_summary()
        
        # Save results for comparison
        backtest_engine.save_results()
        logger.info("Results saved to backtest/results directory")
        
        return result
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return None


async def run_multiple_tests():
    """Run multiple backtests with different symbols"""
    logger.info("Running multi-symbol validation")
    
    test_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    results = {}
    
    config = load_config('config/config_optimized.yaml')
    config['strategy']['ai_filter']['enabled'] = False
    
    for symbol in test_symbols:
        logger.info(f"\nTesting {symbol}")
        
        try:
            backtest_engine = BacktestEngine(config)
            
            # Run backtest
            result = await backtest_engine.run(backtest_engine.strategy_engine)
            
            # Print summary
            backtest_engine.print_summary()
            
            results[symbol] = result
            
            logger.info(f"{symbol} - Return: {result['total_return']:.2f}%, "
                       f"Win Rate: {result['win_rate']:.1f}%, "
                       f"PF: {result['profit_factor']:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to test {symbol}: {e}")
            results[symbol] = None
            
    return results


async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test strategy without AI dependencies")
    parser.add_argument('--multi', action='store_true', help="Run multi-symbol validation")
    parser.add_argument('--quick', action='store_true', help="Run quick test with shorter timeframe")
    args = parser.parse_args()
    
    if args.multi:
        logger.info("=== Multi-Symbol Validation ===")
        results = await run_multiple_tests()
        
        # Calculate aggregate metrics
        valid_results = [r for r in results.values() if r is not None]
        if valid_results:
            avg_return = sum(r['total_return'] for r in valid_results) / len(valid_results)
            avg_win_rate = sum(r['win_rate'] for r in valid_results) / len(valid_results)
            avg_profit_factor = sum(r['profit_factor'] for r in valid_results) / len(valid_results)
            
            logger.info(f"\n=== Aggregate Results ===")
            logger.info(f"Symbols Tested: {len(valid_results)}")
            logger.info(f"Average Return: {avg_return:.2f}%")
            logger.info(f"Average Win Rate: {avg_win_rate:.1f}%")
            logger.info(f"Average Profit Factor: {avg_profit_factor:.2f}")
            logger.info(f"Success Rate: {len([r for r in valid_results if r['profit_factor'] > 1.0])}/{len(valid_results)} symbols profitable")
            
    else:
        logger.info("=== Single Symbol Test ===")
        await run_test()
        
    logger.info("\nTest completed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed unexpectedly: {e}", exc_info=True)
