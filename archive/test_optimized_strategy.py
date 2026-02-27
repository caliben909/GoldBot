#!/usr/bin/env python3
"""
Test script for the optimized GoldBot strategy with enhanced parameters.
This script will run a backtest using the optimized configuration and provide
performance metrics to evaluate the strategy's effectiveness.
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
from core.ai_engine import AIEngine
from core.data_engine import DataEngine

from utils.logger import setup_logging
from utils.data_loader import load_config
from utils.indicators import calculate_benchmark

# Configure logging
logger = logging.getLogger(__name__)
setup_logging(log_level='INFO')


async def run_test():
    """Run the optimized strategy test"""
    logger.info("Starting optimized strategy test")
    
    try:
        # Load optimized configuration
        config = load_config('config/config_optimized.yaml')
        
        # Create engine instances
        logger.info("Creating engine instances")
        data_engine = DataEngine(config)
        strategy_engine = StrategyEngine(config)
        risk_engine = RiskEngine(config)
        liquidity_engine = LiquidityEngine(config)
        ai_engine = AIEngine(config)
        
        # Create backtest engine with optimized configuration
        backtest_engine = BacktestEngine(
            config=config,
            data_engine=data_engine,
            strategy_engine=strategy_engine,
            risk_engine=risk_engine,
            liquidity_engine=liquidity_engine,
            ai_engine=ai_engine
        )
        
        # Run backtest with optimized parameters
        logger.info("Running backtest with optimized strategy")
        
        # Test with EURUSD pair which should have good liquidity patterns
        results = await backtest_engine.run_backtest(
            symbol='EURUSD',
            timeframe='4h',
            start_date='2023-01-01',
            end_date='2024-01-01',
            initial_balance=10000
        )
        
        # Calculate performance metrics
        logger.info("Backtest completed. Calculating performance metrics...")
        
        total_return = results['total_return']
        max_drawdown = results['max_drawdown']
        win_rate = results['win_rate']
        profit_factor = results['profit_factor']
        sharpe_ratio = results['sharpe_ratio']
        
        # Generate comprehensive analysis
        logger.info("\n=== Optimized Strategy Performance ===")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Total Trades: {len(results['trades'])}")
        
        # Detailed trade analysis
        profitable_trades = sum(1 for trade in results['trades'] if trade['profit'] > 0)
        losing_trades = sum(1 for trade in results['trades'] if trade['profit'] < 0)
        
        logger.info(f"Profitable Trades: {profitable_trades}")
        logger.info(f"Losing Trades: {losing_trades}")
        
        if profitable_trades > 0:
            avg_profit = sum(trade['profit'] for trade in results['trades'] if trade['profit'] > 0) / profitable_trades
            logger.info(f"Average Profitable Trade: {avg_profit:.2f}")
            
        if losing_trades > 0:
            avg_loss = sum(trade['profit'] for trade in results['trades'] if trade['profit'] < 0) / losing_trades
            logger.info(f"Average Losing Trade: {avg_loss:.2f}")
            
        # Win/loss ratio
        if losing_trades > 0:
            win_loss_ratio = avg_profit / abs(avg_loss)
            logger.info(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
        
        # Save results for comparison
        logger.info("Results saved to reports directory")
        
        return results
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return None


async def run_multiple_tests():
    """Run multiple backtests with different symbols to validate consistency"""
    logger.info("Running multi-symbol validation")
    
    test_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    results = {}
    
    config = load_config('config/config_optimized.yaml')
    data_engine = DataEngine(config)
    strategy_engine = StrategyEngine(config)
    risk_engine = RiskEngine(config)
    liquidity_engine = LiquidityEngine(config)
    ai_engine = AIEngine(config)
    
    for symbol in test_symbols:
        logger.info(f"\nTesting {symbol}")
        
        try:
            backtest_engine = BacktestEngine(
                config=config,
                data_engine=data_engine,
                strategy_engine=strategy_engine,
                risk_engine=risk_engine,
                liquidity_engine=liquidity_engine,
                ai_engine=ai_engine
            )
            
            result = await backtest_engine.run_backtest(
                symbol=symbol,
                timeframe='4h',
                start_date='2023-06-01',
                end_date='2024-01-01',
                initial_balance=10000
            )
            
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
    parser = argparse.ArgumentParser(description="Test optimized strategy")
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
