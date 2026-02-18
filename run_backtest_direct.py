#!/usr/bin/env python3
"""
Direct backtest runner to bypass main.py issues
"""
import asyncio
import sys
from pathlib import Path
import yaml
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


async def run_backtest():
    """Run the backtest directly"""
    logger.info("=" * 60)
    logger.info("RUNNING BACKTEST DIRECTLY")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        with open('config/config_optimized.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("Configuration loaded successfully")
        
        # Import BacktestEngine
        from backtest.backtest_engine import BacktestEngine
        
        logger.info("BacktestEngine imported successfully")
        
        # Create backtest engine
        backtest_engine = BacktestEngine(config)
        
        logger.info("BacktestEngine created successfully")
        
        # Run backtest
        logger.info("Starting backtest...")
        result = await backtest_engine.run_backtest()
        
        logger.info("Backtest completed!")
        
        # Print results
        logger.info("\nBacktest Results:")
        logger.info(f"Start Date: {result['start_date']}")
        logger.info(f"End Date: {result['end_date']}")
        logger.info(f"Initial Capital: ${result['initial_capital']:,.2f}")
        logger.info(f"Final Capital: ${result['final_capital']:,.2f}")
        logger.info(f"Total Returns: {result['total_returns']:.2%}")
        logger.info(f"Win Rate: {result['win_rate']:.2%}")
        logger.info(f"Profit Factor: {result['profit_factor']:.2f}")
        logger.info(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {result['max_drawdown']:.2%}")
        
        logger.info("\n✅ Backtest executed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}")
        logger.error(f"Stack Trace:")
        import traceback
        logger.error(traceback.format_exc())
        return False


asyncio.run(run_backtest())
