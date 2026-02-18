#!/usr/bin/env python3
"""
Main entry point for the Institutional Trading Bot
"""
import asyncio
import argparse
import sys
from pathlib import Path
import yaml
import logging
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_engine import DataEngine
from core.execution_engine import ExecutionEngine
from core.risk_engine import RiskEngine
from core.strategy_engine import StrategyEngine
from core.session_engine import SessionEngine
# from core.ai_engine import AIEngine
from core.liquidity_engine import LiquidityEngine
from live.live_engine import LiveEngine
from backtest.backtest_engine import BacktestEngine
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Institutional Trading Bot')
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['live', 'backtest', 'optimize', 'train'],
        default='backtest',
        help='Trading mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='Symbols to trade'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--risk',
        type=float,
        default=0.5,
        help='Risk per trade (%)'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=100000,
        help='Initial capital'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from file"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            import json
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")
    
    # Override with environment variables
    load_dotenv()
    
    return config


async def run_live_mode(config: dict, args):
    """Run live trading mode"""
    logger.info("=" * 60)
    logger.info("STARTING LIVE TRADING MODE")
    logger.info("=" * 60)
    
    # Override config with command line args
    if args.symbols:
        config['assets']['forex']['symbols'] = [s for s in args.symbols if len(s) == 6]
        config['assets']['crypto']['symbols'] = [s for s in args.symbols if 'USDT' in s]
    
    if args.risk:
        config['risk_management']['max_risk_per_trade'] = args.risk
    
    # Create and start live engine
    engine = LiveEngine(config)
    
    try:
        await engine.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        await engine.shutdown()
    except Exception as e:
        logger.error(f"Live trading failed: {e}", exc_info=True)
        await engine.shutdown()
        sys.exit(1)


async def run_backtest_mode(config: dict, args):
    """Run backtest mode"""
    logger.info("=" * 60)
    logger.info("STARTING BACKTEST MODE")
    logger.info("=" * 60)
    
    # Override config with command line args
    if args.symbols:
        config['assets']['forex']['symbols'] = [s for s in args.symbols if len(s) == 6]
        config['assets']['crypto']['symbols'] = [s for s in args.symbols if 'USDT' in s]
    
    if args.start:
        config['backtesting']['start_date'] = args.start
    
    if args.end:
        config['backtesting']['end_date'] = args.end
    
    if args.capital:
        config['backtesting']['initial_capital'] = args.capital
    
    # Create engines
    data_engine = DataEngine(config)
    strategy_engine = StrategyEngine(config)
    risk_engine = RiskEngine(config)
    
    # Set risk engine reference for strategy engine
    strategy_engine.set_risk_engine(risk_engine)
    backtest_engine = BacktestEngine(config)
    
    try:
        # Initialize
        await data_engine.initialize()
        
        # Load data
        data = {}
        for symbol in config['assets']['forex']['symbols'] + config['assets']['crypto']['symbols']:
            df = await data_engine.get_historical_data(
                symbol=symbol,
                timeframe="H1",
                start=datetime.strptime(config['backtesting']['start_date'], '%Y-%m-%d'),
                end=datetime.strptime(config['backtesting']['end_date'], '%Y-%m-%d')
            )
            if df is not None:
                data[symbol] = df
        
        # Run backtest
        metrics = await backtest_engine.run(strategy_engine, data)
        
        # Print results
        backtest_engine.print_summary()
        
        # Save results
        backtest_engine.save_results()
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        await data_engine.shutdown()


async def run_optimize_mode(config: dict, args):
    """Run parameter optimization mode"""
    logger.info("=" * 60)
    logger.info("STARTING OPTIMIZATION MODE")
    logger.info("=" * 60)
    
    # TODO: Implement parameter optimization
    logger.info("Optimization mode not yet implemented")
    pass


async def run_train_mode(config: dict, args):
    """Run model training mode"""
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING MODE")
    logger.info("=" * 60)
    
    # Create engines
    data_engine = DataEngine(config)
    # ai_engine = AIEngine(config)
    
    try:
        await data_engine.initialize()
        # await ai_engine.initialize()
        
        # Load training data
        X_train, y_train = await prepare_training_data(data_engine, config)
        
        # Train model
        # result = await ai_engine.train(X_train, y_train)
        
        logger.info(f"Training complete: {result}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        await data_engine.shutdown()
        await ai_engine.shutdown()


async def prepare_training_data(data_engine, config):
    """Prepare training data for AI model"""
    # TODO: Implement data preparation
    return None, None


async def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logger(__name__, config['logging'])
    
    # Run selected mode
    if args.mode == 'live':
        await run_live_mode(config, args)
    elif args.mode == 'backtest':
        await run_backtest_mode(config, args)
    elif args.mode == 'optimize':
        await run_optimize_mode(config, args)
    elif args.mode == 'train':
        await run_train_mode(config, args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())