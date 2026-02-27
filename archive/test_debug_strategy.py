#!/usr/bin/env python3
"""Debug script to test strategy signal generation with smaller timeframes"""
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
    logger = setup_logger(__name__, {'level': 'DEBUG'})
    
    # Load configuration
    with open('config/config_optimized.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("="*60)
    logger.info("STRATEGY SIGNAL GENERATION DEBUG")
    logger.info("="*60)
    
    # Override config for debugging
    config['assets']['forex']['symbols'] = ['XAUUSD']
    config['backtesting']['timeframes'] = ['4H']  # Try 4-hour timeframe
    
    # Adjust strategy parameters for better signal generation on 4H timeframe
    config['strategy']['smc']['swing_length'] = 5
    config['strategy']['smc']['fvg_min_size'] = 0.001
    config['strategy']['smc']['liquidity_lookback'] = 10
    
    config['strategy']['kill_zone']['enabled'] = False
    config['strategy']['ai_filter']['enabled'] = False
    
    config['risk_management']['dxy_correlation']['enabled'] = False
    
    logger.debug(f"Strategy config: {yaml.dump(config['strategy'])}")
    
    # Create engines
    data_engine = DataEngine(config)
    strategy_engine = StrategyEngine(config)
    risk_engine = RiskEngine(config)
    
    strategy_engine.set_risk_engine(risk_engine)
    backtest_engine = BacktestEngine(config)
    
    try:
        await data_engine.initialize()
        
        # Run backtest on 4H timeframe (last 60 days)
        df = await data_engine.get_historical_data(
            symbol='XAUUSD',
            timeframe='4H',
            start=datetime(2024, 10, 1),
            end=datetime(2024, 12, 1)
        )
        
        data = {'XAUUSD': df}
        
        logger.info(f"Loaded {len(df)} 4H bars of data")
        
        # Debug: Print liquidity and FVG analysis
        logger.debug("\n=== Liquidity and FVG Analysis ===")
        liquidity_engine = strategy_engine.liquidity_engine
        liquidity_analysis = liquidity_engine.analyze_market(df)
        
        logger.debug(f"Swing Highs: {len(liquidity_analysis['structure']['swing_highs'])}")
        if liquidity_analysis['structure']['swing_highs']:
            logger.debug(f"Last Swing High: {liquidity_analysis['structure']['swing_highs'][-1]}")
        
        logger.debug(f"Swing Lows: {len(liquidity_analysis['structure']['swing_lows'])}")
        if liquidity_analysis['structure']['swing_lows']:
            logger.debug(f"Last Swing Low: {liquidity_analysis['structure']['swing_lows'][-1]}")
        
        logger.debug(f"FVG Zones: {len(liquidity_analysis['fvg_zones'])}")
        for i, fvg in enumerate(liquidity_analysis['fvg_zones'][-3:], 1):
            logger.debug(f"  FVG {i}: {fvg.type} - {fvg.top:.2f} - {fvg.bottom:.2f} (mitigated: {fvg.mitigated})")
        
        logger.debug(f"Order Blocks: {len(liquidity_analysis['order_blocks'])}")
        for i, ob in enumerate(liquidity_analysis['order_blocks'][-3:], 1):
            logger.debug(f"  OB {i}: {ob.direction} - {ob.price_range} (mitigated: {ob.mitigated})")
        
        # Debug: Try to generate signals directly
        logger.debug("\n=== Attempting Signal Generation ===")
        try:
            signals = await strategy_engine.generate_trading_signals(df, 'XAUUSD')
            logger.debug(f"Generated {len(signals)} signals")
            
            if signals:
                logger.debug("\nSignals Generated:")
                for i, signal in enumerate(signals, 1):
                    logger.debug(f"{i}. Signal Type: {signal.signal_type.value}")
                    logger.debug(f"   Direction: {signal.direction}")
                    logger.debug(f"   Entry Price: {signal.entry_price:.2f}")
                    logger.debug(f"   Stop Loss: {signal.stop_loss:.2f}")
                    logger.debug(f"   Take Profit: {signal.take_profit:.2f}")
                    logger.debug(f"   Confidence: {signal.confidence:.2f}")
                    logger.debug(f"   Confluences: {signal.confluences}")
                    logger.debug()
            else:
                logger.warning("No signals generated")
                
        except Exception as e:
            logger.error(f"Error generating signals: {e}", exc_info=True)
        
        # Run full backtest
        logger.debug("\n=== Running Backtest ===")
        metrics = await backtest_engine.run(strategy_engine, data)
        
        logger.info("\nBacktest completed")
        logger.info(f"Total trades: {metrics.total_trades}")
        logger.info(f"Win rate: {metrics.win_rate:.2f}%")
        logger.info(f"Total return: {metrics.total_return:.2f}%")
        logger.info(f"Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        
        if metrics.total_trades > 0:
            logger.info("\nFirst 5 Trades:")
            for i, trade in enumerate(backtest_engine.trades[:5], 1):
                logger.info(f"{i}. Symbol: {trade.symbol}")
                logger.info(f"   Direction: {trade.direction}")
                logger.info(f"   Entry Price: {trade.entry_price:.2f}")
                logger.info(f"   Exit Price: {trade.exit_price:.2f}")
                logger.info(f"   Profit: ${trade.profit:.2f}")
                logger.info(f"   Signal Type: {trade.signal_type}")
                logger.info("")
                
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        
    finally:
        await data_engine.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
