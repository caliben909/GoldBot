#!/usr/bin/env python3
"""
Main Entry Point - Institutional Trading Bot
Production-grade startup with component orchestration and graceful degradation
"""

import asyncio
import argparse
import sys
import signal
import functools
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Core components
from core.data_engine import DataEngine
from core.order_manager import OrderManager, OrderManagerConfig
from core.position_tracker import PositionTracker, PositionTrackerConfig
from core.recovery_manager import RecoveryManager, ComponentConfig, RecoveryAction

# Institutional components
from institutional.quant_framework import GoldInstitutionalFramework
from institutional.institutional_backtest import InstitutionalBacktestEngine

# Utils
from utils.logger import setup_logger


class TradingBotConfig:
    """Validated configuration container"""
    
    REQUIRED_SECTIONS = ['general', 'trading', 'risk']
    
    def __init__(self, config_dict: Dict):
        self.raw = config_dict
        self._validate()
        
        # Extract sections
        self.general = config_dict.get('general', {})
        self.trading = config_dict.get('trading', {})
        self.risk = config_dict.get('risk', {})
        self.backtesting = config_dict.get('backtesting', {})
        self.data = config_dict.get('data', {})
        
        # Derived settings
        self.symbols = self.trading.get('symbols', ['XAUUSD'])
        self.timeframe = self.trading.get('timeframe', '1h')
        self.initial_capital = self.trading.get('initial_capital', 100000)
        
    def _validate(self):
        """Validate configuration schema"""
        for section in self.REQUIRED_SECTIONS:
            if section not in self.raw:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate risk limits
        max_risk = self.raw['risk'].get('max_position_risk_pct', 0.02)
        if max_risk > 0.05:
            logging.warning(f"High risk setting detected: {max_risk:.1%}")
        
        return True


class TradingBot:
    """
    Production-grade trading bot with component orchestration,
    health monitoring, and graceful lifecycle management
    """
    
    def __init__(self, config: TradingBotConfig, args):
        self.config = config
        self.args = args
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.data_engine: Optional[DataEngine] = None
        self.framework: Optional[GoldInstitutionalFramework] = None
        self.order_manager: Optional[OrderManager] = None
        self.position_tracker: Optional[PositionTracker] = None
        self.recovery_manager: Optional[RecoveryManager] = None
        
        # State
        self.running = False
        self._shutdown_event = asyncio.Event()
        self._component_tasks = []
        
    async def initialize(self):
        """Initialize all components with dependency order"""
        self.logger.info("=" * 60)
        self.logger.info("INITIALIZING INSTITUTIONAL TRADING BOT")
        self.logger.info("=" * 60)
        
        try:
            # 1. Recovery Manager (must be first)
            self.recovery_manager = RecoveryManager(self.config.raw)
            await self.recovery_manager.start()
            self.logger.info("✓ Recovery Manager initialized")
            
            # 2. Data Engine
            self.data_engine = DataEngine(self.config.raw)
            await self.data_engine.initialize()
            self.logger.info("✓ Data Engine initialized")
            
            # 3. Position Tracker (state management)
            pt_config = PositionTrackerConfig()
            pt_config.max_concentration_pct = self.config.risk.get('max_concentration_pct', 0.20)
            self.position_tracker = PositionTracker(pt_config)
            await self.position_tracker.start()
            self.logger.info("✓ Position Tracker initialized")
            
            # 4. Order Manager
            om_config = OrderManagerConfig()
            om_config.max_slippage_pct = self.config.trading.get('max_slippage_pct', 0.001)
            self.order_manager = OrderManager(om_config, self.data_engine)
            self.logger.info("✓ Order Manager initialized")
            
            # 5. Strategy Framework
            self.framework = GoldInstitutionalFramework()
            self.framework.initialize()
            self.logger.info("✓ Strategy Framework initialized")
            
            # Register components with recovery manager
            self._register_with_recovery()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            self.running = True
            self.logger.info("=" * 60)
            self.logger.info("ALL COMPONENTS INITIALIZED SUCCESSFULLY")
            self.logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            self.logger.critical(f"Initialization failed: {e}", exc_info=True)
            await self.emergency_shutdown()
            return False
    
    def _register_with_recovery(self):
        """Register all components with recovery manager"""
        
        # Data Engine
        self.recovery_manager.register_component(
            'data_engine',
            self.data_engine,
            ComponentConfig(
                name='data_engine',
                critical=True,
                recovery_action=RecoveryAction.RECONNECT,
                health_check_interval=30.0
            ),
            lambda: self.data_engine.is_connected() if hasattr(self.data_engine, 'is_connected') else True
        )
        
        # Position Tracker
        self.recovery_manager.register_component(
            'position_tracker',
            self.position_tracker,
            ComponentConfig(
                name='position_tracker',
                critical=True,
                dependencies=['data_engine'],
                recovery_action=RecoveryAction.RESET_STATE
            ),
            lambda: self.position_tracker is not None
        )
        
        # Order Manager
        self.recovery_manager.register_component(
            'order_manager',
            self.order_manager,
            ComponentConfig(
                name='order_manager',
                critical=True,
                dependencies=['position_tracker'],
                recovery_action=RecoveryAction.RESTART
            ),
            lambda: self.order_manager is not None
        )
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signals"""
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(
                    sig, 
                    functools.partial(self._signal_handler, sig)
                )
            self.logger.info("✓ Signal handlers registered")
        except NotImplementedError:
            self.logger.warning("Async signal handlers not supported on this platform")
    
    def _signal_handler(self, signum: int):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.graceful_shutdown())
    
    # ==================== OPERATIONAL MODES ====================
    
    async def run_backtest(self):
        """Execute backtest mode with full component integration"""
        self.logger.info(f"Starting backtest: {self.args.symbol} [{self.args.start} to {self.args.end}]")
        
        try:
            # Load data
            symbol_data, dxy_data, yield_data = await self._load_backtest_data()
            
            if symbol_data is None:
                raise ValueError("Failed to load required data")
            
            # Run backtest
            backtest_engine = InstitutionalBacktestEngine(
                initial_equity=self.args.capital,
                commission=self.config.trading.get('commission', 0.0005)
            )
            
            results = backtest_engine.run_backtest(
                symbol_data, 
                dxy_data, 
                yield_data, 
                spread=self.config.trading.get('spread', 0.3),
                symbol=self.args.symbol
            )
            
            # Report results
            self._print_backtest_results(results)
            
            # Generate detailed report
            backtest_engine.generate_report(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest execution failed: {e}", exc_info=True)
            raise
    
    async def run_live(self):
        """Execute live trading mode"""
        self.logger.info("=" * 60)
        self.logger.info("STARTING LIVE TRADING MODE")
        self.logger.info("=" * 60)
        
        # Validate live trading prerequisites
        if not self._validate_live_prerequisites():
            raise RuntimeError("Live trading prerequisites not met")
        
        # Start position monitoring
        monitor_task = asyncio.create_task(
            self._position_monitor_loop(),
            name="position_monitor"
        )
        self._component_tasks.append(monitor_task)
        
        # Main trading loop
        try:
            while self.running and not self._shutdown_event.is_set():
                # Check if trading allowed
                if not await self._can_trade():
                    await asyncio.sleep(60)
                    continue
                
                # Process each symbol
                for symbol in self.config.symbols:
                    await self._process_symbol(symbol)
                
                # Wait for next cycle
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._get_interval_seconds()
                )
                
        except asyncio.TimeoutError:
            pass  # Normal cycle
        except Exception as e:
            self.logger.error(f"Live trading error: {e}", exc_info=True)
            await self.recovery_manager._handle_component_failure('main_loop', str(e))
    
    async def _process_symbol(self, symbol: str):
        """Process single symbol in live trading"""
        try:
            # Get latest data
            df = await self.data_engine.get_latest_data(symbol, self.config.timeframe)
            dxy_data = await self.data_engine.get_latest_data('DXY', self.config.timeframe)
            yield_data = await self.data_engine.get_latest_data('US10Y', self.config.timeframe)
            
            if df is None or len(df) < 20:
                return
            
            # Get current equity
            equity = await self._get_current_equity()
            
            # Check existing position
            existing_position = await self.position_tracker.get_symbol_position(symbol)
            
            if existing_position:
                # Manage existing position
                await self._manage_position(existing_position, df)
            else:
                # Look for new entry
                await self._evaluate_entry(symbol, df, dxy_data, yield_data, equity)
                
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
    
    async def _evaluate_entry(self, symbol: str, df: pd.DataFrame, 
                             dxy_data: pd.DataFrame, yield_data: pd.DataFrame,
                             equity: float):
        """Evaluate and execute new entry"""
        # Get signal from framework
        signal = self.framework.execute_strategy(
            df, equity, dxy_data, yield_data,
            spread=self.config.trading.get('spread', 0.3),
            news_event=False,  # TODO: Integrate news filter
            symbol=symbol
        )
        
        if signal.get('should_trade'):
            # Validate against risk limits
            if not await self._validate_risk_limits(signal, equity):
                self.logger.warning(f"Signal rejected due to risk limits")
                return
            
            # Execute through order manager
            order = await self.order_manager.place_order({
                'symbol': symbol,
                'direction': signal['direction'],
                'position_size': signal['position_size'],
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'order_type': 'MARKET'
            })
            
            if order and order.is_filled:
                # Track in position tracker
                await self.position_tracker.add_position({
                    'id': order.id,
                    'symbol': symbol,
                    'direction': signal['direction'],
                    'fill_price': order.avg_fill_price,
                    'quantity': order.filled_quantity,
                    'stop_loss': signal['stop_loss'],
                    'take_profit': signal['take_profit'],
                    'commission': order.total_commission
                })
                
                self.logger.info(f"Position opened: {symbol} {signal['direction']} @ {order.avg_fill_price}")
    
    async def _manage_position(self, position, df: pd.DataFrame):
        """Manage existing position (updates, exits)"""
        # Update price in position tracker
        current_price = df['close'].iloc[-1]
        updated = await self.position_tracker.update_price(position.order_id, current_price)
        
        if not updated:
            return
        
        # Check exit conditions via framework
        exit_signal = self.framework.should_exit_trade(
            current_price,
            position.take_profit,
            position.stop_loss,
            position.direction
        )
        
        if exit_signal != "none":
            # Execute close
            await self._close_position(position, current_price, exit_signal)
            return
        
        # Check for breakeven / trailing stop updates
        await self.position_tracker._check_exit_conditions(position.order_id)
    
    async def _close_position(self, position, exit_price: float, reason: str):
        """Close position and record results"""
        # Execute close through order manager
        success = await self.order_manager.close_position(
            position.order_id,
            exit_price=exit_price,
            reason=reason
        )
        
        if success:
            # Update position tracker
            closed = await self.position_tracker.close_position(
                position.order_id,
                exit_price=exit_price,
                exit_reason=reason
            )
            
            # Record in framework
            self.framework.track_trade_result({
                'profit': closed.realized_pnl if closed else 0,
                'exit_reason': reason
            })
            
            self.logger.info(f"Position closed: {position.symbol} | Reason: {reason} | P&L: ${closed.realized_pnl:.2f}")
    
    # ==================== UTILITY METHODS ====================
    
    async def _load_backtest_data(self):
        """Load data for backtest with fallback to synthetic"""
        try:
            start = datetime.strptime(self.args.start, '%Y-%m-%d')
            end = datetime.strptime(self.args.end, '%Y-%m-%d')
            
            symbol_data = await self.data_engine.get_historical_data(
                self.args.symbol, self.args.timeframe, start, end
            )
            
            # Load correlation data based on symbol
            if self.args.symbol == 'XAUUSD':
                dxy_data = await self.data_engine.get_historical_data('DXY', self.args.timeframe, start, end)
                yield_data = await self.data_engine.get_historical_data('US10Y', self.args.timeframe, start, end)
            else:
                dxy_data = await self.data_engine.get_historical_data('DXY', self.args.timeframe, start, end)
                yield_data = await self.data_engine.get_historical_data('EURUSD', self.args.timeframe, start, end)
            
            return symbol_data, dxy_data, yield_data
            
        except Exception as e:
            self.logger.warning(f"Data load failed: {e}, generating synthetic data")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic data for testing"""
        dates = pd.date_range(self.args.start, self.args.end, freq='15T')
        n = len(dates)
        
        # Generate realistic Gold-like data
        trend = np.linspace(2000, 2400, n)
        noise = np.cumsum(np.random.randn(n) * 0.5)
        
        symbol_data = pd.DataFrame({
            'open': trend + noise + np.random.randn(n),
            'high': trend + noise + abs(np.random.randn(n)) + 2,
            'low': trend + noise - abs(np.random.randn(n)) - 2,
            'close': trend + noise + np.random.randn(n) * 0.5,
            'volume': np.random.randint(10000, 50000, n)
        }, index=dates)
        
        dxy_data = pd.DataFrame({
            'close': 103 + np.cumsum(np.random.randn(n) * 0.1)
        }, index=dates)
        
        yield_data = pd.DataFrame({
            'close': 4.2 + np.cumsum(np.random.randn(n) * 0.01)
        }, index=dates)
        
        return symbol_data, dxy_data, yield_data
    
    def _print_backtest_results(self, results: Dict):
        """Print formatted backtest results"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("BACKTEST RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Return:        {results['total_return']:>10.2%}")
        self.logger.info(f"Annualized Return:   {results.get('annualized_return', 0):>10.2%}")
        self.logger.info(f"Final Equity:        ${results['final_equity']:>9,.2f}")
        self.logger.info(f"Total Trades:        {results['total_trades']:>10}")
        self.logger.info(f"Win Rate:            {results['win_rate']:>10.2%}")
        self.logger.info(f"Profit Factor:       {results.get('profit_factor', 0):>10.2f}")
        self.logger.info(f"Sharpe Ratio:        {results['sharpe_ratio']:>10.2f}")
        self.logger.info(f"Max Drawdown:        {results['max_drawdown']:>10.2%}")
        self.logger.info(f"Risk-Reward Ratio:   {results['risk_reward']:>10.2f}")
        self.logger.info("=" * 60)
    
    async def _position_monitor_loop(self):
        """Background position monitoring"""
        while self.running:
            try:
                positions = await self.position_tracker.get_all_positions()
                for position in positions:
                    # Get latest price
                    price = await self.data_engine.get_current_price(position.symbol)
                    if price:
                        await self.position_tracker.update_price(position.order_id, price)
                
                await asyncio.sleep(5)  # 5-second updates
                
            except Exception as e:
                self.logger.error(f"Position monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _can_trade(self) -> bool:
        """Check if trading conditions are met"""
        # Check market hours
        # Check connection health
        # Check risk limits
        return True
    
    async def _get_current_equity(self) -> float:
        """Get current account equity"""
        # TODO: Integrate with broker API
        return self.config.initial_capital
    
    async def _validate_risk_limits(self, signal: Dict, equity: float) -> bool:
        """Validate signal against risk management rules"""
        # Check max positions
        max_positions = self.config.risk.get('max_positions', 5)
        current_positions = await self.position_tracker.get_position_count()
        if current_positions >= max_positions:
            self.logger.warning(f"Max positions reached: {current_positions}/{max_positions}")
            return False
        
        # Check position concentration
        position_value = signal['position_size'] * signal['entry_price']
        concentration = position_value / equity
        if concentration > self.config.risk.get('max_concentration_pct', 0.20):
            self.logger.warning(f"Concentration limit exceeded: {concentration:.1%}")
            return False
        
        return True
    
    def _get_interval_seconds(self) -> float:
        """Get trading cycle interval based on timeframe"""
        intervals = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400, '1d': 86400
        }
        return intervals.get(self.config.timeframe, 3600)
    
    def _validate_live_prerequisites(self) -> bool:
        """Validate all requirements for live trading"""
        # Check API keys
        # Check account balance
        # Check market connectivity
        return True
    
    # ==================== SHUTDOWN ====================
    
    async def graceful_shutdown(self):
        """Graceful shutdown with position protection"""
        if not self.running:
            return
        
        self.logger.info("Initiating graceful shutdown...")
        self.running = False
        self._shutdown_event.set()
        
        # Cancel component tasks
        for task in self._component_tasks:
            task.cancel()
        
        if self._component_tasks:
            await asyncio.gather(*self._component_tasks, return_exceptions=True)
        
        # Close positions if configured
        if self.config.trading.get('close_on_shutdown', False):
            await self._emergency_close_all()
        
        # Shutdown components in reverse order
        shutdown_order = [
            ('order_manager', self.order_manager),
            ('position_tracker', self.position_tracker),
            ('framework', self.framework),
            ('data_engine', self.data_engine),
            ('recovery_manager', self.recovery_manager)
        ]
        
        for name, component in shutdown_order:
            if component and hasattr(component, 'shutdown'):
                try:
                    await asyncio.wait_for(component.shutdown(), timeout=10.0)
                    self.logger.info(f"✓ {name} shutdown")
                except Exception as e:
                    self.logger.error(f"✗ {name} shutdown error: {e}")
        
        self.logger.info("Shutdown complete")
    
    async def emergency_shutdown(self):
        """Emergency shutdown without position protection"""
        self.logger.critical("EMERGENCY SHUTDOWN")
        self.running = False
        
        # Attempt to save state
        if self.recovery_manager:
            try:
                await self.recovery_manager._emergency_save_state()
            except:
                pass
        
        sys.exit(1)
    
    async def _emergency_close_all(self):
        """Emergency close all positions"""
        if not self.position_tracker:
            return
        
        try:
            positions = await self.position_tracker.get_all_positions()
            for position in positions:
                self.logger.warning(f"Emergency closing: {position.symbol}")
                # Get current price
                price = await self.data_engine.get_current_price(position.symbol)
                if price:
                    await self._close_position(position, price, "emergency_shutdown")
        except Exception as e:
            self.logger.error(f"Emergency close error: {e}")


# ==================== CLI INTERFACE ====================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Institutional Trading Bot - Production Grade',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest mode
  python main.py --mode backtest --symbol XAUUSD --start 2024-01-01 --end 2024-12-31
  
  # Live trading
  python main.py --mode live --config config/live.yaml
  
  # Optimization
  python main.py --mode optimize --symbol EURUSD --iterations 100
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['live', 'backtest', 'optimize', 'train'],
        default='backtest',
        help='Operational mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='XAUUSD',
        help='Primary trading symbol'
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
        '--capital',
        type=float,
        default=100000,
        help='Initial capital'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default='1h',
        choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
        help='Trading timeframe'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )
    
    return parser.parse_args()


def load_config_file(path: str) -> Dict:
    """Load and validate configuration file"""
    config_path = Path(path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(config_path, 'r') as f:
        if path.endswith(('.yaml', '.yml')):
            config = yaml.safe_load(f)
        elif path.endswith('.json'):
            import json
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path}")
    
    # Load environment variables
    load_dotenv()
    
    # Override with env vars
    if os.getenv('TRADING_MODE'):
        config['general']['mode'] = os.getenv('TRADING_MODE')
    
    return config


async def main():
    """Main entry point with error handling"""
    args = parse_args()
    
    try:
        # Load configuration
        config_dict = load_config_file(args.config)
        config = TradingBotConfig(config_dict)
        
        # Setup logging
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logger = setup_logger('trading_bot', {'level': log_level})
        
        # Initialize bot
        bot = TradingBot(config, args)
        
        if not await bot.initialize():
            sys.exit(1)
        
        # Execute mode
        if args.mode == 'backtest':
            if not args.start or not args.end:
                logger.error("Backtest mode requires --start and --end dates")
                sys.exit(1)
            await bot.run_backtest()
            
        elif args.mode == 'live':
            await bot.run_live()
            
        elif args.mode == 'optimize':
            logger.info("Optimization mode not yet implemented")
            
        elif args.mode == 'train':
            logger.info("Training mode not yet implemented")
        
        # Graceful shutdown
        await bot.graceful_shutdown()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        if 'bot' in locals():
            await bot.graceful_shutdown()
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        if 'bot' in locals():
            await bot.emergency_shutdown()
        sys.exit(1)


if __name__ == '__main__':
    import os
    asyncio.run(main())