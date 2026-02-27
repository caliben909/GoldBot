"""
Backtest engine for Gold Institutional Quant Framework
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import os
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Data class for storing trade information"""
    timestamp: datetime
    direction: str
    entry_price: float
    exit_price: float
    position_size: float
    profit: float
    exit_reason: str
    confidence_score: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class BacktestResults:
    """Data class for backtest results"""
    total_return: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_trade: float
    risk_reward: float
    max_drawdown: float
    sharpe_ratio: float
    final_equity: float
    trades: pd.DataFrame
    equity_curve: List[float]
    drawdown_curve: List[float]
    
    def print_summary(self) -> None:
        """Print formatted summary"""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total Return:      {self.total_return:>10.2%}")
        print(f"Final Equity:      ${self.final_equity:>10,.2f}")
        print(f"Total Trades:      {self.total_trades:>10}")
        print(f"Win Rate:          {self.win_rate:>10.2%}")
        print(f"Winning Trades:    {self.winning_trades:>10}")
        print(f"Losing Trades:     {self.losing_trades:>10}")
        print(f"Average Trade:     ${self.average_trade:>10,.2f}")
        print(f"Risk-Reward Ratio: {self.risk_reward:>10.2f}")
        print(f"Max Drawdown:      {self.max_drawdown:>10.2%}")
        print(f"Sharpe Ratio:      {self.sharpe_ratio:>10.2f}")
        print("=" * 60)


class Position(Enum):
    """Position direction enum"""
    NONE = 0
    LONG = 1
    SHORT = -1


class InstitutionalBacktestEngine:
    """
    Backtest engine for Gold Institutional Quant Framework
    
    This engine simulates trading strategies with realistic constraints including
    spreads, commissions, and position management.
    """
    
    def __init__(self, initial_equity: float = 10000, commission: float = 0.001):
        """
        Initialize the backtest engine
        
        Args:
            initial_equity: Starting capital
            commission: Commission rate as decimal (e.g., 0.001 = 0.1%)
        
        Raises:
            ValueError: If invalid parameters are provided
        """
        if initial_equity <= 0:
            raise ValueError(f"Initial equity must be positive, got {initial_equity}")
        if commission < 0:
            raise ValueError(f"Commission cannot be negative, got {commission}")
            
        self.initial_equity = initial_equity
        self.commission = commission
        self.framework = None  # Will be initialized when needed
        
        # Performance tracking
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = []
        self.drawdown_curve: List[float] = []
        self.max_drawdown: float = 0
        self.max_equity: float = initial_equity
        
        # Current position information
        self.current_position: Position = Position.NONE
        self.position_size: float = 0
        self.entry_price: float = 0
        self.stop_loss: float = 0
        self.take_profit: float = 0
        self.confidence_score: float = 0
        
        logger.info(f"Institutional Backtest Engine initialized (Equity: ${initial_equity:,.2f})")
        
    def _initialize_framework(self) -> None:
        """Lazy initialization of the trading framework"""
        try:
            from institutional.quant_framework import GoldInstitutionalFramework
            self.framework = GoldInstitutionalFramework()
            self.framework.initialize()
            logger.debug("Trading framework initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import trading framework: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize trading framework: {e}")
            raise
            
    def _validate_input_data(self, df: pd.DataFrame, dxy_data: pd.DataFrame, 
                            yield_data: pd.DataFrame, symbol: str) -> None:
        """
        Validate input data for backtesting
        
        Args:
            df: Price data
            dxy_data: DXY data
            yield_data: Yield/correlation data
            symbol: Trading symbol
            
        Raises:
            ValueError: If data is invalid
        """
        if df.empty:
            raise ValueError("Price data is empty")
            
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in price data: {missing_cols}")
            
        if dxy_data.empty:
            raise ValueError("DXY data is empty")
            
        if yield_data.empty:
            raise ValueError("Yield data is empty")
            
        # Check index alignment (simplified)
        if df.index[-1] != dxy_data.index[-1] or df.index[-1] != yield_data.index[-1]:
            logger.warning("Data indices may not be properly aligned")
            
        logger.info(f"Validated data for {symbol}: {len(df)} candles")
        
    def _resample_data(self, df: pd.DataFrame, timeframe: str = '15T') -> pd.DataFrame:
        """
        Resample data to specified timeframe
        
        Args:
            df: Original data
            timeframe: Target timeframe (e.g., '15T' for 15 minutes)
            
        Returns:
            Resampled DataFrame
        """
        try:
            resampled = df.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            logger.debug(f"Resampled data from {len(df)} to {len(resampled)} candles")
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling data: {e}")
            return df
            
    def _get_spread_adjustment(self, timestamp: datetime, base_spread: float) -> float:
        """
        Calculate spread adjustment based on time of day
        
        Args:
            timestamp: Current timestamp
            base_spread: Base spread value
            
        Returns:
            Adjusted spread
        """
        hour = timestamp.hour
        
        # Higher spreads during off-hours
        if hour < 1 or hour > 22:  # Midnight - 1 AM or 10 PM - midnight
            return base_spread * 1.5
        elif 1 <= hour <= 4:  # Early Asian session
            return base_spread * 1.3
        elif 13 <= hour <= 16:  # London/NY overlap
            return base_spread * 0.8
        else:
            return base_spread
            
    def run_backtest(self, df: pd.DataFrame, dxy_data: pd.DataFrame, 
                    yield_data: pd.DataFrame, spread: float = 0.3, 
                    symbol: str = "XAUUSD", timeframe: str = '15T') -> BacktestResults:
        """
        Run backtest on historical data
        
        Args:
            df: Price data for the trading symbol
            dxy_data: DXY price data
            yield_data: Correlation data based on symbol
            spread: Average spread
            symbol: Trading symbol (e.g., XAUUSD, EURUSD)
            timeframe: Resampling timeframe
            
        Returns:
            BacktestResults object with performance metrics
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If backtest fails
        """
        logger.info(f"Starting backtest for {symbol}...")
        
        try:
            # Initialize framework if needed
            if self.framework is None:
                self._initialize_framework()
                
            # Validate input data
            self._validate_input_data(df, dxy_data, yield_data, symbol)
            
            # Resample data
            df = self._resample_data(df, timeframe)
            
            if len(df) < 50:
                raise ValueError(f"Insufficient data after resampling: {len(df)} candles")
            
            # Reset tracking variables
            self.equity = self.initial_equity
            self.max_equity = self.initial_equity
            self.trades = []
            self.equity_curve = [self.initial_equity]
            self.drawdown_curve = [0]
            self.current_position = Position.NONE
            
            # Run backtest
            for i in range(48, len(df)):  # Start after sufficient history
                self._process_candle(df, dxy_data, yield_data, i, spread, symbol)
                
            # Calculate final results
            results = self._calculate_results()
            
            logger.info(f"Backtest completed: {results.total_trades} trades, "
                       f"Return: {results.total_return:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise RuntimeError(f"Backtest execution failed: {e}") from e
            
    def _process_candle(self, df: pd.DataFrame, dxy_data: pd.DataFrame, 
                       yield_data: pd.DataFrame, idx: int, spread: float, 
                       symbol: str) -> None:
        """
        Process a single candle in the backtest
        
        Args:
            df: Full price dataframe
            dxy_data: DXY data
            yield_data: Yield data
            idx: Current index
            spread: Base spread
            symbol: Trading symbol
        """
        # Get data slice up to current point
        current_df = df.iloc[:idx+1]
        
        # Calculate spread based on time
        current_spread = self._get_spread_adjustment(current_df.index[-1], spread)
        
        try:
            # Execute strategy
            result = self.framework.execute_strategy(
                current_df,
                self.equity,
                dxy_data.iloc[:idx+1],
                yield_data.iloc[:idx+1],
                current_spread,
                False,  # Not in live trading
                symbol
            )
            
            # Check for entry signal
            if result.get('should_trade', False) and self.current_position == Position.NONE:
                self._enter_position(result)
                
            # Manage existing position
            elif self.current_position != Position.NONE:
                self._manage_position(current_df)
                
            # Update performance tracking
            self._update_performance_metrics(current_df)
            
        except Exception as e:
            logger.error(f"Error processing candle at {df.index[idx]}: {e}")
            
    def _enter_position(self, result: Dict[str, Any]) -> None:
        """
        Enter a new position based on strategy signals
        
        Args:
            result: Strategy execution result
        """
        try:
            direction = result.get('direction', 'long')
            position_size = result.get('position_size', 0)
            
            if position_size <= 0:
                logger.warning(f"Invalid position size: {position_size}")
                return
                
            self.current_position = Position.LONG if direction == 'long' else Position.SHORT
            self.position_size = position_size if direction == 'long' else -position_size
            self.entry_price = result.get('entry_price', 0)
            self.stop_loss = result.get('stop_loss', 0)
            self.take_profit = result.get('take_profit', 0)
            self.confidence_score = result.get('confidence_score', 0)
            
            logger.info(f"Entered {direction.upper()} position: "
                       f"Size={abs(self.position_size):.2f}, "
                       f"Entry={self.entry_price:.2f}, "
                       f"SL={self.stop_loss:.2f}, "
                       f"TP={self.take_profit:.2f}, "
                       f"Confidence={self.confidence_score:.2f}")
                       
        except Exception as e:
            logger.error(f"Error entering position: {e}")
            self.current_position = Position.NONE
            
    def _manage_position(self, df: pd.DataFrame) -> None:
        """
        Manage existing position with stop loss and take profit
        
        Args:
            df: Current dataframe slice
        """
        if self.current_position == Position.NONE:
            return
            
        try:
            current_price = df['close'].iloc[-1]
            
            # Determine direction
            direction = "long" if self.current_position == Position.LONG else "short"
            
            # Calculate trailing stop
            trailing_stop = self.framework.calculate_trailing_stop(
                current_price,
                self.entry_price,
                direction,
                self.stop_loss
            )
            
            # Check exit conditions
            exit_reason = self.framework.should_exit_trade(
                current_price,
                self.take_profit,
                trailing_stop,
                direction
            )
            
            if exit_reason in ["take_profit", "stop_loss"]:
                self._close_position(df, exit_reason)
                
        except Exception as e:
            logger.error(f"Error managing position: {e}")
            
    def _close_position(self, df: pd.DataFrame, exit_reason: str) -> None:
        """
        Close current position and calculate trade results
        
        Args:
            df: Current dataframe slice
            exit_reason: Reason for exit
        """
        if self.current_position == Position.NONE:
            return
            
        try:
            current_price = df['close'].iloc[-1]
            
            # Calculate profit
            if self.current_position == Position.LONG:
                profit = (current_price - self.entry_price) * abs(self.position_size)
            else:
                profit = (self.entry_price - current_price) * abs(self.position_size)
                
            # Subtract commission
            commission_cost = abs(self.position_size) * self.commission
            profit -= commission_cost
            
            # Update equity
            self.equity += profit
            
            # Record trade
            trade = TradeRecord(
                timestamp=df.index[-1],
                direction="long" if self.current_position == Position.LONG else "short",
                entry_price=self.entry_price,
                exit_price=current_price,
                position_size=abs(self.position_size),
                profit=profit,
                exit_reason=exit_reason,
                confidence_score=self.confidence_score
            )
            self.trades.append(trade)
            
            logger.info(f"Closed {trade.direction.upper()} position: "
                       f"Exit={current_price:.2f}, "
                       f"Profit={profit:.2f}, "
                       f"Reason={exit_reason}")
            
            # Track trade result in framework
            if hasattr(self.framework, 'track_trade_result'):
                self.framework.track_trade_result({'profit': profit})
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            
        finally:
            # Reset position regardless of errors
            self.current_position = Position.NONE
            self.position_size = 0
            self.entry_price = 0
            self.stop_loss = 0
            self.take_profit = 0
            
    def _update_performance_metrics(self, df: pd.DataFrame) -> None:
        """
        Update performance metrics during backtest
        
        Args:
            df: Current dataframe slice
        """
        try:
            # Track equity curve
            self.equity_curve.append(self.equity)
            
            # Track maximum equity
            if self.equity > self.max_equity:
                self.max_equity = self.equity
                
            # Calculate drawdown
            drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0
            self.drawdown_curve.append(drawdown)
            
            # Update maximum drawdown
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
                
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
            
    def _calculate_results(self) -> BacktestResults:
        """
        Calculate final performance metrics
        
        Returns:
            BacktestResults object
        """
        try:
            # Convert trades to DataFrame
            if self.trades:
                trades_df = pd.DataFrame([t.to_dict() for t in self.trades])
            else:
                trades_df = pd.DataFrame()
            
            # Calculate total return
            total_return = (self.equity - self.initial_equity) / self.initial_equity if self.initial_equity > 0 else 0
            
            # Calculate metrics based on trades
            if not trades_df.empty and len(trades_df) > 0:
                winning_trades = len(trades_df[trades_df['profit'] > 0])
                losing_trades = len(trades_df[trades_df['profit'] <= 0])
                win_rate = winning_trades / len(trades_df) if len(trades_df) > 0 else 0
                average_trade = trades_df['profit'].mean()
                
                # Risk-reward ratio
                winning_mean = trades_df[trades_df['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
                losing_mean = abs(trades_df[trades_df['profit'] <= 0]['profit'].mean()) if losing_trades > 0 else 1
                rr_ratio = winning_mean / losing_mean if losing_mean > 0 else 0
                
                # Sharpe ratio (simplified)
                returns = trades_df['profit'].values / self.initial_equity
                volatility = returns.std() if len(returns) > 1 else 0
                sharpe_ratio = (total_return - 0.02) / volatility if volatility > 0 else 0
            else:
                winning_trades = losing_trades = 0
                win_rate = average_trade = rr_ratio = sharpe_ratio = 0
                
            return BacktestResults(
                total_return=total_return,
                win_rate=win_rate,
                total_trades=len(trades_df),
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                average_trade=average_trade,
                risk_reward=rr_ratio,
                max_drawdown=self.max_drawdown,
                sharpe_ratio=sharpe_ratio,
                final_equity=self.equity,
                trades=trades_df,
                equity_curve=self.equity_curve,
                drawdown_curve=self.drawdown_curve
            )
            
        except Exception as e:
            logger.error(f"Error calculating results: {e}")
            # Return empty results
            return BacktestResults(
                total_return=0, win_rate=0, total_trades=0,
                winning_trades=0, losing_trades=0, average_trade=0,
                risk_reward=0, max_drawdown=0, sharpe_ratio=0,
                final_equity=self.equity, trades=pd.DataFrame(),
                equity_curve=self.equity_curve, drawdown_curve=self.drawdown_curve
            )
            
    def generate_report(self, results: BacktestResults, output_dir: str = "reports/institutional") -> None:
        """
        Generate comprehensive backtest report
        
        Args:
            results: Backtest results
            output_dir: Directory to save reports
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Print summary to console
            results.print_summary()
            
            # Generate performance chart
            self._plot_performance(results, output_dir)
            
            # Save trade details
            if not results.trades.empty:
                trade_file = os.path.join(output_dir, "institutional_trades.csv")
                results.trades.to_csv(trade_file, index=False)
                logger.info(f"Trade details saved to {trade_file}")
            else:
                logger.warning("No trades to save")
                
            # Save performance summary
            summary_file = os.path.join(output_dir, "institutional_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("Gold Institutional Quant Framework - Backtest Results\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Initial Equity: ${self.initial_equity:,.2f}\n")
                f.write(f"Final Equity: ${results.final_equity:,.2f}\n")
                f.write(f"Total Return: {results.total_return:.2%}\n")
                f.write(f"Total Trades: {results.total_trades}\n")
                f.write(f"Win Rate: {results.win_rate:.2%}\n")
                f.write(f"Winning Trades: {results.winning_trades}\n")
                f.write(f"Losing Trades: {results.losing_trades}\n")
                f.write(f"Average Trade: ${results.average_trade:.2f}\n")
                f.write(f"Risk-Reward Ratio: {results.risk_reward:.2f}\n")
                f.write(f"Maximum Drawdown: {results.max_drawdown:.2%}\n")
                f.write(f"Sharpe Ratio: {results.sharpe_ratio:.2f}\n")
                
            logger.info(f"Report generated successfully in {output_dir}")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            
    def _plot_performance(self, results: BacktestResults, output_dir: str) -> None:
        """
        Plot performance charts
        
        Args:
            results: Backtest results
            output_dir: Directory to save charts
        """
        try:
            plt.figure(figsize=(14, 10))
            
            # Equity curve
            plt.subplot(3, 1, 1)
            plt.plot(results.equity_curve, linewidth=1.5, color='blue')
            plt.axhline(y=self.initial_equity, color='gray', linestyle='--', alpha=0.7)
            plt.title(f'Equity Curve (Final: ${results.final_equity:,.2f}, Return: {results.total_return:.2%})')
            plt.ylabel('Equity ($)')
            plt.grid(True, alpha=0.3)
            
            # Drawdown
            plt.subplot(3, 1, 2)
            drawdown_pct = [d * 100 for d in results.drawdown_curve]
            plt.fill_between(range(len(drawdown_pct)), drawdown_pct, color='red', alpha=0.3)
            plt.plot(drawdown_pct, color='red', linewidth=1)
            plt.title(f'Drawdown (Max: {results.max_drawdown:.2%})')
            plt.ylabel('Drawdown (%)')
            plt.grid(True, alpha=0.3)
            
            # Trade profit distribution
            plt.subplot(3, 1, 3)
            if not results.trades.empty:
                profits = results.trades['profit'].values
                colors = ['green' if p > 0 else 'red' for p in profits]
                plt.bar(range(len(profits)), profits, color=colors, alpha=0.7)
                plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                plt.title(f'Trade Profit Distribution (Win Rate: {results.win_rate:.2%})')
                plt.ylabel('Profit ($)')
                plt.xlabel('Trade Number')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No trades executed', 
                        horizontalalignment='center', verticalalignment='center')
                
            plt.tight_layout()
            chart_file = os.path.join(output_dir, "institutional_performance.png")
            plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.debug(f"Performance chart saved to {chart_file}")
            
        except Exception as e:
            logger.error(f"Error plotting performance: {e}")


def generate_sample_data(days: int = 90, freq: str = '1H') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate sample data for testing
    
    Args:
        days: Number of days of data
        freq: Frequency of data
        
    Returns:
        Tuple of (gold_data, dxy_data, yield_data)
    """
    periods = days * 24  # Approximate
    dates = pd.date_range(datetime.now() - timedelta(days=days), periods=periods, freq=freq)
    
    np.random.seed(42)  # For reproducibility
    
    # Gold data with trend and volatility
    gold_base = 2000
    gold_trend = np.linspace(0, 50, periods)  # Upward trend
    gold_noise = np.random.normal(0, 5, periods)
    gold_prices = gold_base + gold_trend + gold_noise
    
    gold_data = pd.DataFrame({
        'open': gold_prices + np.random.normal(0, 1, periods),
        'high': gold_prices + np.random.normal(2, 2, periods),
        'low': gold_prices - np.random.normal(2, 2, periods),
        'close': gold_prices + np.random.normal(0, 1, periods),
        'volume': np.random.randint(5000, 15000, periods)
    }, index=dates)
    
    # DXY data (inverse correlation with gold)
    dxy_base = 103
    dxy_trend = -gold_trend * 0.1  # Inverse relationship
    dxy_noise = np.random.normal(0, 0.5, periods)
    dxy_data = pd.DataFrame({
        'close': dxy_base + dxy_trend + dxy_noise
    }, index=dates)
    
    # Yield data
    yield_base = 4.2
    yield_trend = np.linspace(0, -0.5, periods)  # Slight downward trend
    yield_noise = np.random.normal(0, 0.1, periods)
    yield_data = pd.DataFrame({
        'close': yield_base + yield_trend + yield_noise
    }, index=dates)
    
    return gold_data, dxy_data, yield_data


def main():
    """Main function to run backtest with sample data"""
    print("\n" + "=" * 60)
    print("GOLD INSTITUTIONAL QUANT FRAMEWORK BACKTEST")
    print("=" * 60)
    
    try:
        # Generate sample data
        print("\nGenerating sample data...")
        gold_data, dxy_data, yield_data = generate_sample_data(days=90)
        print(f"Generated {len(gold_data)} candles of data")
        
        # Initialize backtest engine
        print("\nInitializing backtest engine...")
        backtest = InstitutionalBacktestEngine(initial_equity=10000, commission=0.001)
        
        # Run backtest
        print("\nRunning backtest...")
        results = backtest.run_backtest(
            gold_data, 
            dxy_data, 
            yield_data, 
            spread=0.3,
            symbol="XAUUSD"
        )
        
        # Generate report
        print("\nGenerating report...")
        backtest.generate_report(results)
        
        print("\n✅ Backtest completed successfully!")
        print(f"📊 Report saved to 'reports/institutional'")
        
        return results
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        print(f"\n❌ Backtest failed: {e}")
        return None


if __name__ == "__main__":
    main()