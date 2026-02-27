"""
Backtest engine for Gold Institutional Quant Framework
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import os
from institutional.quant_framework import GoldInstitutionalFramework, MarketState, SessionEvent
from core.strategy_engine import TradingSignal, SignalType, MarketRegime

# Configure logger
logger = logging.getLogger(__name__)


class InstitutionalBacktestEngine:
    """Backtest engine for Gold Institutional Quant Framework"""
    
    def __init__(self, initial_equity: float = 10000, commission: float = 0.001):
        self.initial_equity = initial_equity
        self.commission = commission
        self.equity = initial_equity
        self.framework = GoldInstitutionalFramework()
        self.framework.initialize()
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.max_drawdown = 0
        self.max_equity = initial_equity
        
        # Current position information
        self.current_position = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        
        logger.info("Institutional Backtest Engine initialized")
        
    def run_backtest(self, df: pd.DataFrame, dxy_data: pd.DataFrame, 
                     yield_data: pd.DataFrame, spread: float = 0.3, symbol: str = "XAUUSD") -> dict:
        """
        Run backtest on historical data
        
        Args:
            df: Price data for the trading symbol
            dxy_data: DXY price data
            yield_data: Correlation data based on symbol
            spread: Average spread
            symbol: Trading symbol (e.g., XAUUSD, EURUSD)
            
        Returns:
            Backtest results
        """
        logger.info("Starting backtest...")
        
        # Resample data to 15-minute timeframe
        df = df.resample('15T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Initialize equity tracking
        self.equity_curve = [self.initial_equity]
        self.drawdown_curve = [0]
        
        # Run backtest
        for i in range(48, len(df)):
            # Get current timeframe data
            current_df = df.iloc[:i+1]
            
            # Calculate spread (mock calculation based on time of day)
            time_of_day = current_df.index[-1].time()
            current_spread = spread
            
            # Execute strategy
            result = self.framework.execute_strategy(
                current_df,
                self.equity,
                dxy_data.iloc[:i+1],
                yield_data.iloc[:i+1],
                current_spread,
                False,
                symbol
            )
            
            # Check for entry signal
            if result['should_trade'] and self.current_position == 0:
                self.enter_position(result)
                
            # Manage existing position
            elif self.current_position != 0:
                self.manage_position(current_df)
                
            # Update performance tracking
            self.update_performance_metrics(current_df)
            
        # Final performance summary
        results = self.calculate_final_results()
        
        logger.info("Backtest completed")
        
        return results
        
    def enter_position(self, result: dict) -> None:
        """Enter a new position based on strategy signals"""
        # Set negative position size for short trades
        self.current_position = result['position_size'] if result['direction'] == 'long' else -result['position_size']
        self.entry_price = result['entry_price']
        self.stop_loss = result['stop_loss']
        self.take_profit = result['take_profit']
        self.confidence_score = result['confidence_score']
        
        logger.info(f"Entered {result['direction']} position: "
                   f"Size={abs(self.current_position):.2f}, "
                   f"Entry={self.entry_price:.2f}, "
                   f"SL={self.stop_loss:.2f}, "
                   f"TP={self.take_profit:.2f}, "
                   f"Confidence={self.confidence_score:.2f}")
        
    def manage_position(self, df: pd.DataFrame) -> None:
        """Manage existing position with stop loss and take profit"""
        current_price = df['close'].iloc[-1]
        
        # Calculate trailing stop
        direction = "long" if self.current_position > 0 else "short"
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
            self.close_position(df, exit_reason)
            
    def close_position(self, df: pd.DataFrame, exit_reason: str) -> None:
        """Close current position and calculate trade results"""
        current_price = df['close'].iloc[-1]
        
        # Calculate profit
        if self.current_position > 0:
            profit = (current_price - self.entry_price) * abs(self.current_position)
        else:
            profit = (self.entry_price - current_price) * abs(self.current_position)
            
        # Subtract commission
        commission = abs(self.current_position) * self.commission
        profit -= commission
        
        # Update equity
        self.equity += profit
        
        # Record trade
        self.trades.append({
            'timestamp': df.index[-1],
            'direction': "long" if self.current_position > 0 else "short",
            'entry_price': self.entry_price,
            'exit_price': current_price,
            'position_size': abs(self.current_position),
            'profit': profit,
            'exit_reason': exit_reason,
            'confidence_score': self.confidence_score
        })
        
        logger.info(f"Closed {self.trades[-1]['direction']} position: "
                   f"Exit={current_price:.2f}, "
                   f"Profit={profit:.2f}, "
                   f"Reason={exit_reason}")
        
        # Reset position
        self.current_position = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        
        # Track performance for institutional framework
        self.framework.track_trade_result({'profit': profit})
        
    def update_performance_metrics(self, df: pd.DataFrame) -> None:
        """Update performance metrics during backtest"""
        # Track equity curve
        self.equity_curve.append(self.equity)
        
        # Track maximum equity
        if self.equity > self.max_equity:
            self.max_equity = self.equity
            
        # Calculate drawdown
        drawdown = (self.max_equity - self.equity) / self.max_equity
        self.drawdown_curve.append(drawdown)
        
        # Update maximum drawdown
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            
    def calculate_final_results(self) -> dict:
        """Calculate final performance metrics"""
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate total return
        total_return = (self.equity - self.initial_equity) / self.initial_equity
        
        # Calculate win rate
        if len(trades_df) > 0:
            win_rate = len(trades_df[trades_df['profit'] > 0]) / len(trades_df)
        else:
            win_rate = 0
            
        # Calculate average trade
        if len(trades_df) > 0:
            average_trade = trades_df['profit'].mean()
        else:
            average_trade = 0
            
        # Calculate risk-reward ratio
        if len(trades_df) > 0:
            winning_trades = trades_df[trades_df['profit'] > 0]['profit']
            losing_trades = trades_df[trades_df['profit'] <= 0]['profit'].abs()
            
            if len(winning_trades) > 0 and len(losing_trades) > 0:
                rr_ratio = winning_trades.mean() / losing_trades.mean()
            else:
                rr_ratio = 0
        else:
            rr_ratio = 0
            
        # Calculate Sharpe ratio (assuming risk-free rate = 0.02)
        if len(trades_df) > 0:
            returns = np.array(trades_df['profit']) / self.initial_equity
            volatility = returns.std()
            
            if volatility > 0:
                sharpe_ratio = (total_return - 0.02) / volatility
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
            
        results = {
            'total_return': total_return,
            'win_rate': win_rate,
            'total_trades': len(trades_df),
            'winning_trades': len(trades_df[trades_df['profit'] > 0]) if len(trades_df) > 0 else 0,
            'losing_trades': len(trades_df[trades_df['profit'] <= 0]) if len(trades_df) > 0 else 0,
            'average_trade': average_trade,
            'risk_reward': rr_ratio,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_equity': self.equity,
            'trades': trades_df
        }
        
        return results
        
    def generate_report(self, results: dict, output_dir: str = "reports/institutional") -> None:
        """Generate comprehensive backtest report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate performance chart
        self.plot_performance(results, output_dir)
        
        # Save trade details
        results['trades'].to_csv(os.path.join(output_dir, "institutional_trades.csv"), index=False)
        
        # Save performance summary
        with open(os.path.join(output_dir, "institutional_summary.txt"), 'w') as f:
            f.write("Gold Institutional Quant Framework - Backtest Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Return: {results['total_return']:.2%}\n")
            f.write(f"Final Equity: ${results['final_equity']:.2f}\n")
            f.write(f"Total Trades: {results['total_trades']}\n")
            f.write(f"Win Rate: {results['win_rate']:.2%}\n")
            f.write(f"Winning Trades: {results['winning_trades']}\n")
            f.write(f"Losing Trades: {results['losing_trades']}\n")
            f.write(f"Average Trade: ${results['average_trade']:.2f}\n")
            f.write(f"Risk-Reward Ratio: {results['risk_reward']:.2f}\n")
            f.write(f"Maximum Drawdown: {results['max_drawdown']:.2%}\n")
            f.write(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n")
            
        logger.info(f"Report generated successfully in {output_dir}")
        
    def plot_performance(self, results: dict, output_dir: str) -> None:
        """Plot performance charts"""
        plt.figure(figsize=(12, 8))
        
        # Equity curve
        plt.subplot(2, 1, 1)
        plt.plot(self.equity_curve, label='Equity Curve')
        plt.title('Gold Institutional Quant Framework - Backtest Results')
        plt.ylabel('Equity ($)')
        plt.legend()
        
        # Drawdown
        plt.subplot(2, 1, 2)
        plt.plot([d * 100 for d in self.drawdown_curve], label='Drawdown (%)')
        plt.ylabel('Drawdown (%)')
        plt.xlabel('Trade Number')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "institutional_performance.png"))
        plt.close()


def main():
    """Main function to run backtest with sample data"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=== Gold Institutional Quant Framework Backtest ===")
    
    # Create dummy data
    dates = pd.date_range('2024-01-01', periods=2000, freq='1H')
    
    gold_data = pd.DataFrame({
        'open': [2000 + i * 0.1 + np.random.normal(0, 1) for i in range(2000)],
        'high': [2001 + i * 0.1 + np.random.normal(0, 1.2) for i in range(2000)],
        'low': [1999 + i * 0.1 + np.random.normal(0, 1.2) for i in range(2000)],
        'close': [2000 + i * 0.1 + np.random.normal(0, 1) for i in range(2000)],
        'volume': [10000 + i * 100 + np.random.randint(0, 1000) for i in range(2000)]
    }, index=dates)
    
    dxy_data = pd.DataFrame({
        'close': [103 - i * 0.1 + np.random.normal(0, 0.5) for i in range(2000)]
    }, index=dates)
    
    yield_data = pd.DataFrame({
        'close': [4.2 - i * 0.01 + np.random.normal(0, 0.02) for i in range(2000)]
    }, index=dates)
    
    # Initialize backtest engine
    backtest = InstitutionalBacktestEngine(initial_equity=10000, commission=0.001)
    
    # Run backtest
    results = backtest.run_backtest(gold_data, dxy_data, yield_data, 0.3)
    
    # Print summary
    print(f"\nTotal Return: {results['total_return']:.2%}")
    print(f"Final Equity: ${results['final_equity']:.2f}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Average Trade: ${results['average_trade']:.2f}")
    print(f"Risk-Reward Ratio: {results['risk_reward']:.2f}")
    print(f"Maximum Drawdown: {results['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    
    # Generate report
    backtest.generate_report(results)
    
    print("\nBacktest completed successfully. Report saved to 'reports/institutional'")
    
    return results


if __name__ == "__main__":
    main()
