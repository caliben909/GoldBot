"""
Institutional Trade Analyzer - Production-Grade Performance Analytics
Generates comprehensive trade reports with risk metrics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging
from enum import Enum

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradeResult(Enum):
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


@dataclass
class TradeMetrics:
    """Comprehensive trade performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float
    loss_rate: float
    avg_profit: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    gross_profit: float
    gross_loss: float
    net_profit: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_trade_duration: Optional[timedelta]
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    max_drawdown_pct: float
    recovery_factor: float
    calmar_ratio: float
    risk_reward_avg: float


class TradeAnalyzer:
    """
    Institutional-grade trade analysis with comprehensive risk metrics
    """
    
    REQUIRED_COLUMNS = {
        'direction', 'profit', 'entry_price', 'exit_price', 
        'entry_time', 'exit_time', 'stop_loss', 'take_profit'
    }
    
    OPTIONAL_COLUMNS = {
        'symbol', 'position_size', 'risk_amount', 'strategy', 
        'session', 'confidence_score', 'macro_score'
    }
    
    def __init__(self, csv_path: str, output_dir: str = "reports/analysis"):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.df: Optional[pd.DataFrame] = None
        self.metrics: Optional[TradeMetrics] = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TradeAnalyzer initialized: {csv_path}")
    
    def load_data(self) -> bool:
        """Load and validate trade data with error handling"""
        try:
            if not self.csv_path.exists():
                logger.error(f"File not found: {self.csv_path}")
                return False
            
            # Read CSV with flexible parsing
            self.df = pd.read_csv(
                self.csv_path,
                parse_dates=['entry_time', 'exit_time'] if 'entry_time' in pd.read_csv(self.csv_path, nrows=0).columns else False
            )
            
            if self.df.empty:
                logger.error("CSV file is empty")
                return False
            
            # Validate required columns
            missing = self.REQUIRED_COLUMNS - set(self.df.columns)
            if missing:
                logger.error(f"Missing required columns: {missing}")
                return False
            
            # Data type conversions
            self.df['profit'] = pd.to_numeric(self.df['profit'], errors='coerce')
            self.df['entry_price'] = pd.to_numeric(self.df['entry_price'], errors='coerce')
            self.df['exit_price'] = pd.to_numeric(self.df['exit_price'], errors='coerce')
            
            # Parse dates if string
            for col in ['entry_time', 'exit_time']:
                if col in self.df.columns and self.df[col].dtype == 'object':
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            
            # Calculate derived metrics
            self._calculate_derived_metrics()
            
            logger.info(f"Loaded {len(self.df)} trades")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            return False
    
    def _calculate_derived_metrics(self):
        """Calculate additional trade metrics"""
        # Trade duration
        if 'entry_time' in self.df.columns and 'exit_time' in self.df.columns:
            self.df['duration'] = self.df['exit_time'] - self.df['entry_time']
            self.df['duration_minutes'] = self.df['duration'].dt.total_seconds() / 60
        
        # Trade result classification
        self.df['result'] = self.df['profit'].apply(
            lambda x: TradeResult.WIN if x > 0 else (TradeResult.LOSS if x < 0 else TradeResult.BREAKEVEN)
        )
        
        # Calculate R-multiple (profit relative to initial risk)
        if 'stop_loss' in self.df.columns and 'entry_price' in self.df.columns:
            self.df['initial_risk'] = abs(self.df['entry_price'] - self.df['stop_loss'])
            self.df['r_multiple'] = self.df['profit'] / self.df['initial_risk'].replace(0, np.nan)
        
        # Cumulative equity curve (assuming fixed starting capital or sequential)
        self.df = self.df.sort_values('exit_time' if 'exit_time' in self.df.columns else self.df.index)
        self.df['cumulative_pnl'] = self.df['profit'].cumsum()
        self.df['peak'] = self.df['cumulative_pnl'].cummax()
        self.df['drawdown'] = self.df['cumulative_pnl'] - self.df['peak']
        self.df['drawdown_pct'] = (self.df['drawdown'] / (self.df['peak'].abs() + 10000)) * 100  # Assume 10k base
        
        # Monthly/weekly groupings
        if 'exit_time' in self.df.columns:
            self.df['year_month'] = self.df['exit_time'].dt.to_period('M')
            self.df['year_week'] = self.df['exit_time'].dt.to_period('W')
    
    def calculate_metrics(self) -> Optional[TradeMetrics]:
        """Calculate comprehensive performance metrics"""
        if self.df is None or self.df.empty:
            return None
        
        try:
            profits = self.df['profit']
            wins = profits[profits > 0]
            losses = profits[profits < 0]
            breakeven = profits[profits == 0]
            
            total_trades = len(profits)
            winning_trades = len(wins)
            losing_trades = len(losses)
            breakeven_trades = len(breakeven)
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            loss_rate = losing_trades / total_trades if total_trades > 0 else 0
            
            gross_profit = wins.sum() if len(wins) > 0 else 0
            gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
            net_profit = gross_profit - gross_loss
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = losses.mean() if len(losses) > 0 else 0
            expectancy = (win_rate * avg_win) - (loss_rate * abs(avg_loss)) if total_trades > 0 else 0
            
            # Consecutive streaks
            results = self.df['result'].tolist()
            max_consecutive_wins = self._max_consecutive(results, TradeResult.WIN)
            max_consecutive_losses = self._max_consecutive(results, TradeResult.LOSS)
            
            # Duration
            avg_duration = self.df['duration'].mean() if 'duration' in self.df.columns else None
            
            # Risk metrics
            returns = profits.pct_change().dropna() if 'entry_price' in self.df.columns else profits
            sharpe = self._calculate_sharpe(returns)
            sortino = self._calculate_sortino(returns)
            
            max_dd_pct = self.df['drawdown_pct'].min() if 'drawdown_pct' in self.df.columns else 0
            recovery_factor = abs(net_profit / max_dd_pct) if max_dd_pct != 0 else float('inf')
            calmar = (net_profit / 12) / abs(max_dd_pct) if max_dd_pct != 0 else 0  # Monthly return / max DD
            
            # R-multiple stats
            risk_reward_avg = self.df['r_multiple'].mean() if 'r_multiple' in self.df.columns else 0
            
            self.metrics = TradeMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                breakeven_trades=breakeven_trades,
                win_rate=win_rate,
                loss_rate=loss_rate,
                avg_profit=profits.mean(),
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                expectancy=expectancy,
                gross_profit=gross_profit,
                gross_loss=gross_loss,
                net_profit=net_profit,
                max_consecutive_wins=max_consecutive_wins,
                max_consecutive_losses=max_consecutive_losses,
                avg_trade_duration=avg_duration,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown_pct=max_dd_pct,
                recovery_factor=recovery_factor,
                calmar_ratio=calmar,
                risk_reward_avg=risk_reward_avg
            )
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}", exc_info=True)
            return None
    
    def _max_consecutive(self, results: List[TradeResult], target: TradeResult) -> int:
        """Calculate maximum consecutive occurrences"""
        max_count = 0
        current = 0
        
        for r in results:
            if r == target:
                current += 1
                max_count = max(max_count, current)
            else:
                current = 0
        
        return max_count
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.0) -> Optional[float]:
        """Calculate annualized Sharpe ratio"""
        if len(returns) < 2 or returns.std() == 0:
            return None
        
        # Assuming daily returns, annualize
        excess_returns = returns - risk_free_rate
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)
    
    def _calculate_sortino(self, returns: pd.Series, risk_free_rate: float = 0.0) -> Optional[float]:
        """Calculate Sortino ratio (downside deviation only)"""
        if len(returns) < 2:
            return None
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if returns.mean() > 0 else 0
        
        downside_std = downside_returns.std()
        return (returns.mean() - risk_free_rate) / downside_std * np.sqrt(252)
    
    def generate_text_report(self) -> str:
        """Generate formatted text report"""
        if not self.metrics:
            return "No metrics available"
        
        m = self.metrics
        
        report = f"""
{'='*60}
INSTITUTIONAL TRADE PERFORMANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

EXECUTIVE SUMMARY
{'-'*60}
Total Trades:           {m.total_trades:,}
Net Profit:             ${m.net_profit:>12,.2f}
Win Rate:               {m.win_rate*100:>11.1f}%
Profit Factor:          {m.profit_factor:>12.2f}
Expectancy per Trade:   ${m.expectancy:>12,.2f}

TRADE DISTRIBUTION
{'-'*60}
Winning Trades:         {m.winning_trades:,} ({m.win_rate*100:.1f}%)
Losing Trades:          {m.losing_trades:,} ({m.loss_rate*100:.1f}%)
Breakeven Trades:       {m.breakeven_trades:,}

PROFITABILITY METRICS
{'-'*60}
Gross Profit:           ${m.gross_profit:>12,.2f}
Gross Loss:             ${m.gross_loss:>12,.2f}
Average Win:            ${m.avg_win:>12,.2f}
Average Loss:           ${m.avg_loss:>12,.2f}
Average Trade:          ${m.avg_profit:>12,.2f}

RISK METRICS
{'-'*60}
Max Drawdown:           {m.max_drawdown_pct:>11.2f}%
Recovery Factor:        {m.recovery_factor:>12.2f}
Calmar Ratio:           {m.calmar_ratio:>12.2f}
Sharpe Ratio:           {m.sharpe_ratio if m.sharpe_ratio else 0:>12.2f}
Sortino Ratio:          {m.sortino_ratio if m.sortino_ratio else 0:>12.2f}
Avg R-Multiple:         {m.risk_reward_avg:>12.2f}R

STREAK ANALYSIS
{'-'*60}
Max Consecutive Wins:   {m.max_consecutive_wins}
Max Consecutive Losses: {m.max_consecutive_losses}
"""
        
        # Directional breakdown
        if 'direction' in self.df.columns:
            report += self._directional_analysis()
        
        # Time-based analysis
        if 'year_month' in self.df.columns:
            report += self._time_analysis()
        
        report += f"\n{'='*60}\n"
        return report
    
    def _directional_analysis(self) -> str:
        """Analyze long vs short performance"""
        report = "\nDIRECTIONAL PERFORMANCE\n" + "-"*60 + "\n"
        
        for direction in ['long', 'short']:
            dir_df = self.df[self.df['direction'] == direction]
            if len(dir_df) == 0:
                continue
            
            dir_profit = dir_df['profit'].sum()
            dir_wins = len(dir_df[dir_df['profit'] > 0])
            dir_win_rate = dir_wins / len(dir_df) * 100 if len(dir_df) > 0 else 0
            
            report += f"{direction.upper():<8} | Trades: {len(dir_df):>3} | "
            report += f"Profit: ${dir_profit:>10,.2f} | Win Rate: {dir_win_rate:>5.1f}%\n"
        
        return report
    
    def _time_analysis(self) -> str:
        """Analyze performance by time period"""
        report = "\nMONTHLY PERFORMANCE\n" + "-"*60 + "\n"
        
        monthly = self.df.groupby('year_month').agg({
            'profit': ['sum', 'count', 'mean'],
            'result': lambda x: (x == TradeResult.WIN).sum() / len(x) * 100 if len(x) > 0 else 0
        }).round(2)
        
        monthly.columns = ['Profit', 'Trades', 'Avg Trade', 'Win Rate %']
        
        for period, row in monthly.iterrows():
            report += f"{period} | ${row['Profit']:>10,.2f} | {int(row['Trades']):>3} trades | "
            report += f"{row['Win Rate %']:>5.1f}% wins | Avg: ${row['Avg Trade']:>8,.2f}\n"
        
        return report
    
    def generate_charts(self) -> List[Path]:
        """Generate visualization charts"""
        if self.df is None or self.df.empty:
            return []
        
        chart_paths = []
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Equity Curve with Drawdown
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        if 'exit_time' in self.df.columns:
            x_vals = self.df['exit_time']
        else:
            x_vals = range(len(self.df))
        
        # Equity curve
        ax1.plot(x_vals, self.df['cumulative_pnl'], linewidth=2, label='Equity Curve')
        ax1.fill_between(x_vals, self.df['cumulative_pnl'], alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_title('Equity Curve & Drawdown', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative P&L ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2.fill_between(x_vals, self.df['drawdown'], color='red', alpha=0.5)
        ax2.set_ylabel('Drawdown ($)')
        ax2.set_xlabel('Trade Number' if 'exit_time' not in self.df.columns else 'Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = self.output_dir / 'equity_curve.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        chart_paths.append(path)
        plt.close()
        
        # 2. Trade Distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Profit distribution
        ax = axes[0, 0]
        profits = self.df['profit']
        ax.hist(profits, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(profits.mean(), color='red', linestyle='--', label=f'Mean: ${profits.mean():.2f}')
        ax.set_title('Profit Distribution')
        ax.set_xlabel('Profit ($)')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Win/Loss by direction
        ax = axes[0, 1]
        if 'direction' in self.df.columns:
            direction_profit = self.df.groupby('direction')['profit'].sum()
            colors = ['green' if x > 0 else 'red' for x in direction_profit]
            direction_profit.plot(kind='bar', ax=ax, color=colors)
            ax.set_title('P&L by Direction')
            ax.set_ylabel('Profit ($)')
            ax.tick_params(axis='x', rotation=0)
        
        # R-Multiple distribution
        ax = axes[1, 0]
        if 'r_multiple' in self.df.columns:
            r_vals = self.df['r_multiple'].dropna()
            ax.hist(r_vals, bins=20, edgecolor='black', alpha=0.7, color='purple')
            ax.axvline(r_vals.mean(), color='red', linestyle='--', label=f'Avg: {r_vals.mean():.2f}R')
            ax.set_title('R-Multiple Distribution')
            ax.set_xlabel('R-Multiple')
            ax.legend()
        
        # Monthly performance
        ax = axes[1, 1]
        if 'year_month' in self.df.columns:
            monthly_pnl = self.df.groupby('year_month')['profit'].sum()
            colors = ['green' if x > 0 else 'red' for x in monthly_pnl]
            monthly_pnl.plot(kind='bar', ax=ax, color=colors)
            ax.set_title('Monthly P&L')
            ax.set_ylabel('Profit ($)')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        path = self.output_dir / 'trade_distribution.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        chart_paths.append(path)
        plt.close()
        
        # 3. Advanced Analytics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Cumulative win rate
        ax = axes[0, 0]
        self.df['cumulative_win_rate'] = (self.df['result'] == TradeResult.WIN).expanding().mean() * 100
        ax.plot(x_vals, self.df['cumulative_win_rate'], linewidth=2, color='blue')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Cumulative Win Rate %')
        ax.set_ylabel('Win Rate %')
        ax.set_ylim(0, 100)
        
        # Trade duration vs profit
        ax = axes[0, 1]
        if 'duration_minutes' in self.df.columns:
            scatter = ax.scatter(self.df['duration_minutes'], self.df['profit'], 
                             c=self.df['profit'] > 0, cmap='RdYlGn', alpha=0.6)
            ax.set_title('Trade Duration vs Profit')
            ax.set_xlabel('Duration (minutes)')
            ax.set_ylabel('Profit ($)')
        
        # Rolling Sharpe (20-trade window)
        ax = axes[1, 0]
        if len(self.df) >= 20:
            rolling_returns = self.df['profit'].rolling(20).mean() / self.df['profit'].rolling(20).std()
            ax.plot(x_vals, rolling_returns, linewidth=2, color='purple')
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Sharpe=1')
            ax.set_title('20-Trade Rolling Sharpe')
            ax.legend()
        
        # Drawdown duration
        ax = axes[1, 1]
        underwater = self.df['drawdown'] < 0
        ax.fill_between(x_vals, self.df['drawdown'], where=underwater, color='red', alpha=0.5)
        ax.set_title('Underwater Periods')
        ax.set_ylabel('Drawdown ($)')
        
        plt.tight_layout()
        path = self.output_dir / 'advanced_analytics.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        chart_paths.append(path)
        plt.close()
        
        logger.info(f"Generated {len(chart_paths)} charts")
        return chart_paths
    
    def export_json(self) -> Path:
        """Export metrics to JSON"""
        if not self.metrics:
            return None
        
        data = {
            'generated_at': datetime.now().isoformat(),
            'file_analyzed': str(self.csv_path),
            'metrics': {
                k: (v.isoformat() if isinstance(v, timedelta) else v)
                for k, v in self.metrics.__dict__.items()
            }
        }
        
        path = self.output_dir / 'metrics.json'
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return path
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Execute complete analysis pipeline"""
        results = {
            'success': False,
            'report': None,
            'charts': [],
            'json_export': None,
            'errors': []
        }
        
        # Load data
        if not self.load_data():
            results['errors'].append("Failed to load data")
            return results
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        if not metrics:
            results['errors'].append("Failed to calculate metrics")
            return results
        
        # Generate outputs
        try:
            # Text report
            report = self.generate_text_report()
            results['report'] = report
            
            report_path = self.output_dir / 'performance_report.txt'
            with open(report_path, 'w') as f:
                f.write(report)
            
            # Charts
            results['charts'] = self.generate_charts()
            
            # JSON export
            results['json_export'] = self.export_json()
            
            results['success'] = True
            
            # Print to console
            print(report)
            print(f"\nReports saved to: {self.output_dir.absolute()}")
            
        except Exception as e:
            logger.error(f"Error generating outputs: {e}")
            results['errors'].append(str(e))
        
        return results


def quick_analysis(csv_path: str) -> None:
    """Quick analysis function for command line use"""
    analyzer = TradeAnalyzer(csv_path)
    results = analyzer.run_full_analysis()
    
    if not results['success']:
        print("Analysis failed:")
        for error in results['errors']:
            print(f"  - {error}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = 'reports/institutional/institutional_trades.csv'
    
    quick_analysis(csv_file)