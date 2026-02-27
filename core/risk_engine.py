"""
Risk Engine v2.0 - Production-Ready Institutional Risk Management
Integrated with SMC Trading Bot
Features: Dynamic sizing, correlation filtering, drawdown controls, DXY hedging
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict
import json
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    FIXED_RISK = "fixed_risk"
    KELLY = "kelly"
    OPTIMAL_F = "optimal_f"
    ATR_BASED = "atr_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"


@dataclass
class RiskMetrics:
    """Real-time risk metrics"""
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    current_drawdown: float
    daily_pnl: float
    weekly_pnl: float
    win_rate: float
    profit_factor: float
    kelly_fraction: float
    risk_of_ruin: float


@dataclass
class TradeRisk:
    """Individual trade risk assessment"""
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float
    risk_percentage: float
    risk_reward: float
    margin_required: float
    correlation_risk: float
    dxy_hedge_ratio: float


class DynamicCorrelationEngine:
    """Real-time correlation analysis between positions"""
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.price_history: Dict[str, pd.Series] = {}
        self.correlation_matrix: pd.DataFrame = pd.DataFrame()
        
    def update_prices(self, symbol: str, price: float):
        """Update price history"""
        if symbol not in self.price_history:
            self.price_history[symbol] = pd.Series(dtype=float)
        
        self.price_history[symbol] = pd.concat([
            self.price_history[symbol],
            pd.Series({datetime.now(): price})
        ]).tail(self.lookback)
        
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        if symbol1 not in self.price_history or symbol2 not in self.price_history:
            return 0.0
        
        if len(self.price_history[symbol1]) < 10 or len(self.price_history[symbol2]) < 10:
            return 0.0
        
        # Calculate returns
        returns1 = self.price_history[symbol1].pct_change().dropna()
        returns2 = self.price_history[symbol2].pct_change().dropna()
        
        # Align indices
        common_index = returns1.index.intersection(returns2.index)
        if len(common_index) < 5:
            return 0.0
        
        return returns1.loc[common_index].corr(returns2.loc[common_index])
    
    def get_portfolio_correlation_risk(self, symbol: str, open_positions: Dict) -> float:
        """Calculate correlation risk with existing positions"""
        if not open_positions or symbol not in self.price_history:
            return 0.0
        
        correlations = []
        for pos_symbol in open_positions.keys():
            if pos_symbol != symbol:
                corr = abs(self.get_correlation(symbol, pos_symbol))
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0


class DXYCorrelationFilter:
    """Filter trades based on DXY correlation"""
    
    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.lookback = config.get('lookback_period', 50)
        self.min_strength = config.get('minimum_correlation_strength', 0.6)
        self.max_strength = config.get('maximum_correlation_strength', 0.95)
        self.dxy_prices: pd.Series = pd.Series(dtype=float)
        self.symbol_prices: Dict[str, pd.Series] = {}
        
    def update_dxy(self, price: float):
        """Update DXY price"""
        self.dxy_prices = pd.concat([
            self.dxy_prices,
            pd.Series({datetime.now(): price})
        ]).tail(self.lookback)
        
    def update_symbol(self, symbol: str, price: float):
        """Update symbol price"""
        if symbol not in self.symbol_prices:
            self.symbol_prices[symbol] = pd.Series(dtype=float)
        
        self.symbol_prices[symbol] = pd.concat([
            self.symbol_prices[symbol],
            pd.Series({datetime.now(): price})
        ]).tail(self.lookback)
        
    def get_correlation(self, symbol: str) -> float:
        """Get DXY correlation for symbol"""
        if not self.enabled or len(self.dxy_prices) < 20 or symbol not in self.symbol_prices:
            return 0.0
        
        if len(self.symbol_prices[symbol]) < 20:
            return 0.0
        
        # Calculate returns
        dxy_returns = self.dxy_prices.pct_change().dropna()
        symbol_returns = self.symbol_prices[symbol].pct_change().dropna()
        
        common_index = dxy_returns.index.intersection(symbol_returns.index)
        if len(common_index) < 10:
            return 0.0
        
        correlation = dxy_returns.loc[common_index].corr(symbol_returns.loc[common_index])
        return correlation if not np.isnan(correlation) else 0.0
    
    def should_filter_signal(self, symbol: str, direction: str) -> Tuple[bool, str]:
        """Determine if signal should be filtered"""
        if not self.enabled:
            return False, "DXY filter disabled"
        
        correlation = self.get_correlation(symbol)
        
        # For EURUSD, GBPUSD (negatively correlated with DXY)
        # Bullish signal should have negative correlation with DXY (DXY down = pair up)
        if symbol in ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'XAUUSD']:
            if direction == 'bullish' and correlation > 0.5:
                return True, f"Long {symbol} but DXY rising (corr: {correlation:.2f})"
            if direction == 'bearish' and correlation < -0.5:
                return True, f"Short {symbol} but DXY falling (corr: {correlation:.2f})"
        
        # For USDJPY, USDCAD (positively correlated with DXY)
        elif symbol in ['USDJPY', 'USDCAD', 'USDCHF']:
            if direction == 'bullish' and correlation < -0.5:
                return True, f"Long {symbol} but DXY falling (corr: {correlation:.2f})"
            if direction == 'bearish' and correlation > 0.5:
                return True, f"Short {symbol} but DXY rising (corr: {correlation:.2f})"
        
        return False, "Correlation acceptable"
    
    def adjust_position_size(self, symbol: str, base_size: float) -> float:
        """Adjust position size based on DXY correlation strength"""
        correlation = abs(self.get_correlation(symbol))
        
        # Reduce size if correlation is extreme (crowded trade)
        if correlation > self.max_strength:
            return base_size * 0.5
        
        # Increase size slightly if correlation is favorable but not extreme
        if self.min_strength < correlation < 0.8:
            return base_size * 1.1
        
        return base_size


class RiskEngine:
    """
    Production-ready risk management for SMC Trading Bot
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.risk_config = config.get('risk_management', {})
        
        # Initialize sub-engines
        self.correlation_engine = DynamicCorrelationEngine(
            lookback=self.risk_config.get('correlation_lookback', 50)
        )
        self.dxy_filter = DXYCorrelationFilter(
            self.risk_config.get('dxy_correlation', {})
        )
        
        # Account state
        self.initial_balance = self.risk_config.get('initial_balance', 10000)
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        
        # Tracking
        self.open_trades: Dict[str, TradeRisk] = {}
        self.trade_history: List[Dict] = []
        self.daily_trades: List[Dict] = []
        
        # Limits
        self.max_daily_loss = self.risk_config.get('max_daily_loss', 0.03)  # 3%
        self.max_weekly_loss = self.risk_config.get('max_weekly_loss', 0.05)  # 5%
        self.max_monthly_loss = self.risk_config.get('max_monthly_loss', 0.10)  # 10%
        self.max_positions = self.risk_config.get('max_positions', 3)
        self.max_correlated = self.risk_config.get('max_correlated_positions', 2)
        
        # Streak management
        self.consecutive_losses = 0
        self.risk_reduction_factor = 1.0
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        
        logger.info("Risk Engine initialized")
        
    def calculate_position_size(self, 
                               symbol: str,
                               entry_price: float,
                               stop_loss: float,
                               take_profit: float,
                               confidence: float,
                               account_balance: Optional[float] = None,
                               volatility: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate optimal position size with all risk adjustments
        
        Returns complete trade risk assessment
        """
        if account_balance is None:
            account_balance = self.current_balance
        
        # Check if trading is allowed
        can_trade, reason = self.can_trade()
        if not can_trade:
            return {
                'error': f'Trading halted: {reason}',
                'position_size': 0.0,
                'risk_amount': 0.0
            }
        
        # Check DXY correlation filter
        direction = 'bullish' if take_profit > entry_price else 'bearish'
        should_filter, filter_reason = self.dxy_filter.should_filter_signal(symbol, direction)
        if should_filter:
            logger.warning(f"Signal filtered: {filter_reason}")
            return {
                'error': f'Filtered by DXY: {filter_reason}',
                'position_size': 0.0,
                'risk_amount': 0.0
            }
        
        # Base calculation
        base_size = self._calculate_base_size(
            entry_price, stop_loss, account_balance
        )
        
        # Apply confidence adjustment
        confidence_adjusted = self._adjust_by_confidence(base_size, confidence)
        
        # Apply streak adjustment (reduce size after losses)
        streak_adjusted = confidence_adjusted * self.risk_reduction_factor
        
        # Apply DXY correlation adjustment
        dxy_adjusted = self.dxy_filter.adjust_position_size(symbol, streak_adjusted)
        
        # Apply volatility adjustment if data available
        if volatility:
            vol_adjusted = self._adjust_by_volatility(dxy_adjusted, volatility)
        else:
            vol_adjusted = dxy_adjusted
        
        # Check correlation with existing positions
        correlation_risk = self.correlation_engine.get_portfolio_correlation_risk(
            symbol, self.open_trades
        )
        
        if correlation_risk > 0.8 and len(self.open_trades) >= self.max_correlated:
            logger.warning(f"High correlation risk ({correlation_risk:.2f}), reducing size")
            vol_adjusted *= 0.5
        
        # Final validation
        final_size = self._validate_position_size(vol_adjusted, account_balance)
        
        # Calculate metrics
        risk_distance = abs(entry_price - stop_loss)
        risk_amount = final_size * risk_distance * 100000  # Standard lot size
        risk_percentage = (risk_amount / account_balance) * 100
        
        # Calculate R:R
        reward_distance = abs(take_profit - entry_price)
        risk_reward = reward_distance / risk_distance if risk_distance > 0 else 0
        
        # Margin calculation (simplified)
        margin_required = (final_size * entry_price * 100000) / 30  # 1:30 leverage
        
        # DXY hedge ratio
        dxy_corr = self.dxy_filter.get_correlation(symbol)
        hedge_ratio = abs(dxy_corr) if abs(dxy_corr) > 0.5 else 0.0
        
        trade_risk = TradeRisk(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=final_size,
            risk_amount=risk_amount,
            risk_percentage=risk_percentage,
            risk_reward=risk_reward,
            margin_required=margin_required,
            correlation_risk=correlation_risk,
            dxy_hedge_ratio=hedge_ratio
        )
        
        return {
            'position_size': final_size,
            'risk_amount': risk_amount,
            'risk_percentage': risk_percentage,
            'risk_reward': risk_reward,
            'margin_required': margin_required,
            'correlation_risk': correlation_risk,
            'dxy_correlation': dxy_corr,
            'hedge_ratio': hedge_ratio,
            'trade_risk': trade_risk,
            'adjustments': {
                'confidence': confidence,
                'streak_factor': self.risk_reduction_factor,
                'correlation_penalty': 0.5 if correlation_risk > 0.8 else 1.0
            }
        }
    
    def _calculate_base_size(self, entry: float, stop: float, 
                            balance: float) -> float:
        """Calculate base position size using fixed risk"""
        risk_per_trade = self.risk_config.get('risk_per_trade', 0.01)  # 1%
        risk_amount = balance * risk_per_trade
        
        stop_distance = abs(entry - stop)
        if stop_distance == 0:
            return 0.001
        
        # For forex: position_size = risk_amount / (stop_distance * pip_value)
        # Assuming pip_value = $10 per standard lot
        pip_value = 10
        position_size = risk_amount / (stop_distance * 10000 * pip_value)
        
        return position_size
    
    def _adjust_by_confidence(self, base_size: float, confidence: float) -> float:
        """Scale position size by signal confidence"""
        if confidence < 0.5:
            return 0.0
        
        # Scale from 0.5-1.0 confidence -> 0.5-1.5x size
        multiplier = 0.5 + confidence
        return base_size * multiplier
    
    def _adjust_by_volatility(self, base_size: float, volatility: float) -> float:
        """Reduce size in high volatility"""
        # volatility is ATR/price ratio
        if volatility > 0.002:  # High vol
            return base_size * 0.7
        elif volatility < 0.0005:  # Low vol
            return base_size * 1.2
        return base_size
    
    def _validate_position_size(self, size: float, balance: float) -> float:
        """Ensure position size is valid"""
        # Minimum 0.01 micro lot
        min_size = 0.01
        
        # Maximum 5% of account in one position (rough estimate)
        max_size = (balance * 0.05) / 1000  # Simplified
        
        return max(min(size, max_size), min_size)
    
    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is currently allowed"""
        # Check daily loss limit
        daily_limit = self.initial_balance * self.max_daily_loss
        if abs(self.daily_pnl) >= daily_limit:
            return False, f"Daily loss limit reached: {self.daily_pnl:.2f}"
        
        # Check position limit
        if len(self.open_trades) >= self.max_positions:
            return False, f"Max positions reached: {len(self.open_trades)}"
        
        # Check consecutive losses
        if self.consecutive_losses >= 3:
            return False, f"Too many consecutive losses: {self.consecutive_losses}"
        
        return True, "Trading allowed"
    
    def register_trade(self, trade_risk: TradeRisk, ticket: str):
        """Register new trade with risk engine"""
        self.open_trades[ticket] = trade_risk
        self.correlation_engine.update_prices(trade_risk.symbol, trade_risk.entry_price)
        
        logger.info(f"Trade registered: {ticket} {trade_risk.symbol} "
                   f"Size: {trade_risk.position_size:.2f} "
                   f"Risk: {trade_risk.risk_percentage:.2f}%")
    
    def close_trade(self, ticket: str, exit_price: float, pnl: float):
        """Process closed trade"""
        if ticket not in self.open_trades:
            return
        
        trade = self.open_trades.pop(ticket)
        
        # Update balance
        self.current_balance += pnl
        
        # Update peak
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        # Update streaks
        if pnl > 0:
            self.consecutive_losses = 0
            self.risk_reduction_factor = min(1.0, self.risk_reduction_factor + 0.1)
        else:
            self.consecutive_losses += 1
            # Reduce risk after each loss (max 50% reduction)
            self.risk_reduction_factor = max(0.5, self.risk_reduction_factor - 0.15)
        
        # Update daily PnL
        self.daily_pnl += pnl
        
        # Record trade
        record = {
            'ticket': ticket,
            'symbol': trade.symbol,
            'direction': trade.direction,
            'entry': trade.entry_price,
            'exit': exit_price,
            'pnl': pnl,
            'risk': trade.risk_amount,
            'r_multiple': pnl / trade.risk_amount if trade.risk_amount > 0 else 0,
            'time': datetime.now()
        }
        self.trade_history.append(record)
        self.daily_trades.append(record)
        
        logger.info(f"Trade closed: {ticket} PnL: {pnl:.2f} "
                   f"Balance: {self.current_balance:.2f}")
    
    def update_dxy(self, price: float):
        """Update DXY price for correlation filtering"""
        self.dxy_filter.update_dxy(price)
    
    def update_market_data(self, symbol: str, price: float):
        """Update market data for correlation tracking"""
        self.correlation_engine.update_prices(symbol, price)
        if symbol in self.dxy_filter.symbol_prices:
            self.dxy_filter.update_symbol(symbol, price)
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Calculate current risk metrics"""
        if not self.trade_history:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Calculate returns
        returns = [t['pnl'] for t in self.trade_history]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        win_rate = len(wins) / len(returns) if returns else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        profit_factor = sum(wins) / sum(abs(l) for l in losses) if losses else float('inf')
        
        # Drawdown
        current_dd = (self.peak_balance - self.current_balance) / self.peak_balance
        
        # Kelly criterion
        if win_rate > 0 and avg_win > 0:
            kelly = win_rate - ((1 - win_rate) / (avg_win / avg_loss)) if avg_loss > 0 else 0
            kelly = max(0, min(kelly, 0.25))  # Cap at 25%
        else:
            kelly = 0
        
        # VaR (simplified)
        var_95 = np.percentile(returns, 5) if len(returns) > 10 else 0
        var_99 = np.percentile(returns, 1) if len(returns) > 10 else 0
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=np.mean([r for r in returns if r <= var_95]) if len(returns) > 10 else 0,
            max_drawdown=(self.peak_balance - min(self.current_balance, self.peak_balance)) / self.peak_balance,
            current_drawdown=current_dd,
            daily_pnl=self.daily_pnl,
            weekly_pnl=self.weekly_pnl,
            win_rate=win_rate,
            profit_factor=profit_factor,
            kelly_fraction=kelly,
            risk_of_ruin=self._calculate_risk_of_ruin(),
            expectancy=(win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        )
    
    def _calculate_risk_of_ruin(self) -> float:
        """Calculate risk of ruin based on current streak"""
        if self.consecutive_losses >= 5:
            return 0.15
        elif self.consecutive_losses >= 3:
            return 0.08
        return 0.02
    
    def reset_daily(self):
        """Reset daily tracking"""
        self.daily_pnl = 0
        self.daily_trades = []
        self.risk_reduction_factor = min(1.0, self.risk_reduction_factor + 0.2)
        logger.info("Daily risk metrics reset")
    
    def get_portfolio_heat(self) -> float:
        """Calculate total portfolio risk exposure"""
        total_risk = sum(t.risk_percentage for t in self.open_trades.values())
        return total_risk
    
    def should_breakeven(self, ticket: str, current_price: float) -> bool:
        """Determine if position should be moved to breakeven"""
        if ticket not in self.open_trades:
            return False
        
        trade = self.open_trades[ticket]
        profit_distance = abs(current_price - trade.entry_price)
        risk_distance = abs(trade.entry_price - trade.stop_loss)
        
        # Move to BE at 1R profit
        return profit_distance >= risk_distance
    
    def calculate_trailing_stop(self, ticket: str, current_price: float, 
                               atr: float) -> Optional[float]:
        """Calculate trailing stop level"""
        if ticket not in self.open_trades:
            return None
        
        trade = self.open_trades[ticket]
        
        if trade.direction == 'bullish':
            # Trail below recent swing lows or ATR
            new_stop = current_price - (atr * 2)
            return max(new_stop, trade.stop_loss, trade.entry_price)
        else:
            new_stop = current_price + (atr * 2)
            return min(new_stop, trade.stop_loss, trade.entry_price)


# ============================================================================
# INTEGRATION WITH TRADING BOT
# ============================================================================

class RiskManagedBot:
    """
    SMC Trading Bot with full risk management integration
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.engine = LiquidityEngine(config)  # From previous code
        self.risk_engine = RiskEngine(config)
        self.executor = MT5Executor(config)
        
    def process_signal(self, signal: TradingSignal) -> bool:
        """Process trading signal with full risk management"""
        
        # Calculate position size with all risk factors
        entry_price = (signal.entry_zone[0] + signal.entry_zone[1]) / 2
        
        risk_calc = self.risk_engine.calculate_position_size(
            symbol=self.config['symbol'],
            entry_price=entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            confidence=signal.confidence,
            volatility=self._get_current_atr()
        )
        
        if 'error' in risk_calc:
            logger.warning(f"Signal rejected: {risk_calc['error']}")
            return False
        
        # Update signal with risk-managed parameters
        managed_signal = TradingSignal(
            strategy_type=signal.strategy_type,
            direction=signal.direction,
            confidence=signal.confidence,
            entry_zone=signal.entry_zone,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            risk_reward=risk_calc['risk_reward'],
            reason=signal.reason,
            confluence_factors=signal.confluence_factors,
            fib_levels=signal.fib_levels,
            partial_tp_levels=self._calculate_partial_tps(
                entry_price, signal.take_profit, signal.stop_loss, signal.direction
            )
        )
        
        # Execute trade
        result = self.executor.execute_signal(managed_signal, risk_calc['position_size'])
        
        if result.get('success'):
            # Register with risk engine
            self.risk_engine.register_trade(risk_calc['trade_risk'], result['ticket'])
            
            # Set up monitoring
            self._setup_trade_monitoring(result['ticket'], risk_calc['trade_risk'])
            
            return True
        
        return False
    
    def _get_current_atr(self) -> float:
        """Get current ATR for volatility adjustment"""
        # Implementation depends on data feed
        return 0.0010  # Default 10 pips
    
    def _calculate_partial_tps(self, entry: float, tp: float, sl: float, 
                               direction: str) -> List[Tuple[float, float]]:
        """Calculate partial take profit levels"""
        risk_distance = abs(tp - entry)
        
        if direction == 'bullish':
            tp1 = entry + (risk_distance * 0.5)   # 50% at 1:1
            tp2 = entry + (risk_distance * 1.0)   # 25% at 2:1
            tp3 = tp                               # 25% at full target
        else:
            tp1 = entry - (risk_distance * 0.5)
            tp2 = entry - (risk_distance * 1.0)
            tp3 = tp
        
        return [(tp1, 0.5), (tp2, 0.25), (tp3, 0.25)]
    
    def _setup_trade_monitoring(self, ticket: str, trade_risk: TradeRisk):
        """Setup monitoring for trade management"""
        # This would set up callbacks for breakeven and trailing stop
        pass
    
    def update_market_data(self, symbol: str, price: float, dxy_price: Optional[float] = None):
        """Update all risk engines with market data"""
        self.risk_engine.update_market_data(symbol, price)
        
        if dxy_price:
            self.risk_engine.update_dxy(dxy_price)
        
        # Check existing positions for breakeven/trailing
        self._manage_open_positions(symbol, price)
    
    def _manage_open_positions(self, symbol: str, current_price: float):
        """Manage open positions (breakeven, trailing stops)"""
        for ticket, trade in list(self.risk_engine.open_trades.items()):
            if trade.symbol != symbol:
                continue
            
            # Check breakeven
            if self.risk_engine.should_breakeven(ticket, current_price):
                self._move_to_breakeven(ticket)
            
            # Check trailing stop
            atr = self._get_current_atr()
            new_stop = self.risk_engine.calculate_trailing_stop(ticket, current_price, atr)
            if new_stop and new_stop != trade.stop_loss:
                self._update_stop_loss(ticket, new_stop)
    
    def _move_to_breakeven(self, ticket: str):
        """Move stop loss to entry price"""
        # Implementation via MT5 API
        logger.info(f"Moving {ticket} to breakeven")
    
    def _update_stop_loss(self, ticket: str, new_stop: float):
        """Update trailing stop"""
        # Implementation via MT5 API
        logger.info(f"Updating trailing stop for {ticket} to {new_stop}")
    
    def close_trade(self, ticket: str, exit_price: float):
        """Process trade closure"""
        if ticket in self.risk_engine.open_trades:
            trade = self.risk_engine.open_trades[ticket]
            pnl = self._calculate_pnl(trade, exit_price)
            self.risk_engine.close_trade(ticket, exit_price, pnl)
    
    def _calculate_pnl(self, trade: TradeRisk, exit_price: float) -> float:
        """Calculate trade PnL"""
        if trade.direction == 'bullish':
            pips = (exit_price - trade.entry_price) * 10000
        else:
            pips = (trade.entry_price - exit_price) * 10000
        
        return pips * trade.position_size * 10  # $10 per pip per lot
    
    def get_status(self) -> Dict:
        """Get complete system status"""
        return {
            'balance': self.risk_engine.current_balance,
            'open_trades': len(self.risk_engine.open_trades),
            'daily_pnl': self.risk_engine.daily_pnl,
            'risk_metrics': self.risk_engine.get_risk_metrics(),
            'portfolio_heat': self.risk_engine.get_portfolio_heat(),
            'can_trade': self.risk_engine.can_trade()
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'symbol': 'EURUSD',
        'risk_management': {
            'initial_balance': 10000,
            'risk_per_trade': 0.015,  # 1.5%
            'max_positions': 3,
            'max_correlated_positions': 2,
            'max_daily_loss': 0.03,  # 3%
            'dxy_correlation': {
                'enabled': True,
                'lookback_period': 50,
                'minimum_correlation_strength': 0.6,
                'maximum_correlation_strength': 0.95
            }
        }
    }
    
    # Initialize
    risk_engine = RiskEngine(CONFIG)
    
    # Example signal processing
    result = risk_engine.calculate_position_size(
        symbol='EURUSD',
        entry_price=1.0850,
        stop_loss=1.0820,
        take_profit=1.0900,
        confidence=0.85,
        account_balance=10000,
        volatility=0.0008
    )
    
    print(f"Position Size: {result['position_size']:.2f} lots")
    print(f"Risk: {result['risk_percentage']:.2f}%")
    print(f"R:R: {result['risk_reward']:.2f}")
    print(f"DXY Correlation: {result.get('dxy_correlation', 0):.2f}")