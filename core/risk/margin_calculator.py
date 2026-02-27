"""
Margin Calculator v2.0 - Production-Ready Implementation
Optimized for MetaTrader 5 Forex Trading
Features: Real-time margin monitoring, liquidation prevention, position sizing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarginMode(Enum):
    STANDARD = "standard"  # Standard forex lot sizing
    MINI = "mini"          # Mini lots (0.1)
    MICRO = "micro"        # Micro lots (0.01)
    NANO = "nano"          # Nano lots (0.001)


@dataclass
class SymbolSpecs:
    """Symbol specifications for margin calculation"""
    symbol: str
    contract_size: float = 100000  # Standard forex lot
    margin_currency: str = "USD"
    profit_currency: str = "USD"
    swap_long: float = 0.0
    swap_short: float = 0.0
    point_size: float = 0.0001  # 1 pip for most pairs
    digits: int = 5
    trade_calc_mode: int = 0  # 0=Forex, 1=CFD, 2=Futures, etc.


@dataclass
class PositionMargin:
    """Position margin and risk metrics"""
    symbol: str
    direction: str  # 'long' or 'short'
    quantity: float  # Lot size
    entry_price: float
    current_price: float
    
    # Margin metrics
    initial_margin: float
    used_margin: float
    maintenance_margin: float
    
    # Risk metrics
    unrealized_pnl: float
    unrealized_pnl_pips: float
    margin_level: float  # Equity / Used Margin * 100
    
    # Liquidation levels
    liquidation_price: float
    margin_call_price: float
    stop_out_price: float
    
    # Distance metrics
    distance_to_liquidation_pips: float
    distance_to_margin_call_pips: float
    
    # Swap
    daily_swap: float


@dataclass
class AccountMargin:
    """Account-level margin status"""
    balance: float
    equity: float
    used_margin: float
    free_margin: float
    
    margin_level: float  # Percentage
    margin_call_level: float  # Percentage threshold
    stop_out_level: float  # Percentage threshold
    
    total_unrealized_pnl: float
    total_daily_swap: float
    
    positions: List[PositionMargin]
    
    # Risk status
    status: str  # 'safe', 'warning', 'margin_call', 'stop_out'
    risk_percentage: float  # How close to stop_out (0-100)


class MarginCalculator:
    """
    Production-ready margin calculator for MT5 forex trading
    
    Features:
    - Accurate MT5-style margin calculations
    - Real-time margin level monitoring
    - Liquidation price calculation
    - Position size optimization
    - Multi-currency support
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Default settings
        self.leverage = self.config.get('leverage', 30)  # 1:30 default
        self.margin_call_level = self.config.get('margin_call_level', 100)  # 100%
        self.stop_out_level = self.config.get('stop_out_level', 50)  # 50%
        self.account_currency = self.config.get('account_currency', 'USD')
        
        # Symbol specifications cache
        self.symbol_specs: Dict[str, SymbolSpecs] = {}
        
        # Current market prices
        self.current_prices: Dict[str, float] = {}
        self.price_history: Dict[str, List[float]] = {}
        
        logger.info(f"MarginCalculator initialized (leverage={self.leverage}:1)")
    
    def set_symbol_specs(self, symbol: str, specs: SymbolSpecs):
        """Set specifications for a symbol"""
        self.symbol_specs[symbol] = specs
    
    def update_price(self, symbol: str, price: float):
        """Update current market price"""
        self.current_prices[symbol] = price
        
        # Keep price history for volatility calc
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(price)
        
        # Keep last 100 prices
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]
    
    def calculate_position_margin(self, symbol: str, quantity: float,
                                  entry_price: float, direction: str,
                                  current_price: Optional[float] = None) -> PositionMargin:
        """
        Calculate complete margin metrics for a position
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            quantity: Position size in lots
            entry_price: Entry price
            direction: 'long' or 'short'
            current_price: Current price (optional, uses entry if not provided)
            
        Returns:
            PositionMargin with all metrics
        """
        if current_price is None:
            current_price = self.current_prices.get(symbol, entry_price)
        
        # Get symbol specs
        specs = self.symbol_specs.get(symbol, SymbolSpecs(symbol=symbol))
        
        # Calculate notional value
        notional_value = quantity * specs.contract_size * entry_price
        
        # Calculate margin (MT5 style)
        # Margin = Notional / Leverage
        initial_margin = notional_value / self.leverage
        
        # Used margin is the same for forex (no maintenance margin separation)
        used_margin = initial_margin
        
        # Calculate unrealized PnL
        if direction == 'long':
            unrealized_pnl = quantity * specs.contract_size * (current_price - entry_price)
            # For pairs where USD is quote currency (EURUSD, GBPUSD)
            # PnL is already in USD
        else:
            unrealized_pnl = quantity * specs.contract_size * (entry_price - current_price)
        
        # Convert to pips
        price_diff = current_price - entry_price
        if direction == 'short':
            price_diff = -price_diff
        
        # Handle JPY pairs (3 decimal places)
        if 'JPY' in symbol:
            unrealized_pnl_pips = price_diff / 0.001
        else:
            unrealized_pnl_pips = price_diff / 0.0001
        
        # Calculate liquidation price
        # Formula: Liquidation when Equity = Used Margin * StopOut%
        # Equity = Balance + Unrealized PnL
        # At liquidation: Balance + Unrealized PnL = Used Margin * (StopOut/100)
        
        # For a position: Unrealized PnL = (Current - Entry) * Quantity * ContractSize
        # At liquidation: Balance + ((LiquidPrice - Entry) * Q * CS) = UsedMargin * 0.5
        
        # Simplified: Assuming this is the only position and balance = used_margin initially
        # Liquidation when loss = 50% of used margin
        
        max_loss = used_margin * (self.stop_out_level / 100)
        
        if direction == 'long':
            # Price drops to cause max_loss
            price_drop = max_loss / (quantity * specs.contract_size)
            liquidation_price = entry_price - price_drop
            margin_call_price = entry_price - (max_loss * 0.5 / (quantity * specs.contract_size))
            stop_out_price = liquidation_price
        else:
            # Price rises to cause max_loss
            price_rise = max_loss / (quantity * specs.contract_size)
            liquidation_price = entry_price + price_rise
            margin_call_price = entry_price + (max_loss * 0.5 / (quantity * specs.contract_size))
            stop_out_price = liquidation_price
        
        # Calculate distances
        if 'JPY' in symbol:
            pip_size = 0.001
        else:
            pip_size = 0.0001
        
        if direction == 'long':
            distance_to_liquidation = (current_price - liquidation_price) / pip_size
            distance_to_margin_call = (current_price - margin_call_price) / pip_size
        else:
            distance_to_liquidation = (liquidation_price - current_price) / pip_size
            distance_to_margin_call = (margin_call_price - current_price) / pip_size
        
        # Calculate daily swap
        if direction == 'long':
            daily_swap = quantity * specs.contract_size * specs.swap_long
        else:
            daily_swap = quantity * specs.contract_size * specs.swap_short
        
        # Calculate margin level (will be updated at account level)
        margin_level = 0.0  # Placeholder, calculated at account level
        
        return PositionMargin(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            initial_margin=initial_margin,
            used_margin=used_margin,
            maintenance_margin=used_margin * 0.5,  # 50% for MT5
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pips=unrealized_pnl_pips,
            margin_level=margin_level,
            liquidation_price=liquidation_price,
            margin_call_price=margin_call_price,
            stop_out_price=stop_out_price,
            distance_to_liquidation_pips=distance_to_liquidation,
            distance_to_margin_call_pips=distance_to_margin_call,
            daily_swap=daily_swap
        )
    
    def calculate_account_margin(self, balance: float,
                                  positions: List[Dict],
                                  current_prices: Optional[Dict[str, float]] = None) -> AccountMargin:
        """
        Calculate account-level margin status
        
        Args:
            balance: Account balance
            positions: List of position dicts with symbol, quantity, entry_price, direction
            current_prices: Dict of symbol -> current price
            
        Returns:
            AccountMargin with complete status
        """
        if current_prices:
            self.current_prices.update(current_prices)
        
        # Calculate each position
        position_margins = []
        total_used_margin = 0.0
        total_unrealized_pnl = 0.0
        total_daily_swap = 0.0
        
        for pos in positions:
            pm = self.calculate_position_margin(
                symbol=pos['symbol'],
                quantity=pos['quantity'],
                entry_price=pos['entry_price'],
                direction=pos['direction'],
                current_price=self.current_prices.get(pos['symbol'])
            )
            
            position_margins.append(pm)
            total_used_margin += pm.used_margin
            total_unrealized_pnl += pm.unrealized_pnl
            total_daily_swap += pm.daily_swap
        
        # Calculate equity and free margin
        equity = balance + total_unrealized_pnl
        free_margin = equity - total_used_margin
        
        # Calculate margin level
        if total_used_margin > 0:
            margin_level = (equity / total_used_margin) * 100
        else:
            margin_level = 0.0
        
        # Determine status
        if margin_level <= self.stop_out_level:
            status = 'stop_out'
            risk_percentage = 100.0
        elif margin_level <= self.margin_call_level:
            status = 'margin_call'
            risk_percentage = ((self.margin_call_level - margin_level) / 
                             (self.margin_call_level - self.stop_out_level)) * 100
        elif margin_level <= self.margin_call_level * 1.5:
            status = 'warning'
            risk_percentage = 50.0
        else:
            status = 'safe'
            risk_percentage = 0.0
        
        return AccountMargin(
            balance=balance,
            equity=equity,
            used_margin=total_used_margin,
            free_margin=free_margin,
            margin_level=margin_level,
            margin_call_level=self.margin_call_level,
            stop_out_level=self.stop_out_level,
            total_unrealized_pnl=total_unrealized_pnl,
            total_daily_swap=total_daily_swap,
            positions=position_margins,
            status=status,
            risk_percentage=risk_percentage
        )
    
    def calculate_position_size(self, symbol: str, risk_amount: float,
                                 stop_loss_pips: float,
                                 entry_price: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate optimal position size based on risk
        
        Args:
            symbol: Trading symbol
            risk_amount: Amount to risk in account currency
            stop_loss_pips: Stop loss distance in pips
            entry_price: Entry price (optional)
            
        Returns:
            Dict with position size and margin required
        """
        if entry_price is None:
            entry_price = self.current_prices.get(symbol, 1.0)
        
        specs = self.symbol_specs.get(symbol, SymbolSpecs(symbol=symbol))
        
        # Calculate pip value
        if 'JPY' in symbol:
            pip_value = specs.contract_size * 0.001  # 0.001 for JPY pairs
        else:
            pip_value = specs.contract_size * 0.0001  # 0.0001 for standard pairs
        
        # Calculate position size
        # Risk = PositionSize * PipValue * StopLossPips
        # PositionSize = Risk / (PipValue * StopLossPips)
        
        total_risk_per_lot = pip_value * stop_loss_pips
        
        if total_risk_per_lot == 0:
            return {
                'position_size': 0.0,
                'lots': 0.0,
                'margin_required': 0.0,
                'error': 'Invalid stop loss'
            }
        
        position_size = risk_amount / total_risk_per_lot
        
        # Round to standard lot sizes
        if position_size >= 1.0:
            lots = round(position_size, 2)  # Standard lots, 2 decimal places
        elif position_size >= 0.1:
            lots = round(position_size, 2)  # Mini lots
        else:
            lots = round(position_size, 3)  # Micro lots
        
        # Ensure minimum lot size
        lots = max(lots, 0.001)  # Minimum 0.001 lot (nano)
        
        # Calculate margin required
        notional = lots * specs.contract_size * entry_price
        margin_required = notional / self.leverage
        
        return {
            'position_size': lots,
            'lots': lots,
            'units': lots * specs.contract_size,
            'margin_required': margin_required,
            'notional_value': notional,
            'risk_amount': risk_amount,
            'stop_loss_pips': stop_loss_pips,
            'pip_value': pip_value
        }
    
    def calculate_max_position_size(self, symbol: str, 
                                     available_margin: Optional[float] = None,
                                     entry_price: Optional[float] = None) -> float:
        """
        Calculate maximum position size based on available margin
        
        Args:
            symbol: Trading symbol
            available_margin: Available margin (free margin)
            entry_price: Entry price
            
        Returns:
            Maximum lot size
        """
        if entry_price is None:
            entry_price = self.current_prices.get(symbol, 1.0)
        
        if available_margin is None:
            available_margin = 1000  # Default $1000
        
        specs = self.symbol_specs.get(symbol, SymbolSpecs(symbol=symbol))
        
        # Max position = (Available Margin * Leverage) / (Contract Size * Price)
        max_notional = available_margin * self.leverage
        max_lots = max_notional / (specs.contract_size * entry_price)
        
        # Apply safety buffer (use 90% of available margin)
        max_lots *= 0.9
        
        # Round down to safe lot size
        if max_lots >= 1.0:
            return round(max_lots - 0.005, 2)  # Round down
        elif max_lots >= 0.1:
            return round(max_lots - 0.0005, 2)
        else:
            return round(max_lots - 0.00005, 3)
    
    def check_margin_for_trade(self, symbol: str, quantity: float,
                                direction: str, entry_price: float,
                                account_balance: float,
                                current_positions: List[Dict]) -> Tuple[bool, str, Dict]:
        """
        Check if a new trade can be opened with current margin
        
        Args:
            symbol: Symbol to trade
            quantity: Lot size
            direction: 'long' or 'short'
            entry_price: Entry price
            account_balance: Current balance
            current_positions: Existing positions
            
        Returns:
            (can_trade, reason, details) tuple
        """
        # Calculate current account state
        current_state = self.calculate_account_margin(account_balance, current_positions)
        
        # Calculate new position margin
        new_position = self.calculate_position_margin(symbol, quantity, entry_price, direction)
        
        # Calculate post-trade state
        new_used_margin = current_state.used_margin + new_position.used_margin
        new_equity = current_state.equity  # No unrealized PnL yet for new position
        new_free_margin = new_equity - new_used_margin
        
        # Check if we have enough free margin
        if new_free_margin < 0:
            return False, "Insufficient free margin", {
                'required_margin': new_position.used_margin,
                'available_margin': current_state.free_margin,
                'shortfall': abs(new_free_margin)
            }
        
        # Check margin level after trade
        if new_used_margin > 0:
            new_margin_level = (new_equity / new_used_margin) * 100
        else:
            new_margin_level = 0
        
        if new_margin_level < self.margin_call_level:
            return False, f"Margin level would be {new_margin_level:.1f}% (below {self.margin_call_level}%)", {
                'current_margin_level': current_state.margin_level,
                'projected_margin_level': new_margin_level
            }
        
        # Check if we're over-leveraged
        total_notional = sum(
            pos['quantity'] * self.symbol_specs.get(pos['symbol'], SymbolSpecs(symbol=pos['symbol'])).contract_size * 
            self.current_prices.get(pos['symbol'], pos['entry_price'])
            for pos in current_positions
        ) + (quantity * self.symbol_specs.get(symbol, SymbolSpecs(symbol=symbol)).contract_size * entry_price)
        
        effective_leverage = total_notional / account_balance if account_balance > 0 else 0
        
        if effective_leverage > self.leverage * 0.9:
            return False, f"Effective leverage {effective_leverage:.1f}x too high", {
                'max_leverage': self.leverage,
                'effective_leverage': effective_leverage
            }
        
        return True, "Trade approved", {
            'required_margin': new_position.used_margin,
            'new_margin_level': new_margin_level,
            'new_free_margin': new_free_margin,
            'effective_leverage': effective_leverage
        }
    
    def get_margin_alerts(self, account_margin: AccountMargin) -> List[Dict]:
        """
        Generate margin alerts based on account status
        
        Args:
            account_margin: Current account margin status
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        if account_margin.status == 'stop_out':
            alerts.append({
                'level': 'critical',
                'type': 'stop_out_imminent',
                'message': f"STOP OUT IMMINENT! Margin level: {account_margin.margin_level:.1f}%",
                'action': 'Close positions immediately',
                'timestamp': datetime.now()
            })
        
        elif account_margin.status == 'margin_call':
            alerts.append({
                'level': 'high',
                'type': 'margin_call',
                'message': f"Margin Call! Level: {account_margin.margin_level:.1f}%",
                'action': 'Close positions or add funds',
                'suggestion': f"Deposit at least ${account_margin.used_margin * 0.5:.2f}",
                'timestamp': datetime.now()
            })
        
        elif account_margin.status == 'warning':
            alerts.append({
                'level': 'medium',
                'type': 'margin_warning',
                'message': f"Margin level low: {account_margin.margin_level:.1f}%",
                'action': 'Monitor closely',
                'suggestion': 'Consider reducing position sizes',
                'timestamp': datetime.now()
            })
        
        # Check individual positions
        for pos in account_margin.positions:
            if pos.distance_to_liquidation_pips < 50:
                alerts.append({
                    'level': 'high',
                    'type': 'position_near_liquidation',
                    'symbol': pos.symbol,
                    'message': f"{pos.symbol} {pos.distance_to_liquidation_pips:.0f} pips from liquidation",
                    'action': f"Close {pos.symbol} or add margin",
                    'timestamp': datetime.now()
                })
        
        return alerts
    
    def suggest_position_adjustments(self, account_margin: AccountMargin,
                                      target_margin_level: float = 200.0) -> List[Dict]:
        """
        Suggest position adjustments to improve margin level
        
        Args:
            account_margin: Current account margin status
            target_margin_level: Target margin level percentage
            
        Returns:
            List of adjustment suggestions
        """
        suggestions = []
        
        if account_margin.margin_level >= target_margin_level:
            return suggestions
        
        # Calculate how much margin we need to free up
        current_equity = account_margin.equity
        target_used_margin = (current_equity / target_margin_level) * 100
        margin_to_free = account_margin.used_margin - target_used_margin
        
        if margin_to_free <= 0:
            return suggestions
        
        # Sort positions by margin used (largest first)
        sorted_positions = sorted(
            account_margin.positions,
            key=lambda x: x.used_margin,
            reverse=True
        )
        
        freed_margin = 0.0
        for pos in sorted_positions:
            if freed_margin >= margin_to_free:
                break
            
            # Suggest closing this position
            suggestions.append({
                'action': 'close_position',
                'symbol': pos.symbol,
                'current_size': pos.quantity,
                'margin_freed': pos.used_margin,
                'unrealized_pnl': pos.unrealized_pnl,
                'reason': f'Free up {pos.used_margin:.2f} margin'
            })
            
            freed_margin += pos.used_margin
        
        # If closing positions isn't enough, suggest deposit
        if freed_margin < margin_to_free:
            deposit_needed = (account_margin.used_margin * (target_margin_level / 100)) - account_margin.equity
            suggestions.append({
                'action': 'deposit_funds',
                'amount': deposit_needed,
                'reason': f'Reach {target_margin_level}% margin level'
            })
        
        return suggestions
    
    def get_liquidation_estimate(self, positions: List[Dict],
                                  account_balance: float,
                                  price_changes: Dict[str, float]) -> Dict[str, Any]:
        """
        Estimate liquidation risk under various price scenarios
        
        Args:
            positions: Current positions
            account_balance: Account balance
            price_changes: Dict of symbol -> price change percentage
            
        Returns:
            Liquidation risk assessment
        """
        # Apply price changes
        stressed_prices = {}
        for pos in positions:
            symbol = pos['symbol']
            current = self.current_prices.get(symbol, pos['entry_price'])
            
            if symbol in price_changes:
                stressed_prices[symbol] = current * (1 + price_changes[symbol])
            else:
                stressed_prices[symbol] = current
        
        # Calculate stressed margin
        stressed_state = self.calculate_account_margin(account_balance, positions, stressed_prices)
        
        return {
            'current_margin_level': self.calculate_account_margin(account_balance, positions).margin_level,
            'stressed_margin_level': stressed_state.margin_level,
            'liquidation_risk': stressed_state.status == 'stop_out',
            'price_changes': price_changes,
            'stressed_equity': stressed_state.equity,
            'stressed_free_margin': stressed_state.free_margin
        }


# ============================================================================
# MT5 INTEGRATION
# ============================================================================

class MT5MarginMonitor:
    """
    Real-time margin monitor for MT5 integration
    """
    
    def __init__(self, calculator: MarginCalculator):
        self.calculator = calculator
        self.last_check: Optional[datetime] = None
        self.check_interval = timedelta(seconds=1)
        
        # Alert callbacks
        self.alert_handlers: List[callable] = []
    
    def register_alert_handler(self, handler: callable):
        """Register callback for margin alerts"""
        self.alert_handlers.append(handler)
    
    def update_from_mt5(self, account_info: Dict, positions: List[Dict]):
        """
        Update from MT5 account info
        
        Args:
            account_info: Dict with balance, equity, margin, etc.
            positions: List of position dicts from MT5
        """
        current_time = datetime.now()
        
        # Throttle checks
        if self.last_check and (current_time - self.last_check) < self.check_interval:
            return
        
        self.last_check = current_time
        
        # Update prices from positions
        for pos in positions:
            self.calculator.update_price(pos['symbol'], pos['current_price'])
        
        # Calculate margin status
        margin_status = self.calculator.calculate_account_margin(
            balance=account_info.get('balance', 0),
            positions=positions
        )
        
        # Check for alerts
        alerts = self.calculator.get_margin_alerts(margin_status)
        
        # Trigger callbacks
        for alert in alerts:
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler error: {e}")
        
        return margin_status
    
    def get_trade_permission(self, symbol: str, lots: float,
                            direction: str, mt5_account_info: Dict,
                            mt5_positions: List[Dict]) -> Tuple[bool, str]:
        """
        Check if MT5 should allow a trade
        
        Args:
            symbol: Symbol to trade
            lots: Lot size
            direction: 'buy' or 'sell'
            mt5_account_info: MT5 account info
            mt5_positions: Current MT5 positions
            
        Returns:
            (allowed, reason) tuple
        """
        entry_price = self.calculator.current_prices.get(symbol, 0)
        
        can_trade, reason, details = self.calculator.check_margin_for_trade(
            symbol=symbol,
            quantity=lots,
            direction='long' if direction == 'buy' else 'short',
            entry_price=entry_price,
            account_balance=mt5_account_info.get('balance', 0),
            current_positions=mt5_positions
        )
        
        return can_trade, reason


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize calculator
    config = {
        'leverage': 30,
        'margin_call_level': 100,
        'stop_out_level': 50
    }
    
    calc = MarginCalculator(config)
    
    # Set up symbol specs
    calc.set_symbol_specs('EURUSD', SymbolSpecs(
        symbol='EURUSD',
        contract_size=100000,
        point_size=0.0001,
        digits=5
    ))
    
    calc.set_symbol_specs('USDJPY', SymbolSpecs(
        symbol='USDJPY',
        contract_size=100000,
        point_size=0.001,
        digits=3
    ))
    
    # Update prices
    calc.update_price('EURUSD', 1.0850)
    calc.update_price('USDJPY', 150.25)
    
    # Example: Calculate position size for $100 risk with 50 pip stop
    print("Position Sizing Example:")
    print("=" * 50)
    
    size_calc = calc.calculate_position_size('EURUSD', risk_amount=100, stop_loss_pips=50)
    print(f"Risk Amount: ${size_calc['risk_amount']}")
    print(f"Stop Loss: {size_calc['stop_loss_pips']} pips")
    print(f"Position Size: {size_calc['position_size']:.3f} lots")
    print(f"Units: {size_calc['units']:,.0f}")
    print(f"Margin Required: ${size_calc['margin_required']:.2f}")
    
    # Example: Check account margin
    print("\nAccount Margin Example:")
    print("=" * 50)
    
    positions = [
        {'symbol': 'EURUSD', 'quantity': 0.5, 'entry_price': 1.0800, 'direction': 'long'},
        {'symbol': 'USDJPY', 'quantity': 0.3, 'entry_price': 149.50, 'direction': 'short'}
    ]
    
    # Update current prices
    calc.update_price('EURUSD', 1.0850)  # +50 pips for EURUSD long
    calc.update_price('USDJPY', 150.25)  # -75 pips for USDJPY short
    
    account = calc.calculate_account_margin(balance=5000, positions=positions)
    
    print(f"Balance: ${account.balance:,.2f}")
    print(f"Equity: ${account.equity:,.2f}")
    print(f"Used Margin: ${account.used_margin:,.2f}")
    print(f"Free Margin: ${account.free_margin:,.2f}")
    print(f"Margin Level: {account.margin_level:.1f}%")
    print(f"Status: {account.status.upper()}")
    print(f"Total Unrealized PnL: ${account.total_unrealized_pnl:,.2f}")
    
    # Position details
    print("\nPosition Details:")
    for pos in account.positions:
        print(f"\n{pos.symbol} {pos.direction.upper()}")
        print(f"  Size: {pos.quantity} lots")
        print(f"  Entry: {pos.entry_price}")
        print(f"  Current: {pos.current_price}")
        print(f"  Unrealized PnL: ${pos.unrealized_pnl:,.2f} ({pos.unrealized_pnl_pips:+.1f} pips)")
        print(f"  Used Margin: ${pos.used_margin:,.2f}")
        print(f"  Liquidation Price: {pos.liquidation_price}")
        print(f"  Distance to Liquidation: {pos.distance_to_liquidation_pips:.0f} pips")
    
    # Check for trade
    print("\nTrade Permission Check:")
    print("=" * 50)
    
    can_trade, reason, details = calc.check_margin_for_trade(
        symbol='EURUSD',
        quantity=1.0,
        direction='long',
        entry_price=1.0850,
        account_balance=5000,
        current_positions=positions
    )
    
    print(f"Can Trade: {'YES' if can_trade else 'NO'}")
    print(f"Reason: {reason}")
    if details:
        for key, value in details.items():
            print(f"  {key}: {value}")
    
    # Margin alerts
    print("\nMargin Alerts:")
    print("=" * 50)
    alerts = calc.get_margin_alerts(account)
    for alert in alerts:
        print(f"[{alert['level'].upper()}] {alert['type']}: {alert['message']}")