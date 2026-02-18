"""
Advanced Margin Calculator - Exchange-specific margin calculations
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import math
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class MarginConfig:
    """Configuration for margin calculations"""
    # Leverage settings
    max_leverage: int = 100
    default_leverage: int = 10
    
    # Risk parameters
    margin_of_safety: float = 1.5
    liquidation_buffer: float = 0.1
    maintenance_margin_ratio: float = 0.005
    
    # Portfolio margin settings
    portfolio_margin_enabled: bool = False
    portfolio_margin_threshold: float = 0.05
    
    # Margin modes
    margin_mode: str = 'cross'  # 'cross', 'isolated'
    
    # Exchange-specific settings
    binance_margin_mode: str = 'cross'
    binance_initial_margin_rate: float = 0.01
    binance_maintenance_margin_rate: float = 0.005
    
    mt5_margin_calculator: str = 'standard'  # 'standard', 'fifo', 'hedged'
    mt5_initial_margin_rate: float = 0.01
    mt5_maintenance_margin_rate: float = 0.005
    
    # Fee structures
    exchange_fees: Dict[str, float] = field(default_factory=dict)
    maker_fees: Dict[str, float] = field(default_factory=dict)
    taker_fees: Dict[str, float] = field(default_factory=dict)
    
    # Margin reporting
    margin_reporting_enabled: bool = True
    margin_reporting_frequency: int = 300  # seconds


@dataclass
class PositionMargin:
    """Position-specific margin requirements"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    direction: str  # 'long' or 'short'
    
    initial_margin: float
    maintenance_margin: float
    used_margin: float
    available_margin: float
    margin_ratio: float
    margin_level: float
    
    liquidation_price: float
    bankruptcy_price: float
    margin_call_price: float
    
    leverage: float
    effective_leverage: float
    
    mark_price: float
    unrealized_pnl: float
    realized_pnl: float
    
    funding_fee: float
    position_funding_rate: float
    next_funding_time: Optional[datetime] = None


@dataclass
class PortfolioMargin:
    """Portfolio-level margin requirements"""
    account_balance: float
    equity: float
    margin_balance: float
    used_margin: float
    available_margin: float
    margin_ratio: float
    margin_level: float
    
    initial_margin: float
    maintenance_margin: float
    
    liquidation_threshold: float
    margin_call_level: float
    
    positions: List[PositionMargin]
    position_margins: Dict[str, PositionMargin]
    
    max_leverage: float
    effective_leverage: float
    
    funding_rate: float
    premium_index: float
    next_funding_time: Optional[datetime] = None


class MarginCalculator:
    """
    Advanced margin calculator with exchange-specific calculations
    
    Features:
    - Exchange-specific margin algorithms
    - Portfolio margin calculations
    - Leverage management
    - Liquidation price calculation
    - Margin call alerts
    - Funding fee calculations
    """
    
    def __init__(self, config: dict):
        self.config = MarginConfig(**config['risk_management']['margin'])
        self.logger = logging.getLogger(__name__)
        
        # Exchange fee structures
        self._initialize_exchange_fees()
        
        logger.info("MarginCalculator initialized")
    
    def _initialize_exchange_fees(self):
        """Initialize exchange fee structures"""
        # Set default fees if not provided
        if not self.config.exchange_fees:
            self.config.exchange_fees = {
                'binance': 0.001,
                'binance_futures': 0.0004,
                'mt5': 0.0002,
                'ccxt': 0.001
            }
        
        if not self.config.maker_fees:
            self.config.maker_fees = {
                'binance': 0.001,
                'binance_futures': 0.0002,
                'mt5': 0.0001,
                'ccxt': 0.001
            }
        
        if not self.config.taker_fees:
            self.config.taker_fees = {
                'binance': 0.001,
                'binance_futures': 0.0004,
                'mt5': 0.0002,
                'ccxt': 0.001
            }
    
    async def calculate_position_margin(self, position: Dict, 
                                       exchange: str = 'binance',
                                       exchange_info: Optional[Dict] = None) -> PositionMargin:
        """
        Calculate position margin requirements
        
        Args:
            position: Position dictionary with quantity, price, direction
            exchange: Exchange type
            exchange_info: Exchange-specific information
            
        Returns:
            PositionMargin object
        """
        if exchange == 'binance' or exchange == 'binance_futures':
            return await self._calculate_binance_position_margin(position, exchange, exchange_info)
        elif exchange == 'mt5':
            return await self._calculate_mt5_position_margin(position, exchange_info)
        else:
            return await self._calculate_generic_position_margin(position)
    
    async def _calculate_binance_position_margin(self, position: Dict, 
                                                exchange: str = 'binance',
                                                exchange_info: Optional[Dict] = None) -> PositionMargin:
        """Calculate Binance-specific margin requirements"""
        symbol = position['symbol']
        quantity = position['quantity']
        entry_price = position['avg_entry_price']
        current_price = position['current_price']
        direction = position['direction']
        
        # Get symbol specifications
        symbol_specs = self._get_binance_symbol_specs(symbol, exchange_info)
        
        # Calculate mark price
        mark_price = current_price
        
        # Calculate notional value
        notional_value = abs(quantity) * mark_price
        
        # Determine margin mode
        margin_mode = self.config.binance_margin_mode
        
        # Calculate initial margin
        if exchange == 'binance_futures':
            # Futures margin calculation
            if margin_mode == 'cross':
                # Cross margin: uses all available margin
                initial_margin = notional_value * self.config.binance_initial_margin_rate
            else:
                # Isolated margin: initial margin is fixed
                initial_margin = notional_value * (1 / self.config.max_leverage)
        else:
            # Spot margin calculation
            initial_margin = notional_value * (1 / self.config.default_leverage)
        
        # Calculate maintenance margin
        maintenance_margin = notional_value * self.config.binance_maintenance_margin_rate
        
        # Calculate used margin
        used_margin = initial_margin
        
        # Calculate PnL
        unrealized_pnl = 0.0
        
        if direction == 'long':
            unrealized_pnl = quantity * (mark_price - entry_price)
        else:
            unrealized_pnl = quantity * (entry_price - mark_price)
        
        realized_pnl = position.get('realized_pnl', 0.0)
        
        # Calculate liquidation price
        liquidation_price = await self._calculate_binance_liquidation_price(
            position,
            mark_price,
            initial_margin,
            maintenance_margin,
            margin_mode,
            symbol_specs
        )
        
        # Calculate bankruptcy price
        bankruptcy_price = self._calculate_binance_bankruptcy_price(
            position,
            mark_price,
            maintenance_margin,
            margin_mode
        )
        
        # Calculate margin call price
        margin_call_price = await self._calculate_binance_margin_call_price(
            position,
            mark_price,
            initial_margin,
            maintenance_margin,
            margin_mode
        )
        
        # Calculate funding fee
        funding_fee = await self._calculate_binance_funding_fee(position)
        
        return PositionMargin(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            direction=direction,
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            used_margin=used_margin,
            available_margin=0.0,  # Will be calculated at portfolio level
            margin_ratio=initial_margin / notional_value,
            margin_level=(initial_margin + unrealized_pnl) / initial_margin if initial_margin > 0 else 0,
            liquidation_price=liquidation_price,
            bankruptcy_price=bankruptcy_price,
            margin_call_price=margin_call_price,
            leverage=self.config.default_leverage,
            effective_leverage=notional_value / initial_margin if initial_margin > 0 else 0,
            mark_price=mark_price,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            funding_fee=funding_fee,
            position_funding_rate=symbol_specs.get('funding_rate', 0.0),
            next_funding_time=symbol_specs.get('next_funding_time')
        )
    
    async def _calculate_mt5_position_margin(self, position: Dict, 
                                            exchange_info: Optional[Dict] = None) -> PositionMargin:
        """Calculate MT5-specific margin requirements"""
        symbol = position['symbol']
        quantity = position['quantity']
        entry_price = position['avg_entry_price']
        current_price = position['current_price']
        direction = position['direction']
        
        # Get symbol specifications
        symbol_specs = self._get_mt5_symbol_specs(symbol, exchange_info)
        
        # Calculate notional value
        notional_value = abs(quantity) * current_price
        
        # Calculate initial margin
        if symbol_specs['margin_mode'] == 'fxf':  # Forex
            leverage = self.config.default_leverage
            initial_margin = notional_value / leverage
        else:
            # Other instruments use percentage-based margin
            initial_margin = notional_value * self.config.mt5_initial_margin_rate
        
        # Calculate maintenance margin
        maintenance_margin = notional_value * self.config.mt5_maintenance_margin_rate
        
        # Calculate used margin
        used_margin = initial_margin
        
        # Calculate PnL
        unrealized_pnl = 0.0
        
        if direction == 'long':
            unrealized_pnl = quantity * (current_price - entry_price)
        else:
            unrealized_pnl = quantity * (entry_price - current_price)
        
        realized_pnl = position.get('realized_pnl', 0.0)
        
        # Calculate liquidation price
        liquidation_price = await self._calculate_mt5_liquidation_price(
            position,
            current_price,
            initial_margin,
            maintenance_margin
        )
        
        return PositionMargin(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            direction=direction,
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            used_margin=used_margin,
            available_margin=0.0,
            margin_ratio=initial_margin / notional_value,
            margin_level=(initial_margin + unrealized_pnl) / initial_margin if initial_margin > 0 else 0,
            liquidation_price=liquidation_price,
            bankruptcy_price=liquidation_price,
            margin_call_price=await self._calculate_mt5_margin_call_price(
                position, current_price, initial_margin, maintenance_margin
            ),
            leverage=self.config.default_leverage,
            effective_leverage=notional_value / initial_margin if initial_margin > 0 else 0,
            mark_price=current_price,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            funding_fee=0.0,
            position_funding_rate=0.0
        )
    
    async def _calculate_generic_position_margin(self, position: Dict) -> PositionMargin:
        """Calculate generic margin requirements for other exchanges"""
        symbol = position['symbol']
        quantity = position['quantity']
        entry_price = position['avg_entry_price']
        current_price = position['current_price']
        direction = position['direction']
        
        notional_value = abs(quantity) * current_price
        
        initial_margin = notional_value * self.config.margin_of_safety
        maintenance_margin = notional_value * self.config.maintenance_margin_ratio
        
        unrealized_pnl = quantity * (current_price - entry_price) if direction == 'long' else \
                       quantity * (entry_price - current_price)
        
        liquidation_price = await self._calculate_generic_liquidation_price(
            position,
            current_price,
            initial_margin,
            maintenance_margin
        )
        
        return PositionMargin(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            direction=direction,
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            used_margin=initial_margin,
            available_margin=0.0,
            margin_ratio=initial_margin / notional_value,
            margin_level=(initial_margin + unrealized_pnl) / initial_margin if initial_margin > 0 else 0,
            liquidation_price=liquidation_price,
            bankruptcy_price=liquidation_price,
            margin_call_price=liquidation_price * 1.05,  # 5% buffer
            leverage=self.config.default_leverage,
            effective_leverage=notional_value / initial_margin if initial_margin > 0 else 0,
            mark_price=current_price,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=0.0,
            funding_fee=0.0,
            position_funding_rate=0.0
        )
    
    async def calculate_portfolio_margin(self, positions: List[Dict], 
                                        account_info: Dict,
                                        exchange: str = 'binance',
                                        exchange_info: Optional[Dict] = None) -> PortfolioMargin:
        """
        Calculate portfolio-level margin requirements
        
        Args:
            positions: List of positions
            account_info: Account information
            exchange: Exchange type
            exchange_info: Exchange-specific information
            
        Returns:
            PortfolioMargin object
        """
        position_margins = []
        
        for pos in positions:
            pos_margin = await self.calculate_position_margin(pos, exchange, exchange_info)
            position_margins.append(pos_margin)
        
        # Calculate total used margin
        used_margin = sum(pos.used_margin for pos in position_margins)
        
        # Calculate equity
        account_balance = account_info.get('balance', 0.0)
        unrealized_pnl = sum(pos.unrealized_pnl for pos in position_margins)
        equity = account_balance + unrealized_pnl
        
        # Calculate available margin
        available_margin = equity - used_margin
        
        # Calculate initial margin
        initial_margin = sum(pos.initial_margin for pos in position_margins)
        
        # Calculate maintenance margin
        maintenance_margin = sum(pos.maintenance_margin for pos in position_margins)
        
        # Calculate margin ratio and level
        margin_ratio = used_margin / equity if equity > 0 else 0
        margin_level = equity / used_margin if used_margin > 0 else 0
        
        # Calculate portfolio leverage
        total_notional = sum(abs(pos.quantity * pos.mark_price) for pos in position_margins)
        max_leverage = self.config.max_leverage
        effective_leverage = total_notional / equity if equity > 0 else 0
        
        # Calculate funding rate and next funding time
        funding_rate = 0.0
        next_funding_time = None
        
        if exchange == 'binance_futures':
            funding_rate = await self._calculate_binance_portfolio_funding_rate(positions, exchange_info)
            next_funding_time = await self._get_next_binance_funding_time(exchange_info)
        
        # Check for margin call and liquidation levels
        margin_call_level = 0.0
        liquidation_threshold = 0.0
        
        if exchange == 'binance' or exchange == 'binance_futures':
            margin_call_level = 1.1  # 110% margin level
            liquidation_threshold = 0.8  # 80% margin level
        elif exchange == 'mt5':
            margin_call_level = 1.2  # 120% margin level
            liquidation_threshold = 0.9  # 90% margin level
        
        return PortfolioMargin(
            account_balance=account_balance,
            equity=equity,
            margin_balance=account_balance,
            used_margin=used_margin,
            available_margin=available_margin,
            margin_ratio=margin_ratio,
            margin_level=margin_level,
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            liquidation_threshold=liquidation_threshold,
            margin_call_level=margin_call_level,
            positions=positions,
            position_margins={pos.symbol: pos for pos in position_margins},
            max_leverage=max_leverage,
            effective_leverage=effective_leverage,
            funding_rate=funding_rate,
            premium_index=0.0,
            next_funding_time=next_funding_time
        )
    
    async def _calculate_binance_liquidation_price(self, position: Dict,
                                                 mark_price: float,
                                                 initial_margin: float,
                                                 maintenance_margin: float,
                                                 margin_mode: str,
                                                 symbol_specs: Dict) -> float:
        """Calculate Binance liquidation price"""
        quantity = position['quantity']
        entry_price = position['avg_entry_price']
        direction = position['direction']
        
        # Binance futures liquidation formula
        if direction == 'long':
            liquidation_price = entry_price * (
                1 - (1 / self.config.default_leverage) + self.config.maintenance_margin_rate
            )
        else:
            liquidation_price = entry_price * (
                1 + (1 / self.config.default_leverage) - self.config.maintenance_margin_rate
            )
        
        return liquidation_price
    
    def _calculate_binance_bankruptcy_price(self, position: Dict,
                                           mark_price: float,
                                           maintenance_margin: float,
                                           margin_mode: str) -> float:
        """Calculate Binance bankruptcy price"""
        return mark_price * (1 - maintenance_margin)
    
    async def _calculate_binance_margin_call_price(self, position: Dict,
                                                 mark_price: float,
                                                 initial_margin: float,
                                                 maintenance_margin: float,
                                                 margin_mode: str) -> float:
        """Calculate Binance margin call price"""
        return mark_price * 1.05
    
    async def _calculate_mt5_liquidation_price(self, position: Dict,
                                               current_price: float,
                                               initial_margin: float,
                                               maintenance_margin: float) -> float:
        """Calculate MT5 liquidation price"""
        quantity = position['quantity']
        entry_price = position['avg_entry_price']
        direction = position['direction']
        
        # MT5 liquidation formula
        if direction == 'long':
            liquidation_price = entry_price - (initial_margin / quantity)
        else:
            liquidation_price = entry_price + (initial_margin / quantity)
        
        return liquidation_price
    
    async def _calculate_mt5_margin_call_price(self, position: Dict,
                                               current_price: float,
                                               initial_margin: float,
                                               maintenance_margin: float) -> float:
        """Calculate MT5 margin call price"""
        return current_price * 1.03
    
    async def _calculate_generic_liquidation_price(self, position: Dict,
                                                 current_price: float,
                                                 initial_margin: float,
                                                 maintenance_margin: float) -> float:
        """Calculate generic liquidation price"""
        quantity = position['quantity']
        entry_price = position['avg_entry_price']
        direction = position['direction']
        
        if direction == 'long':
            liquidation_price = entry_price - (initial_margin / quantity)
        else:
            liquidation_price = entry_price + (initial_margin / quantity)
        
        return liquidation_price
    
    async def _calculate_binance_funding_fee(self, position: Dict) -> float:
        """Calculate Binance funding fee"""
        quantity = abs(position['quantity'])
        funding_rate = position.get('funding_rate', 0.0)
        mark_price = position['current_price']
        
        if funding_rate == 0.0:
            return 0.0
        
        # Funding fee = position value * funding rate
        return quantity * mark_price * funding_rate
    
    async def _calculate_binance_portfolio_funding_rate(self, positions: List[Dict],
                                                      exchange_info: Optional[Dict]) -> float:
        """Calculate average funding rate for portfolio"""
        total_funding = 0.0
        total_notional = 0.0
        
        for pos in positions:
            if 'funding_rate' in pos and pos['funding_rate'] is not None:
                notional = abs(pos['quantity'] * pos['current_price'])
                funding = pos['funding_rate'] * notional
                total_funding += funding
                total_notional += notional
        
        if total_notional > 0:
            return total_funding / total_notional
        
        return 0.0
    
    async def _get_next_binance_funding_time(self, exchange_info: Optional[Dict]) -> Optional[datetime]:
        """Get next funding time"""
        if not exchange_info or 'next_funding_time' not in exchange_info:
            # Default to next 8-hour interval
            now = datetime.now()
            next_funding = now.replace(hour=(now.hour // 8 + 1) * 8, minute=0, second=0, microsecond=0)
            
            if next_funding <= now:
                next_funding += timedelta(hours=8)
            
            return next_funding
        
        return exchange_info['next_funding_time']
    
    def _get_binance_symbol_specs(self, symbol: str,
                                  exchange_info: Optional[Dict] = None) -> Dict:
        """Get Binance symbol specifications"""
        if exchange_info and 'symbols' in exchange_info:
            for sym in exchange_info['symbols']:
                if sym['symbol'] == symbol:
                    return sym
        
        # Default specifications
        return {
            'leverage_bracket': 1,
            'max_leverage': self.config.max_leverage,
            'initial_margin_rate': self.config.binance_initial_margin_rate,
            'maintenance_margin_rate': self.config.binance_maintenance_margin_rate,
            'funding_rate': 0.0,
            'next_funding_time': None
        }
    
    def _get_mt5_symbol_specs(self, symbol: str,
                             exchange_info: Optional[Dict] = None) -> Dict:
        """Get MT5 symbol specifications"""
        if exchange_info and symbol in exchange_info:
            return exchange_info[symbol]
        
        # Default specifications
        return {
            'margin_mode': 'fxf',  # Forex
            'margin_rate': self.config.mt5_initial_margin_rate,
            'swap_long': 0.0,
            'swap_short': 0.0
        }
    
    async def calculate_trade_margin_requirement(self, symbol: str,
                                                quantity: float,
                                                price: float,
                                                direction: str,
                                                exchange: str = 'binance',
                                                exchange_info: Optional[Dict] = None) -> float:
        """
        Calculate margin requirement for a new trade
        
        Args:
            symbol: Symbol to trade
            quantity: Quantity to trade
            price: Entry price
            direction: Trade direction
            exchange: Exchange type
            exchange_info: Exchange-specific information
            
        Returns:
            Margin required in base currency
        """
        position = {
            'symbol': symbol,
            'quantity': quantity,
            'avg_entry_price': price,
            'current_price': price,
            'direction': direction
        }
        
        margin = await self.calculate_position_margin(position, exchange, exchange_info)
        return margin.initial_margin
    
    async def check_margin_requirements(self, portfolio_margin: PortfolioMargin) -> List[Dict]:
        """
        Check margin requirements and generate alerts
        
        Args:
            portfolio_margin: Portfolio margin information
            
        Returns:
            List of margin alerts
        """
        alerts = []
        
        # Check margin level
        if portfolio_margin.margin_level < portfolio_margin.liquidation_threshold:
            alerts.append({
                'type': 'liquidation_imminent',
                'level': 'critical',
                'message': f"Liquidation imminent! Margin level: {portfolio_margin.margin_level:.1%}",
                'action': 'Close positions immediately',
                'suggestion': 'Reduce leverage or add funds'
            })
        elif portfolio_margin.margin_level < portfolio_margin.margin_call_level:
            alerts.append({
                'type': 'margin_call',
                'level': 'high',
                'message': f"Margin call! Margin level: {portfolio_margin.margin_level:.1%}",
                'action': 'Close positions or add funds',
                'suggestion': f"Reduce exposure or deposit at least ${portfolio_margin.used_margin * 0.1:.2f}"
            })
        
        # Check position margins
        for symbol, pos_margin in portfolio_margin.position_margins.items():
            if pos_margin.margin_level < 0.5:
                alerts.append({
                    'type': 'position_liquidation',
                    'level': 'critical',
                    'symbol': symbol,
                    'message': f"{symbol} position liquidation imminent! Margin level: {pos_margin.margin_level:.1%}",
                    'action': 'Close position',
                    'suggestion': f"Close {symbol} position to reduce risk"
                })
            elif pos_margin.margin_level < 1.0:
                alerts.append({
                    'type': 'position_margin_warning',
                    'level': 'medium',
                    'symbol': symbol,
                    'message': f"{symbol} position margin level low: {pos_margin.margin_level:.1%}",
                    'action': 'Consider closing or reducing position',
                    'suggestion': f"Reduce {symbol} position size by 50%"
                })
        
        # Check leverage
        if portfolio_margin.effective_leverage > self.config.max_leverage * 0.8:
            alerts.append({
                'type': 'leverage_warning',
                'level': 'medium',
                'message': f"Leverage approaching maximum! Current: {portfolio_margin.effective_leverage:.1f}x",
                'action': 'Reduce leverage',
                'suggestion': f"Target leverage below {self.config.max_leverage * 0.6:.1f}x"
            })
        
        # Check margin usage
        margin_utilization = portfolio_margin.used_margin / portfolio_margin.equity
        if margin_utilization > 0.8:
            alerts.append({
                'type': 'margin_utilization',
                'level': 'high',
                'message': f"Margin utilization very high: {margin_utilization:.1%}",
                'action': 'Reduce exposure',
                'suggestion': f"Reduce margin usage to below 50%"
            })
        
        return alerts
    
    async def calculate_margin_optimization(self, portfolio_margin: PortfolioMargin) -> List[Dict]:
        """
        Calculate margin optimization suggestions
        
        Args:
            portfolio_margin: Portfolio margin information
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Calculate available margin
        available_margin = portfolio_margin.available_margin
        
        if available_margin < 0:
            # Negative available margin - need to reduce exposure
            total_reduction = abs(available_margin)
            
            # Sort positions by margin utilization
            sorted_positions = sorted(
                portfolio_margin.position_margins.items(),
                key=lambda x: x[1].margin_ratio,
                reverse=True
            )
            
            for symbol, pos in sorted_positions:
                if total_reduction <= 0:
                    break
                
                # Calculate how much to reduce this position
                pos_reduction = min(pos.used_margin, total_reduction)
                reduction_percentage = pos_reduction / pos.used_margin
                
                suggestions.append({
                    'type': 'position_reduction',
                    'symbol': symbol,
                    'current_quantity': pos.quantity,
                    'suggested_quantity': pos.quantity * (1 - reduction_percentage),
                    'reduction_amount': pos_reduction,
                    'rationale': 'Reduce position to restore positive available margin'
                })
                
                total_reduction -= pos_reduction
        
        else:
            # Positive available margin - suggest increasing leverage or adding positions
            if portfolio_margin.effective_leverage < self.config.max_leverage * 0.5:
                suggestions.append({
                    'type': 'leverage_increase',
                    'current_leverage': portfolio_margin.effective_leverage,
                    'suggested_leverage': min(portfolio_margin.effective_leverage * 1.5, self.config.max_leverage),
                    'rationale': 'Available margin suggests room for increased leverage'
                })
        
        # Suggest portfolio margin optimization
        if self.config.portfolio_margin_enabled and len(portfolio_margin.position_margins) > 3:
            suggestions.append({
                'type': 'portfolio_margin_optimization',
                'rationale': 'Consider portfolio margin for better capital efficiency with diverse positions'
            })
        
        return suggestions


# Helper functions
def calculate_position_margin_quick(quantity: float,
                                   price: float,
                                   leverage: float,
                                   margin_rate: float = 0.01) -> float:
    """Quick position margin calculation"""
    notional = quantity * price
    return notional * margin_rate * leverage


def calculate_liquidation_price_quick(entry_price: float,
                                     quantity: float,
                                     margin: float,
                                     direction: str,
                                     maintenance_margin: float = 0.005) -> float:
    """Quick liquidation price calculation"""
    if direction == 'long':
        return entry_price - (margin / quantity)
    else:
        return entry_price + (margin / quantity)


def calculate_max_position_size(balance: float,
                               price: float,
                               leverage: float,
                               margin_rate: float = 0.01,
                               margin_buffer: float = 0.1) -> float:
    """Calculate maximum position size given margin constraints"""
    available_margin = balance * leverage
    notional = available_margin * (1 - margin_buffer)
    return notional / price
