"""
Risk Engine - Institutional-grade risk management system with dynamic controls
Features:
- Dynamic position sizing (Kelly, ATR-based, Fixed)
- Real-time portfolio risk tracking
- Correlation-based risk reduction
- Drawdown limits (daily, weekly, monthly)
- Consecutive loss scaling
- Breakeven management
- Partial close optimization
- Trailing stop logic
- VaR and Expected Shortfall calculations
- Stress testing
- Advanced margin calculations (exchange-specific)
- Dynamic correlation analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
from collections import deque, defaultdict
import json
import pickle
from scipy import stats
from scipy.optimize import minimize
import warnings

from core.risk.dynamic_correlation import DynamicCorrelationEngine
from core.risk.margin_calculator import MarginCalculator
from core.risk.dxy_correlation_filter import DXYCorrelationFilter

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class RiskModel(Enum):
    """Risk calculation models"""
    KELLY = "kelly"
    ATR = "atr"
    FIXED = "fixed"
    OPTIMAL_F = "optimal_f"
    VAR = "var"


class PositionSizingMethod(Enum):
    """Position sizing methods"""
    FIXED_RISK = "fixed_risk"
    PERCENTAGE_RISK = "percentage_risk"
    KELLY = "kelly"
    OPTIMAL_F = "optimal_f"
    ATR_BASED = "atr_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    var_95: float  # Value at Risk (95%)
    var_99: float  # Value at Risk (99%)
    expected_shortfall: float  # Conditional VaR
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    max_drawdown: float
    current_drawdown: float
    volatility: float
    beta: float
    alpha: float
    correlation_matrix: Dict[str, float]
    kelly_fraction: float
    optimal_f: float
    risk_of_ruin: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    win_rate: float
    expectancy: float


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_exposure: float
    net_exposure: float
    gross_exposure: float
    leverage: float
    buying_power_used: float
    margin_used: float
    correlation_risk: float
    concentration_risk: float
    sector_exposure: Dict[str, float]
    var_95: float
    var_99: float
    expected_shortfall: float


@dataclass
class TradeRisk:
    """Individual trade risk metrics"""
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float
    risk_percentage: float
    r_multiple: float
    risk_reward: float
    correlation_risk: float
    volatility_risk: float
    liquidity_risk: float


class RiskEngine:
    """
    Professional risk management engine with comprehensive controls
    
    Features:
    - Multi-model position sizing (Kelly, ATR, Fixed)
    - Real-time portfolio risk tracking
    - Dynamic drawdown management
    - Correlation-based risk reduction
    - Consecutive loss protection
    - Automated breakeven and trailing stops
    - VaR and Expected Shortfall calculations
    - Stress testing and scenario analysis
    - Position limits and concentration controls
    - Margin and leverage management
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_configuration()
        
        # Initialize specialized risk engines
        self.correlation_engine = DynamicCorrelationEngine(config)
        self.margin_calculator = MarginCalculator(config)
        self.dxy_correlation_filter = DXYCorrelationFilter(config)
        
        # Trade tracking
        self.trade_history: List[Dict] = []
        self.open_trades: Dict[str, Dict] = {}
        self.closed_trades: List[Dict] = []
        
        # Equity tracking
        self.equity_curve: pd.Series = pd.Series()
        self.daily_equity: Dict[str, List[float]] = defaultdict(list)
        self.weekly_equity: Dict[str, List[float]] = defaultdict(list)
        self.monthly_equity: Dict[str, List[float]] = defaultdict(list)
        
        # Performance tracking
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.monthly_pnl: float = 0.0
        self.total_pnl: float = 0.0
        
        # Streak tracking
        self.current_streak: int = 0
        self.max_consecutive_wins: int = 0
        self.max_consecutive_losses: int = 0
        self.loss_streak_count: int = 0
        
        # Drawdown tracking
        self.peak_equity: float = 0.0
    
    def _validate_configuration(self):
        """Validate risk management configuration"""
        # Validate risk management parameters
        risk_config = self.config['risk_management']
        
        # Validate position sizing
        if 'position_sizing' in risk_config:
            sizing_config = risk_config['position_sizing']
            if sizing_config['method'] not in ['fixed_risk', 'percentage_risk', 'kelly', 'optimal_f', 'atr_based', 'volatility_adjusted']:
                raise ValueError(f"Invalid position sizing method: {sizing_config['method']}")
            if not isinstance(sizing_config['risk_per_trade'], (int, float)) or sizing_config['risk_per_trade'] <= 0 or sizing_config['risk_per_trade'] > 10:
                raise ValueError(f"Invalid risk per trade: {sizing_config['risk_per_trade']} (must be between 0 and 10)")
        
        # Validate drawdown limits
        if 'drawdown_limits' in risk_config:
            dd_config = risk_config['drawdown_limits']
            if not isinstance(dd_config['daily'], (int, float)) or dd_config['daily'] <= 0 or dd_config['daily'] > 50:
                raise ValueError(f"Invalid daily drawdown limit: {dd_config['daily']} (must be between 0 and 50)")
            if not isinstance(dd_config['weekly'], (int, float)) or dd_config['weekly'] <= 0 or dd_config['weekly'] > 100:
                raise ValueError(f"Invalid weekly drawdown limit: {dd_config['weekly']} (must be between 0 and 100)")
            if not isinstance(dd_config['monthly'], (int, float)) or dd_config['monthly'] <= 0 or dd_config['monthly'] > 200:
                raise ValueError(f"Invalid monthly drawdown limit: {dd_config['monthly']} (must be between 0 and 200)")
        
        # Validate DXY correlation filter configuration
        if 'dxy_correlation' in risk_config:
            dxy_config = risk_config['dxy_correlation']
            if not isinstance(dxy_config['enabled'], bool):
                raise ValueError(f"Invalid DXY correlation filter enabled: {dxy_config['enabled']} (must be boolean)")
            if dxy_config['correlation_method'] not in ['pearson', 'spearman']:
                raise ValueError(f"Invalid correlation method: {dxy_config['correlation_method']} (must be pearson or spearman)")
            if not isinstance(dxy_config['lookback_period'], int) or dxy_config['lookback_period'] < 10 or dxy_config['lookback_period'] > 200:
                raise ValueError(f"Invalid correlation lookback period: {dxy_config['lookback_period']} (must be between 10 and 200)")
            if not isinstance(dxy_config['minimum_correlation_strength'], (int, float)) or dxy_config['minimum_correlation_strength'] < 0 or dxy_config['minimum_correlation_strength'] > 1:
                raise ValueError(f"Invalid minimum correlation strength: {dxy_config['minimum_correlation_strength']} (must be between 0 and 1)")
            if not isinstance(dxy_config['maximum_correlation_strength'], (int, float)) or dxy_config['maximum_correlation_strength'] <= 0 or dxy_config['maximum_correlation_strength'] > 1:
                raise ValueError(f"Invalid maximum correlation strength: {dxy_config['maximum_correlation_strength']} (must be between 0 and 1)")
            if dxy_config['minimum_correlation_strength'] >= dxy_config['maximum_correlation_strength']:
                raise ValueError(f"Minimum correlation strength ({dxy_config['minimum_correlation_strength']}) must be less than maximum ({dxy_config['maximum_correlation_strength']})")
        
        logger.info("Risk management configuration validated successfully")
    
    async def update_dxy_data(self, prices: pd.Series):
        """Update DXY price history for correlation analysis"""
        await self.dxy_correlation_filter.update_dxy_data(prices)
    
    async def update_symbol_data_for_correlation(self, symbol: str, prices: pd.Series):
        """Update symbol price history for DXY correlation analysis"""
        await self.dxy_correlation_filter.update_symbol_data(symbol, prices)
    
    async def should_filter_signal_by_dxy_correlation(self, signal: Dict) -> Tuple[bool, str]:
        """
        Determine if a trading signal should be filtered based on DXY correlation
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            (should_filter, reason) tuple
        """
        return await self.dxy_correlation_filter.should_filter_signal(signal)
    
    async def adjust_position_size_by_dxy_correlation(self, symbol: str, base_size: float) -> float:
        """
        Adjust position size based on DXY correlation
        
        Args:
            symbol: Trading symbol
            base_size: Base position size
            
        Returns:
            Adjusted position size
        """
        return await self.dxy_correlation_filter.adjust_position_size(symbol, base_size)
    
    async def calculate_dxy_correlations(self, symbols: List[str]) -> Any:
        """
        Calculate DXY correlations for multiple symbols
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            DXYCorrelationResult object with analysis
        """
        return await self.dxy_correlation_filter.calculate_correlations(symbols)
    
    async def check_extreme_dxy_correlation(self) -> List[Dict]:
        """Check for extreme DXY correlation events"""
        return await self.dxy_correlation_filter.check_extreme_correlation()
    
    async def calculate_dynamic_position_size(self, 
                                           symbol: str,
                                           entry_price: float,
                                           stop_loss: float,
                                           confidence: float = 0.5,
                                           account_balance: float = 10000,
                                           contract_size: float = 1000,
                                           spread: float = 0.0,
                                           margin_requirement: float = 0.01) -> Dict:
        """
        Calculate dynamic position size based on confidence, balance, lot size, 
        contract size, margin, and spread
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            confidence: Prediction confidence (0-1)
            account_balance: Account balance in base currency
            contract_size: Contract size for the instrument
            spread: Spread in pips or points
            margin_requirement: Margin requirement as a decimal (0.01 = 1%)
            
        Returns:
            Dict containing calculated position size and related metrics
        """
        try:
            # Calculate base position size based on fixed risk
            base_size = await self._calculate_base_position_size(
                entry_price, stop_loss, account_balance
            )
            
            # Adjust position size based on confidence
            confidence_adjusted_size = await self._adjust_size_by_confidence(
                base_size, confidence
            )
            
            # Adjust for margin requirements
            margin_adjusted_size = await self._adjust_size_by_margin(
                confidence_adjusted_size, entry_price, contract_size, margin_requirement, account_balance
            )
            
            # Adjust for spread impact
            spread_adjusted_size = await self._adjust_size_by_spread(
                margin_adjusted_size, entry_price, stop_loss, spread
            )
            
            # Calculate final metrics
            risk_amount = await self._calculate_risk_amount(
                spread_adjusted_size, entry_price, stop_loss, contract_size
            )
            
            margin_used = await self._calculate_margin_used(
                spread_adjusted_size, entry_price, contract_size, margin_requirement
            )
            
            # Ensure position size is within valid range
            final_size = await self._validate_position_size(
                spread_adjusted_size, symbol, account_balance
            )
            
            return {
                'symbol': symbol,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'confidence': confidence,
                'base_position_size': base_size,
                'confidence_adjusted_size': confidence_adjusted_size,
                'margin_adjusted_size': margin_adjusted_size,
                'spread_adjusted_size': spread_adjusted_size,
                'final_position_size': final_size,
                'risk_amount': risk_amount,
                'risk_percentage': (risk_amount / account_balance) * 100,
                'margin_used': margin_used,
                'margin_percentage': (margin_used / account_balance) * 100,
                'account_balance': account_balance,
                'contract_size': contract_size,
                'spread': spread,
                'margin_requirement': margin_requirement
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate dynamic position size: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'final_position_size': 0.0
            }
    
    async def _calculate_base_position_size(self, entry_price: float, stop_loss: float, 
                                           account_balance: float) -> float:
        """Calculate base position size using fixed risk method"""
        risk_config = self.config['risk_management']['position_sizing']
        risk_per_trade = risk_config['risk_per_trade']
        
        # Calculate risk per trade in currency
        risk_currency = (account_balance * risk_per_trade) / 100
        
        # Calculate position size based on risk and stop loss distance
        risk_distance = abs(entry_price - stop_loss)
        if risk_distance == 0:
            return 0.0
            
        position_size = risk_currency / risk_distance
        
        return max(position_size, 0.001)  # Minimum 0.001 lot
    
    async def _adjust_size_by_confidence(self, base_size: float, confidence: float) -> float:
        """Adjust position size based on prediction confidence"""
        # Confidence-based adjustment: 0.5 (minimum) to 1.0 (maximum)
        if confidence < 0.5:
            return 0.0  # Reject low confidence signals
        
        # Scale position size linearly with confidence
        adjustment_factor = (confidence - 0.5) * 2.0  # 0 to 1 range
        adjusted_size = base_size * (1.0 + adjustment_factor)
        
        return adjusted_size
    
    async def _adjust_size_by_margin(self, base_size: float, entry_price: float, 
                                     contract_size: float, margin_requirement: float,
                                     account_balance: float) -> float:
        """Adjust position size based on margin requirements"""
        # Calculate maximum position size based on margin
        max_position_value = account_balance / margin_requirement
        max_position_size = max_position_value / (entry_price * contract_size)
        
        return min(base_size, max_position_size)
    
    async def _adjust_size_by_spread(self, base_size: float, entry_price: float, 
                                     stop_loss: float, spread: float) -> float:
        """Adjust position size based on spread impact"""
        # If spread is negligible, return base size
        if spread == 0:
            return base_size
            
        # Calculate risk distance in points
        risk_distance = abs(entry_price - stop_loss)
        
        # If spread is a significant percentage of risk distance, reduce position size
        spread_percentage = (spread / risk_distance) * 100
        
        if spread_percentage > 5:  # If spread is more than 5% of risk distance
            adjustment_factor = 1.0 - (spread_percentage / 100)
            return base_size * adjustment_factor
            
        return base_size
    
    async def _calculate_risk_amount(self, position_size: float, entry_price: float, 
                                     stop_loss: float, contract_size: float) -> float:
        """Calculate total risk amount for the position"""
        risk_distance = abs(entry_price - stop_loss)
        return position_size * contract_size * risk_distance
    
    async def _calculate_margin_used(self, position_size: float, entry_price: float, 
                                    contract_size: float, margin_requirement: float) -> float:
        """Calculate margin used for the position"""
        position_value = position_size * contract_size * entry_price
        return position_value * margin_requirement
    
    async def _validate_position_size(self, position_size: float, symbol: str, 
                                     account_balance: float) -> float:
        """Validate position size is within reasonable limits"""
        # Maximum position size based on account balance (prevents over-leveraging)
        max_position_size = (account_balance * 0.1) / 100  # 0.1% of account per lot
        
        # Minimum position size (usually 0.001 lot)
        min_position_size = 0.001
        
        return max(min(position_size, max_position_size), min_position_size)
       