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
       