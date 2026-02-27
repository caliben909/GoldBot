"""
DXY Correlation Filter v2.0 - Production-Ready Implementation
Real-time US Dollar Index correlation analysis for forex trading
Optimized for institutional-grade risk management
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class DXYTrend(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class DXYCorrelationConfig:
    """Configuration for DXY correlation filter"""
    enabled: bool = True
    correlation_method: str = 'pearson'  # 'pearson' or 'spearman'
    lookback_period: int = 60  # bars
    min_observations: int = 20
    
    # Symbol-specific expected correlations (negative = inverse, positive = direct)
    # EURUSD, GBPUSD, AUDUSD, NZDUSD, XAUUSD should be NEGATIVE (DXY up = pair down)
    # USDJPY, USDCAD, USDCHF should be POSITIVE (DXY up = pair up)
    symbol_correlation_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'EURUSD': -0.75,
        'GBPUSD': -0.70,
        'AUDUSD': -0.65,
        'NZDUSD': -0.60,
        'USDJPY': 0.60,
        'USDCAD': 0.55,
        'USDCHF': 0.50,
        'XAUUSD': -0.70,  # Gold
        'XAGUSD': -0.65,  # Silver
        'BCOUSD': -0.40,  # Brent Oil
        'WTIUSD': -0.40,  # WTI Oil
    })
    
    # Filtering rules
    filter_on_correlation_strength: bool = True
    minimum_correlation_strength: float = 0.50
    maximum_correlation_strength: float = 0.95
    
    # Position sizing
    adjust_position_size_by_correlation: bool = True
    correlation_sizing_multiplier: float = 0.8
    max_correlation_sizing_reduction: float = 0.5
    
    # Trend confirmation
    require_dxy_trend_confirmation: bool = True
    trend_strength_threshold: float = 0.3
    
    # Alert settings
    alert_on_extreme_correlation: bool = True
    extreme_correlation_threshold: float = 0.90
    alert_cooldown_minutes: int = 60


@dataclass
class DXYCorrelationResult:
    """Result of DXY correlation analysis"""
    timestamp: datetime
    dxy_price: float
    dxy_trend: DXYTrend
    dxy_momentum: float
    dxy_volatility: float
    correlations: Dict[str, float]
    correlation_strengths: Dict[str, float]
    trend_confidences: Dict[str, float]
    eligible_symbols: List[str]
    filtered_symbols: List[str]
    position_adjustments: Dict[str, float]
    portfolio_correlation: float
    diversification_score: float


class DXYCorrelationFilter:
    """
    Production-ready DXY correlation filter for institutional trading
    
    Key Features:
    - Real-time correlation calculation with rolling windows
    - Symbol-specific correlation validation
    - Trend confirmation filtering
    - Dynamic position sizing based on correlation strength
    - Portfolio correlation monitoring
    """
    
    def __init__(self, config: Optional[Union[Dict, DXYCorrelationConfig]] = None):
        if isinstance(config, dict):
            self.config = DXYCorrelationConfig(**config.get('dxy_correlation', {}))
        elif isinstance(config, DXYCorrelationConfig):
            self.config = config
        else:
            self.config = DXYCorrelationConfig()
        
        # Data storage with size limits
        self.max_history = self.config.lookback_period * 2
        self.dxy_prices: pd.Series = pd.Series(dtype=float)
        self.symbol_prices: Dict[str, pd.Series] = {}
        self.correlation_history: List[Dict] = []
        
        # Alert tracking
        self.last_alert_time: Dict[str, datetime] = {}
        
        # Cache for performance
        self._correlation_cache: Dict[str, Tuple[float, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)
        
        logger.info("DXYCorrelationFilter initialized")
    
    def update_dxy(self, price: float, timestamp: Optional[datetime] = None):
        """
        Update DXY price (call this every tick/bar)
        
        Args:
            price: Current DXY price
            timestamp: Optional timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Add to series
        self.dxy_prices = pd.concat([
            self.dxy_prices,
            pd.Series({timestamp: price})
        ])
        
        # Trim to max size
        if len(self.dxy_prices) > self.max_history:
            self.dxy_prices = self.dxy_prices.iloc[-self.max_history:]
        
        # Invalidate cache
        self._correlation_cache.clear()
    
    def update_symbol(self, symbol: str, price: float, timestamp: Optional[datetime] = None):
        """
        Update symbol price for correlation calculation
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            price: Current price
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if symbol not in self.symbol_prices:
            self.symbol_prices[symbol] = pd.Series(dtype=float)
        
        self.symbol_prices[symbol] = pd.concat([
            self.symbol_prices[symbol],
            pd.Series({timestamp: price})
        ])
        
        # Trim to max size
        if len(self.symbol_prices[symbol]) > self.max_history:
            self.symbol_prices[symbol] = self.symbol_prices[symbol].iloc[-self.max_history:]
    
    def update_from_dataframe(self, dxy_df: pd.DataFrame, symbol_df: pd.DataFrame, 
                              symbol: str):
        """
        Bulk update from DataFrames (useful for backtesting)
        
        Args:
            dxy_df: DataFrame with DXY prices (index=datetime, column='close')
            symbol_df: DataFrame with symbol prices
            symbol: Symbol name
        """
        self.dxy_prices = dxy_df['close'] if 'close' in dxy_df.columns else dxy_df.iloc[:, 0]
        self.symbol_prices[symbol] = symbol_df['close'] if 'close' in symbol_df.columns else symbol_df.iloc[:, 0]
        
        # Trim to max size
        if len(self.dxy_prices) > self.max_history:
            self.dxy_prices = self.dxy_prices.iloc[-self.max_history:]
        if len(self.symbol_prices[symbol]) > self.max_history:
            self.symbol_prices[symbol] = self.symbol_prices[symbol].iloc[-self.max_history:]
        
        self._correlation_cache.clear()
    
    def get_correlation(self, symbol: str, lookback: Optional[int] = None) -> float:
        """
        Calculate correlation between symbol and DXY
        
        Args:
            symbol: Trading symbol
            lookback: Optional custom lookback period
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        if not self.config.enabled:
            return 0.0
        
        # Check cache
        cache_key = f"{symbol}_{lookback}"
        if cache_key in self._correlation_cache:
            cached_value, cached_time = self._correlation_cache[cache_key]
            if datetime.now() - cached_time < self._cache_ttl:
                return cached_value
        
        # Validate data
        if symbol not in self.symbol_prices:
            logger.warning(f"No price data for {symbol}")
            return 0.0
        
        if len(self.dxy_prices) < self.config.min_observations:
            logger.debug(f"Insufficient DXY data: {len(self.dxy_prices)}")
            return 0.0
        
        if len(self.symbol_prices[symbol]) < self.config.min_observations:
            logger.debug(f"Insufficient data for {symbol}: {len(self.symbol_prices[symbol])}")
            return 0.0
        
        # Calculate returns
        dxy_returns = self.dxy_prices.pct_change().dropna()
        symbol_returns = self.symbol_prices[symbol].pct_change().dropna()
        
        # Align indices
        common_idx = dxy_returns.index.intersection(symbol_returns.index)
        if len(common_idx) < self.config.min_observations:
            logger.debug(f"Insufficient common data points for {symbol}: {len(common_idx)}")
            return 0.0
        
        # Use specified or default lookback
        period = lookback or self.config.lookback_period
        period = min(period, len(common_idx))
        
        dxy_recent = dxy_returns.loc[common_idx].tail(period)
        symbol_recent = symbol_returns.loc[common_idx].tail(period)
        
        # Calculate correlation
        try:
            if self.config.correlation_method == 'pearson':
                corr, p_value = pearsonr(dxy_recent, symbol_recent)
            else:
                corr, p_value = spearmanr(dxy_recent, symbol_recent)
            
            # Only return significant correlations
            if p_value > 0.05:
                logger.debug(f"{symbol} correlation not significant (p={p_value:.3f})")
                corr = 0.0
            
            # Cache result
            self._correlation_cache[cache_key] = (corr, datetime.now())
            
            return corr if not np.isnan(corr) else 0.0
            
        except Exception as e:
            logger.error(f"Correlation calculation failed for {symbol}: {e}")
            return 0.0
    
    def get_correlation_strength(self, symbol: str) -> float:
        """Get absolute correlation strength"""
        return abs(self.get_correlation(symbol))
    
    def should_filter_signal(self, symbol: str, direction: str) -> Tuple[bool, str]:
        """
        Determine if trading signal should be filtered
        
        Args:
            symbol: Trading symbol
            direction: 'long' or 'short'
            
        Returns:
            (should_filter, reason) tuple
        """
        if not self.config.enabled:
            return False, "Filter disabled"
        
        # Get correlation
        correlation = self.get_correlation(symbol)
        strength = abs(correlation)
        
        # Check minimum data
        if strength == 0:
            return False, "Insufficient data for correlation"
        
        # Check correlation strength bounds
        if self.config.filter_on_correlation_strength:
            if strength < self.config.minimum_correlation_strength:
                return False, f"Weak correlation ({strength:.2f}), no filter applied"
            
            if strength > self.config.maximum_correlation_strength:
                return True, f"Extreme correlation ({strength:.2f}), avoiding crowded trade"
        
        # Get expected correlation from config
        expected_corr = self.config.symbol_correlation_thresholds.get(symbol, 0)
        
        # Determine if correlation aligns with trade direction
        dxy_trend = self.get_dxy_trend()
        
        # For negatively correlated pairs (EURUSD, GBPUSD, etc.)
        if expected_corr < 0:
            if direction == 'long':
                # Long EURUSD = Expect DXY to be weak/correlation negative
                if correlation > -0.3 and dxy_trend == DXYTrend.BULLISH:
                    return True, f"Long {symbol} but DXY bullish (corr: {correlation:.2f})"
            else:  # short
                # Short EURUSD = Expect DXY to be strong/correlation positive
                if correlation < 0.3 and dxy_trend == DXYTrend.BEARISH:
                    return True, f"Short {symbol} but DXY bearish (corr: {correlation:.2f})"
        
        # For positively correlated pairs (USDJPY, USDCAD, etc.)
        elif expected_corr > 0:
            if direction == 'long':
                # Long USDJPY = Expect DXY to be strong
                if correlation < 0.3 and dxy_trend == DXYTrend.BEARISH:
                    return True, f"Long {symbol} but DXY bearish (corr: {correlation:.2f})"
            else:  # short
                # Short USDJPY = Expect DXY to be weak
                if correlation > -0.3 and dxy_trend == DXYTrend.BULLISH:
                    return True, f"Short {symbol} but DXY bullish (corr: {correlation:.2f})"
        
        # Trend confirmation check
        if self.config.require_dxy_trend_confirmation:
            trend_strength = self.get_dxy_trend_strength()
            if trend_strength < self.config.trend_strength_threshold:
                return False, f"Weak DXY trend ({trend_strength:.2f}), allowing signal"
        
        return False, "Correlation alignment acceptable"
    
    def adjust_position_size(self, symbol: str, base_size: float) -> float:
        """
        Adjust position size based on correlation strength
        
        Args:
            symbol: Trading symbol
            base_size: Base position size in lots
            
        Returns:
            Adjusted position size
        """
        if not self.config.adjust_position_size_by_correlation:
            return base_size
        
        strength = self.get_correlation_strength(symbol)
        
        # Reduce size for extreme correlations (crowded trades)
        if strength > 0.85:
            reduction = self.config.max_correlation_sizing_reduction
            logger.debug(f"{symbol}: Extreme correlation ({strength:.2f}), reducing size by {reduction*100:.0f}%")
            return base_size * (1 - reduction)
        
        # Slight reduction for very strong correlations
        if strength > 0.75:
            reduction = strength * self.config.correlation_sizing_multiplier * 0.5
            return base_size * (1 - reduction)
        
        # Slight boost for moderate correlations (good signal)
        if 0.60 < strength < 0.75:
            boost = 0.1
            return base_size * (1 + boost)
        
        return base_size
    
    def get_dxy_trend(self) -> DXYTrend:
        """
        Determine current DXY trend
        
        Returns:
            DXYTrend enum value
        """
        if len(self.dxy_prices) < 20:
            return DXYTrend.NEUTRAL
        
        # Calculate moving averages
        ma_short = self.dxy_prices.tail(10).mean()
        ma_long = self.dxy_prices.tail(30).mean()
        
        # Calculate price change
        price_change_5d = (self.dxy_prices.iloc[-1] - self.dxy_prices.iloc[-5]) / self.dxy_prices.iloc[-5]
        price_change_10d = (self.dxy_prices.iloc[-1] - self.dxy_prices.iloc[-10]) / self.dxy_prices.iloc[-10]
        
        # Trend determination
        bullish_signals = 0
        bearish_signals = 0
        
        if ma_short > ma_long * 1.001:
            bullish_signals += 1
        elif ma_short < ma_long * 0.999:
            bearish_signals += 1
        
        if price_change_5d > 0.005:
            bullish_signals += 1
        elif price_change_5d < -0.005:
            bearish_signals += 1
        
        if price_change_10d > 0.01:
            bullish_signals += 1
        elif price_change_10d < -0.01:
            bearish_signals += 1
        
        # Determine trend
        if bullish_signals >= 2 and bullish_signals > bearish_signals:
            return DXYTrend.BULLISH
        elif bearish_signals >= 2 and bearish_signals > bullish_signals:
            return DXYTrend.BEARISH
        
        return DXYTrend.NEUTRAL
    
    def get_dxy_trend_strength(self) -> float:
        """Calculate DXY trend strength (0-1)"""
        if len(self.dxy_prices) < 20:
            return 0.0
        
        # Use R² of linear regression
        x = np.arange(len(self.dxy_prices.tail(20)))
        y = self.dxy_prices.tail(20).values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        return abs(r_value)
    
    def get_dxy_momentum(self) -> float:
        """Calculate DXY momentum (rate of change)"""
        if len(self.dxy_prices) < 10:
            return 0.0
        
        recent = self.dxy_prices.tail(10).pct_change().mean()
        return recent * 100  # As percentage
    
    def get_dxy_volatility(self) -> float:
        """Calculate DXY volatility (annualized)"""
        if len(self.dxy_prices) < 20:
            return 0.0
        
        returns = self.dxy_prices.tail(20).pct_change().dropna()
        return returns.std() * np.sqrt(252) * 100  # As annualized percentage
    
    def analyze_portfolio(self, positions: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze portfolio correlation risk
        
        Args:
            positions: Dict of symbol -> position size (positive=long, negative=short)
            
        Returns:
            Portfolio correlation analysis
        """
        if not positions:
            return {
                'portfolio_correlation': 0.0,
                'dxy_exposure': 0.0,
                'risk_level': 'low',
                'recommendations': []
            }
        
        weighted_correlations = []
        total_exposure = 0
        
        for symbol, size in positions.items():
            corr = self.get_correlation(symbol)
            exposure = abs(size)
            weighted_correlations.append(corr * exposure)
            total_exposure += exposure
        
        if total_exposure == 0:
            return {
                'portfolio_correlation': 0.0,
                'dxy_exposure': 0.0,
                'risk_level': 'low',
                'recommendations': []
            }
        
        # Portfolio correlation (weighted average)
        portfolio_corr = sum(weighted_correlations) / total_exposure
        
        # DXY exposure (positive = portfolio moves with DXY)
        dxy_exposure = portfolio_corr * total_exposure
        
        # Risk level
        abs_exposure = abs(dxy_exposure)
        if abs_exposure > 5.0:
            risk_level = 'high'
        elif abs_exposure > 2.5:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        # Generate recommendations
        recommendations = []
        if portfolio_corr > 0.7:
            recommendations.append("Portfolio highly correlated with DXY - consider hedging")
        elif portfolio_corr < -0.7:
            recommendations.append("Portfolio highly inverse to DXY - monitor dollar strength")
        
        return {
            'portfolio_correlation': portfolio_corr,
            'dxy_exposure': dxy_exposure,
            'risk_level': risk_level,
            'recommendations': recommendations,
            'dxy_trend': self.get_dxy_trend().value,
            'dxy_momentum': self.get_dxy_momentum()
        }
    
    def check_alerts(self) -> List[Dict]:
        """
        Check for extreme correlation conditions
        
        Returns:
            List of alert dictionaries
        """
        if not self.config.alert_on_extreme_correlation:
            return []
        
        alerts = []
        current_time = datetime.now()
        
        for symbol in self.symbol_prices.keys():
            strength = self.get_correlation_strength(symbol)
            
            if strength >= self.config.extreme_correlation_threshold:
                # Check cooldown
                last_alert = self.last_alert_time.get(symbol)
                if last_alert and (current_time - last_alert).total_seconds() < \
                   self.config.alert_cooldown_minutes * 60:
                    continue
                
                correlation = self.get_correlation(symbol)
                
                alerts.append({
                    'timestamp': current_time,
                    'symbol': symbol,
                    'correlation': correlation,
                    'strength': strength,
                    'dxy_trend': self.get_dxy_trend().value,
                    'message': f"Extreme DXY correlation detected for {symbol}: {correlation:.2f}"
                })
                
                self.last_alert_time[symbol] = current_time
        
        return alerts
    
    def get_full_analysis(self, symbols: Optional[List[str]] = None) -> DXYCorrelationResult:
        """
        Get complete correlation analysis
        
        Args:
            symbols: List of symbols to analyze (default: all tracked)
            
        Returns:
            DXYCorrelationResult with full analysis
        """
        if symbols is None:
            symbols = list(self.symbol_prices.keys())
        
        correlations = {}
        strengths = {}
        trend_confs = {}
        eligible = []
        filtered = []
        adjustments = {}
        
        for symbol in symbols:
            corr = self.get_correlation(symbol)
            strength = abs(corr)
            
            correlations[symbol] = corr
            strengths[symbol] = strength
            
            # Simple trend confidence based on correlation stability
            # (would be more sophisticated in production)
            trend_confs[symbol] = strength
            
            # Check eligibility
            should_filter, reason = self.should_filter_signal(symbol, 'long')  # Test long
            if should_filter:
                # Test short
                should_filter_short, _ = self.should_filter_signal(symbol, 'short')
                if should_filter_short:
                    filtered.append(symbol)
                else:
                    eligible.append(symbol)
            else:
                eligible.append(symbol)
            
            # Position adjustment
            adjustments[symbol] = self.adjust_position_size(symbol, 1.0)
        
        # Portfolio metrics
        if correlations:
            avg_corr = np.mean([abs(c) for c in correlations.values()])
            diversification = 1 - avg_corr
        else:
            avg_corr = 0.0
            diversification = 1.0
        
        return DXYCorrelationResult(
            timestamp=datetime.now(),
            dxy_price=self.dxy_prices.iloc[-1] if len(self.dxy_prices) > 0 else 0.0,
            dxy_trend=self.get_dxy_trend(),
            dxy_momentum=self.get_dxy_momentum(),
            dxy_volatility=self.get_dxy_volatility(),
            correlations=correlations,
            correlation_strengths=strengths,
            trend_confidences=trend_confs,
            eligible_symbols=eligible,
            filtered_symbols=filtered,
            position_adjustments=adjustments,
            portfolio_correlation=avg_corr,
            diversification_score=diversification
        )
    
    def get_state(self) -> Dict:
        """Get current filter state for serialization"""
        return {
            'config': {
                'enabled': self.config.enabled,
                'method': self.config.correlation_method,
                'lookback': self.config.lookback_period
            },
            'dxy_price': self.dxy_prices.iloc[-1] if len(self.dxy_prices) > 0 else None,
            'dxy_trend': self.get_dxy_trend().value,
            'tracked_symbols': list(self.symbol_prices.keys()),
            'last_alert_times': {k: v.isoformat() for k, v in self.last_alert_time.items()}
        }


# ============================================================================
# INTEGRATION WITH RISK ENGINE
# ============================================================================

class DXYIntegratedRiskEngine:
    """
    Risk Engine with integrated DXY correlation filtering
    (Extends the RiskEngine from previous code)
    """
    
    def __init__(self, config: dict):
        # Initialize base risk engine components
        self.config = config
        self.risk_config = config.get('risk_management', {})
        
        # Initialize DXY filter
        dxy_config = self.risk_config.get('dxy_correlation', {})
        self.dxy_filter = DXYCorrelationFilter(dxy_config)
        
        # Other risk engine initialization...
        self.correlation_engine = None  # Would be initialized here
        self.current_balance = self.risk_config.get('initial_balance', 10000)
        self.open_trades = {}
        
        logger.info("DXYIntegratedRiskEngine initialized")
    
    def update_market_data(self, symbol: str, price: float, 
                          dxy_price: Optional[float] = None,
                          timestamp: Optional[datetime] = None):
        """
        Update market data for risk engine
        
        Args:
            symbol: Trading symbol
            price: Symbol price
            dxy_price: Optional DXY price
            timestamp: Optional timestamp
        """
        # Update DXY filter
        self.dxy_filter.update_symbol(symbol, price, timestamp)
        
        if dxy_price is not None:
            self.dxy_filter.update_dxy(dxy_price, timestamp)
        
        # Check for alerts
        alerts = self.dxy_filter.check_alerts()
        for alert in alerts:
            logger.warning(f"DXY ALERT: {alert['message']}")
    
    def calculate_position_size(self, symbol: str, entry_price: float,
                               stop_loss: float, take_profit: float,
                               confidence: float, direction: str,
                               **kwargs) -> Dict[str, Any]:
        """
        Calculate position size with DXY correlation adjustment
        """
        # Base calculation (simplified)
        risk_per_trade = self.risk_config.get('risk_per_trade', 0.01)
        risk_amount = self.current_balance * risk_per_trade * confidence
        
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance == 0:
            return {'error': 'Invalid stop loss'}
        
        base_size = risk_amount / (stop_distance * 10000 * 10)  # Simplified forex calc
        
        # Apply DXY correlation filter
        should_filter, reason = self.dxy_filter.should_filter_signal(symbol, direction)
        if should_filter:
            return {
                'error': f'DXY Filter: {reason}',
                'position_size': 0.0,
                'filtered': True,
                'filter_reason': reason
            }
        
        # Adjust for correlation
        adjusted_size = self.dxy_filter.adjust_position_size(symbol, base_size)
        
        # Calculate risk metrics
        risk_amount = adjusted_size * stop_distance * 10000 * 10
        
        return {
            'position_size': adjusted_size,
            'base_size': base_size,
            'risk_amount': risk_amount,
            'risk_percentage': (risk_amount / self.current_balance) * 100,
            'dxy_correlation': self.dxy_filter.get_correlation(symbol),
            'dxy_trend': self.dxy_filter.get_dxy_trend().value,
            'dxy_adjustment': adjusted_size / base_size if base_size > 0 else 1.0
        }
    
    def get_portfolio_risk(self) -> Dict:
        """Get portfolio risk including DXY exposure"""
        positions = {sym: trade.get('size', 0) for sym, trade in self.open_trades.items()}
        dxy_analysis = self.dxy_filter.analyze_portfolio(positions)
        
        return {
            **dxy_analysis,
            'total_exposure': sum(abs(p) for p in positions.values()),
            'open_positions': len(positions)
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'dxy_correlation': {
            'enabled': True,
            'correlation_method': 'pearson',
            'lookback_period': 60,
            'minimum_correlation_strength': 0.50,
            'require_dxy_trend_confirmation': True,
            'adjust_position_size_by_correlation': True
        }
    }
    
    # Initialize filter
    dxy_filter = DXYCorrelationFilter(CONFIG)
    
    # Simulate market data updates
    import random
    
    # Generate synthetic DXY data (trending up)
    base_dxy = 100.0
    for i in range(100):
        dxy_price = base_dxy + i * 0.05 + random.gauss(0, 0.2)
        dxy_filter.update_dxy(dxy_price)
    
    # Generate synthetic EURUSD data (should be negatively correlated)
    base_eur = 1.10
    for i in range(100):
        # EURUSD inversely correlated with DXY
        eur_price = base_eur - i * 0.0004 + random.gauss(0, 0.001)
        dxy_filter.update_symbol('EURUSD', eur_price)
    
    # Generate synthetic USDJPY data (should be positively correlated)
    base_jpy = 150.0
    for i in range(100):
        # USDJPY positively correlated with DXY
        jpy_price = base_jpy + i * 0.08 + random.gauss(0, 0.3)
        dxy_filter.update_symbol('USDJPY', jpy_price)
    
    # Test correlations
    print("DXY Correlation Analysis")
    print("=" * 50)
    
    eur_corr = dxy_filter.get_correlation('EURUSD')
    print(f"EURUSD Correlation: {eur_corr:.3f} (Expected: negative)")
    
    jpy_corr = dxy_filter.get_correlation('USDJPY')
    print(f"USDJPY Correlation: {jpy_corr:.3f} (Expected: positive)")
    
    # Test signal filtering
    print("\nSignal Filter Tests:")
    print("-" * 50)
    
    should_filter, reason = dxy_filter.should_filter_signal('EURUSD', 'long')
    print(f"Long EURUSD: {'FILTERED' if should_filter else 'ALLOWED'} - {reason}")
    
    should_filter, reason = dxy_filter.should_filter_signal('EURUSD', 'short')
    print(f"Short EURUSD: {'FILTERED' if should_filter else 'ALLOWED'} - {reason}")
    
    should_filter, reason = dxy_filter.should_filter_signal('USDJPY', 'long')
    print(f"Long USDJPY: {'FILTERED' if should_filter else 'ALLOWED'} - {reason}")
    
    # Test position sizing
    print("\nPosition Size Adjustments:")
    print("-" * 50)
    
    base_size = 1.0
    eur_adjusted = dxy_filter.adjust_position_size('EURUSD', base_size)
    print(f"EURUSD: {base_size:.2f} -> {eur_adjusted:.2f} lots")
    
    jpy_adjusted = dxy_filter.adjust_position_size('USDJPY', base_size)
    print(f"USDJPY: {base_size:.2f} -> {jpy_adjusted:.2f} lots")
    
    # Full analysis
    print("\nFull Analysis:")
    print("-" * 50)
    analysis = dxy_filter.get_full_analysis(['EURUSD', 'USDJPY'])
    print(f"DXY Trend: {analysis.dxy_trend.value}")
    print(f"DXY Momentum: {analysis.dxy_momentum:.2f}%")
    print(f"Portfolio Correlation: {analysis.portfolio_correlation:.3f}")
    print(f"Diversification Score: {analysis.diversification_score:.3f}")