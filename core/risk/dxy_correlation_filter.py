"""
DXY Correlation Filter - Specialized correlation filter for US Dollar Index
Features:
- Real-time DXY correlation monitoring
- Symbol-specific correlation thresholds
- Time-frame based correlation analysis
- DXY trend and momentum analysis
- Filter for trading signals based on DXY relationship
- Correlation-based position sizing adjustment
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


@dataclass
class DXYCorrelationConfig:
    """Configuration for DXY correlation filter"""
    # Enable/disable the filter
    enabled: bool = True
    
    # Correlation calculation settings
    correlation_method: str = 'pearson'  # 'pearson', 'spearman'
    lookback_period: int = 60  # days
    min_observations: int = 20
    
    # Symbol-specific correlation thresholds
    symbol_correlation_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'EURUSD': -0.85,
        'GBPUSD': -0.75,
        'USDJPY': 0.65,
        'AUDUSD': -0.70,
        'USDCAD': 0.60,
        'NZDUSD': -0.65,
        'USDCHF': 0.55,
        'XAUUSD': -0.80,
        'XAGUSD': -0.70
    })
    
    # Filtering rules
    filter_on_correlation_strength: bool = True
    minimum_correlation_strength: float = 0.6
    maximum_correlation_strength: float = 0.95
    
    # Trend and momentum filters
    require_dxy_trend_confirmation: bool = True
    trend_strength_threshold: float = 0.3
    momentum_threshold: float = 0.1
    
    # Time-frame specific settings
    session_based_correlation: bool = True
    session_correlation_weights: Dict[str, float] = field(default_factory=lambda: {
        'asia': 0.5,
        'london': 1.0,
        'ny': 1.2,
        'overlap': 1.5
    })
    
    # Position sizing adjustment
    adjust_position_size_by_correlation: bool = True
    correlation_sizing_multiplier: float = 0.8
    max_correlation_sizing_reduction: float = 0.5
    
    # Alert settings
    alert_on_extreme_correlation: bool = True
    extreme_correlation_threshold: float = 0.9
    alert_frequency_minutes: int = 60
    
    # Risk management
    max_portfolio_dxy_correlation: float = 0.7
    correlation_diversification_target: float = 0.3


@dataclass
class DXYCorrelationResult:
    """Result of DXY correlation analysis"""
    timestamp: datetime
    dxy_price: float
    dxy_returns: pd.Series
    correlations: Dict[str, float]
    correlation_strengths: Dict[str, float]
    symbol_trend_confidence: Dict[str, float]
    dxy_trend: str
    dxy_momentum: float
    dxy_volatility: float
    eligible_symbols: List[str]
    filtered_symbols: List[str]
    position_size_adjustments: Dict[str, float]
    portfolio_correlation: float
    diversification_score: float


class DXYCorrelationFilter:
    """
    Specialized correlation filter for US Dollar Index (DXY)
    
    Features:
    - Real-time DXY correlation monitoring
    - Symbol-specific correlation thresholds
    - Time-frame based correlation analysis
    - DXY trend and momentum analysis
    - Filter for trading signals based on DXY relationship
    - Correlation-based position sizing adjustment
    """
    
    def __init__(self, config: dict):
        self.config = DXYCorrelationConfig(**config['risk_management']['dxy_correlation'])
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.dxy_price_history: pd.Series = pd.Series()
        self.symbol_price_history: Dict[str, pd.Series] = {}
        self.correlation_history: Dict[datetime, Dict[str, float]] = {}
        self.last_alert_time: Dict[str, datetime] = {}
        
        logger.info("DXYCorrelationFilter initialized")
    
    async def update_dxy_data(self, prices: pd.Series):
        """Update DXY price history"""
        self.dxy_price_history = prices
        logger.debug(f"DXY price history updated with {len(prices)} records")
    
    async def update_symbol_data(self, symbol: str, prices: pd.Series):
        """Update price history for a specific symbol"""
        self.symbol_price_history[symbol] = prices
        logger.debug(f"{symbol} price history updated with {len(prices)} records")
    
    async def calculate_correlations(self, symbols: List[str], 
                                   as_of: Optional[datetime] = None) -> DXYCorrelationResult:
        """
        Calculate DXY correlations for multiple symbols with improved accuracy
        
        Args:
            symbols: List of symbols to analyze
            as_of: Date to calculate correlations for
            
        Returns:
            DXYCorrelationResult object with analysis
        """
        if as_of is None:
            as_of = datetime.now()
        
        # Check if we have DXY data
        if self.dxy_price_history.empty:
            logger.warning("No DXY price history available for correlation calculation")
            return self._empty_correlation_result(symbols, as_of)
        
        # Calculate DXY returns
        dxy_returns = self.dxy_price_history.pct_change().dropna()
        
        correlations = {}
        correlation_strengths = {}
        symbol_trend_confidence = {}
        eligible_symbols = []
        filtered_symbols = []
        position_size_adjustments = {}
        correlation_stabilities = {}
        
        # Calculate correlations for each symbol
        for symbol in symbols:
            if symbol not in self.symbol_price_history:
                logger.warning(f"No price history for symbol: {symbol}")
                continue
                
            symbol_prices = self.symbol_price_history[symbol]
            symbol_returns = symbol_prices.pct_change().dropna()
            
            # Align data
            common_dates = dxy_returns.index.intersection(symbol_returns.index)
            if len(common_dates) < self.config.min_observations:
                logger.debug(f"Not enough common data points for {symbol}: {len(common_dates)}")
                continue
                
            # Calculate correlation with significance test
            if self.config.correlation_method == 'pearson':
                corr, p_value = pearsonr(
                    dxy_returns.loc[common_dates],
                    symbol_returns.loc[common_dates]
                )
            else:  # spearman
                corr, p_value = spearmanr(
                    dxy_returns.loc[common_dates],
                    symbol_returns.loc[common_dates]
                )
            
            # Calculate correlation stability using rolling windows
            stability = await self._calculate_correlation_stability(symbol, common_dates)
            
            # Calculate correlation strength (absolute value)
            strength = abs(corr)
            
            correlations[symbol] = corr
            correlation_strengths[symbol] = strength
            correlation_stabilities[symbol] = stability
            
            # Check eligibility based on correlation
            eligible = await self._is_symbol_eligible(symbol, corr, strength, stability, p_value)
            if eligible:
                eligible_symbols.append(symbol)
                
                # Calculate position size adjustment
                adjustment = await self._calculate_position_size_adjustment(symbol, corr, strength, stability)
                position_size_adjustments[symbol] = adjustment
            else:
                filtered_symbols.append(symbol)
                
            # Calculate trend confidence
            trend_confidence = await self._calculate_trend_confidence(symbol, corr)
            symbol_trend_confidence[symbol] = trend_confidence
        
        # Calculate DXY metrics
        dxy_trend = await self._detect_dxy_trend()
        dxy_momentum = await self._calculate_dxy_momentum()
        dxy_volatility = await self._calculate_dxy_volatility()
        
        # Calculate portfolio-level metrics
        portfolio_correlation, diversification_score = await self._calculate_portfolio_metrics(
            eligible_symbols, correlations
        )
        
        # Get current DXY price
        current_dxy_price = self.dxy_price_history.iloc[-1] if not self.dxy_price_history.empty else 0.0
        
        result = DXYCorrelationResult(
            timestamp=as_of,
            dxy_price=current_dxy_price,
            dxy_returns=dxy_returns,
            correlations=correlations,
            correlation_strengths=correlation_strengths,
            symbol_trend_confidence=symbol_trend_confidence,
            dxy_trend=dxy_trend,
            dxy_momentum=dxy_momentum,
            dxy_volatility=dxy_volatility,
            eligible_symbols=eligible_symbols,
            filtered_symbols=filtered_symbols,
            position_size_adjustments=position_size_adjustments,
            portfolio_correlation=portfolio_correlation,
            diversification_score=diversification_score
        )
        
        # Add stability information to metadata
        setattr(result, 'correlation_stabilities', correlation_stabilities)
        
        return result
    
    async def _calculate_correlation_stability(self, symbol: str, common_dates: pd.DatetimeIndex) -> float:
        """Calculate correlation stability using rolling windows"""
        symbol_prices = self.symbol_price_history[symbol]
        symbol_returns = symbol_prices.pct_change().dropna()
        dxy_returns = self.dxy_price_history.pct_change().dropna()
        
        # Calculate rolling correlations over different time windows
        window_sizes = [10, 20, 30]
        correlation_stabilities = []
        
        for window in window_sizes:
            if len(common_dates) >= window * 2:
                # Calculate correlations for two consecutive windows
                symbol_returns1 = symbol_returns.loc[common_dates[-window*2:-window]]
                dxy_returns1 = dxy_returns.loc[common_dates[-window*2:-window]]
                
                symbol_returns2 = symbol_returns.loc[common_dates[-window:]]
                dxy_returns2 = dxy_returns.loc[common_dates[-window:]]
                
                # Calculate correlations for each window
                if self.config.correlation_method == 'pearson':
                    corr1, _ = pearsonr(dxy_returns1, symbol_returns1)
                    corr2, _ = pearsonr(dxy_returns2, symbol_returns2)
                else:
                    corr1, _ = spearmanr(dxy_returns1, symbol_returns1)
                    corr2, _ = spearmanr(dxy_returns2, symbol_returns2)
                
                # Calculate stability score (1 - absolute difference)
                stability = 1 - abs(corr1 - corr2)
                correlation_stabilities.append(stability)
        
        if correlation_stabilities:
            return np.mean(correlation_stabilities)
        return 0.5
    
    async def _is_symbol_eligible(self, symbol: str, correlation: float, 
                                strength: float, stability: float, p_value: float) -> bool:
        """Check if symbol is eligible based on correlation criteria with improved validation"""
        # Check significance level
        if p_value > 0.05:
            logger.debug(f"{symbol} filtered: Correlation not statistically significant (p-value: {p_value:.2f})")
            return False
        
        # Check stability
        if stability < 0.6:
            logger.debug(f"{symbol} filtered: Correlation stability too low ({stability:.2f})")
            return False
        
        # Check if symbol has specific threshold
        if symbol in self.config.symbol_correlation_thresholds:
            threshold = self.config.symbol_correlation_thresholds[symbol]
            expected_correlation_sign = 1 if threshold > 0 else -1
            actual_correlation_sign = 1 if correlation > 0 else -1
            
            # Check if correlation has expected sign and meets minimum strength
            if expected_correlation_sign == actual_correlation_sign and \
               abs(correlation) >= abs(threshold):
                return True
            else:
                logger.debug(f"{symbol} filtered: Correlation {correlation:.2f} doesn't match expected sign/direction")
                return False
        
        # Default eligibility check
        if self.config.filter_on_correlation_strength:
            if strength < self.config.minimum_correlation_strength or \
               strength > self.config.maximum_correlation_strength:
                logger.debug(f"{symbol} filtered: Correlation strength {strength:.2f} outside range")
                return False
        
        return True
    
    async def _calculate_position_size_adjustment(self, symbol: str, 
                                                 correlation: float, 
                                                 strength: float, 
                                                 stability: float) -> float:
        """Calculate position size adjustment based on correlation and stability"""
        if not self.config.adjust_position_size_by_correlation:
            return 1.0
        
        # Base adjustment on correlation strength
        adjustment = 1.0 - (strength * self.config.correlation_sizing_multiplier)
        
        # Adjust based on stability (lower stability = more reduction)
        stability_factor = 0.5 + (stability * 0.5)  # Scale from 0.5 to 1.0
        adjustment *= stability_factor
        
        # Ensure adjustment doesn't drop below minimum
        adjustment = max(1.0 - self.config.max_correlation_sizing_reduction, adjustment)
        
        logger.debug(f"{symbol} position size adjustment: {adjustment:.2f} (strength: {strength:.2f}, stability: {stability:.2f})")
        return adjustment
    
    async def _calculate_trend_confidence(self, symbol: str, correlation: float) -> float:
        """Calculate dynamic trend confidence based on correlation consistency and stability"""
        if symbol not in self.symbol_price_history or self.dxy_price_history.empty:
            return 0.0
        
        # Get price history for symbol and DXY
        symbol_prices = self.symbol_price_history[symbol]
        dxy_prices = self.dxy_price_history
        
        # Calculate rolling correlations over different time windows to measure stability
        window_sizes = [10, 20, 30]
        correlation_stability_scores = []
        
        for window in window_sizes:
            if len(symbol_prices) >= window * 2 and len(dxy_prices) >= window * 2:
                # Calculate correlations for two consecutive windows
                symbol_returns1 = symbol_prices.pct_change().dropna()[-window*2:-window]
                dxy_returns1 = dxy_prices.pct_change().dropna()[-window*2:-window]
                
                symbol_returns2 = symbol_prices.pct_change().dropna()[-window:]
                dxy_returns2 = dxy_prices.pct_change().dropna()[-window:]
                
                # Calculate correlations for each window
                if self.config.correlation_method == 'pearson':
                    corr1, _ = pearsonr(dxy_returns1, symbol_returns1)
                    corr2, _ = pearsonr(dxy_returns2, symbol_returns2)
                else:
                    corr1, _ = spearmanr(dxy_returns1, symbol_returns1)
                    corr2, _ = spearmanr(dxy_returns2, symbol_returns2)
                
                # Calculate stability score (1 - absolute difference)
                stability = 1 - abs(corr1 - corr2)
                correlation_stability_scores.append(stability)
        
        # Calculate trend strength from price action
        trend_strength = await self._calculate_trend_strength(symbol_prices)
        
        # Calculate correlation strength score (absolute value of correlation)
        correlation_strength_score = abs(correlation)
        
        # Combine scores with weights
        if correlation_stability_scores:
            avg_stability = np.mean(correlation_stability_scores)
            trend_confidence = (
                correlation_strength_score * 0.4 +
                avg_stability * 0.3 +
                trend_strength * 0.3
            )
        else:
            # Fallback to basic calculation if not enough data for rolling windows
            trend_confidence = correlation_strength_score * 0.7 + trend_strength * 0.3
        
        # Ensure trend confidence is within valid range
        trend_confidence = np.clip(trend_confidence, 0.0, 1.0)
        
        logger.debug(f"{symbol} trend confidence: {trend_confidence:.3f} (strength: {correlation_strength_score:.3f}, stability: {np.mean(correlation_stability_scores):.3f}, trend: {trend_strength:.3f})")
        
        return trend_confidence
    
    async def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength from price series"""
        if len(prices) < 20:
            return 0.5  # Neutral trend
        
        # Use multiple trend indicators
        returns = prices.pct_change().dropna()
        
        # 1. Linear regression slope (trend direction and strength)
        x = np.arange(len(prices[-20:]))
        y = prices[-20:].values
        slope, _, r_value, _, _ = np.polyfit(x, y, 1)
        regression_strength = abs(r_value)
        
        # 2. Moving average convergence (trend persistence)
        short_ma = prices.rolling(window=10).mean().iloc[-1]
        long_ma = prices.rolling(window=30).mean().iloc[-1]
        ma_strength = abs(short_ma - long_ma) / prices.iloc[-1]
        
        # 3. Volatility in direction of trend
        trend_direction = 1 if slope > 0 else -1
        directional_volatility = np.std(returns[-20:][returns[-20:] * trend_direction > 0])
        
        # Combine indicators
        trend_strength = (
            regression_strength * 0.5 +
            ma_strength * 0.3 +
            directional_volatility * 0.2
        )
        
        return np.clip(trend_strength, 0.0, 1.0)
    
    async def _detect_dxy_trend(self) -> str:
        """Detect DXY trend direction with improved accuracy using multiple indicators"""
        if len(self.dxy_price_history) < 20:
            return 'neutral'
            
        prices = self.dxy_price_history.values
        
        # 1. Moving average crossover
        short_ma = np.mean(prices[-10:])
        long_ma = np.mean(prices[-30:])
        
        # 2. Price direction over different time frames
        price_change_10d = (prices[-1] - prices[-10]) / prices[-10]
        price_change_20d = (prices[-1] - prices[-20]) / prices[-20]
        price_change_30d = (prices[-1] - prices[-30]) / prices[-30]
        
        # 3. Momentum indicator (RSI)
        returns = np.diff(np.log(prices[-14:]))
        gains = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        rs = gains / losses if losses > 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        # Combine trend signals with voting system
        bullish_signals = 0
        bearish_signals = 0
        
        # MA crossover signal
        if short_ma > long_ma * (1 + self.config.trend_strength_threshold):
            bullish_signals += 2
        elif short_ma < long_ma * (1 - self.config.trend_strength_threshold):
            bearish_signals += 2
            
        # Price direction signals
        if price_change_10d > 0.005:
            bullish_signals += 1
        elif price_change_10d < -0.005:
            bearish_signals += 1
            
        if price_change_20d > 0.01:
            bullish_signals += 1
        elif price_change_20d < -0.01:
            bearish_signals += 1
            
        if price_change_30d > 0.015:
            bullish_signals += 1
        elif price_change_30d < -0.015:
            bearish_signals += 1
            
        # RSI signal
        if rsi > 60:
            bearish_signals += 1  # Overbought
        elif rsi < 40:
            bullish_signals += 1  # Oversold
            
        # Determine trend based on signal strength
        total_signals = bullish_signals + bearish_signals
        
        if total_signals >= 3:
            if bullish_signals > bearish_signals + 1:
                return 'bullish'
            elif bearish_signals > bullish_signals + 1:
                return 'bearish'
                
        return 'neutral'
    
    async def _calculate_dxy_momentum(self) -> float:
        """Calculate DXY momentum"""
        if len(self.dxy_price_history) < 10:
            return 0.0
            
        returns = self.dxy_price_history.pct_change().values[-10:]
        momentum = np.mean(returns)
        
        return momentum
    
    async def _calculate_dxy_volatility(self) -> float:
        """Calculate DXY volatility"""
        if len(self.dxy_price_history) < 20:
            return 0.0
            
        returns = self.dxy_price_history.pct_change().dropna()
        volatility = returns[-20:].std() * np.sqrt(252)
        
        return volatility
    
    async def _calculate_portfolio_metrics(self, symbols: List[str], 
                                         correlations: Dict[str, float]) -> Tuple[float, float]:
        """Calculate portfolio-level DXY correlation metrics"""
        if not symbols:
            return 0.0, 0.0
            
        # Calculate average absolute correlation
        abs_correlations = [abs(correlations[symbol]) for symbol in symbols if symbol in correlations]
        avg_correlation = np.mean(abs_correlations) if abs_correlations else 0.0
        
        # Calculate diversification score (lower correlation = better diversification)
        diversification_score = 1 - avg_correlation
        
        return avg_correlation, diversification_score
    
    def _empty_correlation_result(self, symbols: List[str], 
                                 timestamp: datetime) -> DXYCorrelationResult:
        """Create empty correlation result when no data available"""
        return DXYCorrelationResult(
            timestamp=timestamp,
            dxy_price=0.0,
            dxy_returns=pd.Series(),
            correlations={symbol: 0.0 for symbol in symbols},
            correlation_strengths={symbol: 0.0 for symbol in symbols},
            symbol_trend_confidence={symbol: 0.0 for symbol in symbols},
            dxy_trend='neutral',
            dxy_momentum=0.0,
            dxy_volatility=0.0,
            eligible_symbols=symbols,
            filtered_symbols=[],
            position_size_adjustments={symbol: 1.0 for symbol in symbols},
            portfolio_correlation=0.0,
            diversification_score=1.0
        )
    
    async def should_filter_signal(self, signal: Dict) -> Tuple[bool, str]:
        """
        Determine if a trading signal should be filtered based on DXY correlation
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            (should_filter, reason) tuple
        """
        if not self.config.enabled:
            return False, "DXY correlation filter disabled"
            
        symbol = signal['symbol']
        direction = signal['direction']
        
        # Calculate current correlations
        result = await self.calculate_correlations([symbol])
        
        # Check if symbol is eligible
        if symbol not in result.eligible_symbols:
            return True, f"{symbol} not eligible based on DXY correlation"
            
        # Check trend confirmation if enabled
        if self.config.require_dxy_trend_confirmation:
            correlation = result.correlations.get(symbol, 0)
            trend_confidence = result.symbol_trend_confidence.get(symbol, 0)
            
            if trend_confidence < 0.6:
                return True, f"{symbol} trend confidence too low ({trend_confidence:.2f})"
                
            # For USD pairs, correlation should align with expected relationship
            expected_correlation = self.config.symbol_correlation_thresholds.get(symbol, 0)
            expected_correlation_sign = 1 if expected_correlation > 0 else -1
            
            if expected_correlation_sign == 1 and direction == 'long':
                if result.dxy_trend == 'bearish':
                    return True, f"DXY trend bearish - conflicting with {symbol} long position"
            elif expected_correlation_sign == -1 and direction == 'short':
                if result.dxy_trend == 'bullish':
                    return True, f"DXY trend bullish - conflicting with {symbol} short position"
        
        return False, "Signal passes DXY correlation filter"
    
    async def adjust_position_size(self, symbol: str, base_size: float) -> float:
        """
        Adjust position size based on DXY correlation
        
        Args:
            symbol: Trading symbol
            base_size: Base position size
            
        Returns:
            Adjusted position size
        """
        if not self.config.adjust_position_size_by_correlation:
            return base_size
            
        result = await self.calculate_correlations([symbol])
        adjustment = result.position_size_adjustments.get(symbol, 1.0)
        
        adjusted_size = base_size * adjustment
        logger.debug(f"Adjusted position size for {symbol}: {base_size:.4f} -> {adjusted_size:.4f}")
        
        return adjusted_size
    
    async def check_extreme_correlation(self) -> List[Dict]:
        """Check for extreme DXY correlation events"""
        if not self.config.alert_on_extreme_correlation:
            return []
            
        extreme_events = []
        current_time = datetime.now()
        
        # Calculate current correlations
        all_symbols = list(self.symbol_price_history.keys())
        result = await self.calculate_correlations(all_symbols)
        
        for symbol, strength in result.correlation_strengths.items():
            if strength >= self.config.extreme_correlation_threshold:
                # Check if we've already alerted recently
                last_alert = self.last_alert_time.get(symbol)
                if last_alert is None or (current_time - last_alert).total_seconds() > \
                   self.config.alert_frequency_minutes * 60:
                    extreme_events.append({
                        'symbol': symbol,
                        'correlation_strength': strength,
                        'correlation': result.correlations[symbol],
                        'timestamp': current_time,
                        'dxy_trend': result.dxy_trend,
                        'dxy_momentum': result.dxy_momentum
                    })
                    self.last_alert_time[symbol] = current_time
        
        return extreme_events
