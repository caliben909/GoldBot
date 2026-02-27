"""
Dynamic Correlation Engine v2.0 - Production-Ready Implementation
Optimized for real-time forex trading with SMC strategies
Features: Fast rolling correlations, portfolio risk monitoring, correlation clustering
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy.stats import pearsonr, spearmanr
from collections import deque
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class CorrelationMethod(Enum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"


@dataclass
class CorrelationConfig:
    """Configuration for correlation engine"""
    method: str = "pearson"
    lookback_period: int = 50  # bars
    min_observations: int = 20
    update_frequency: int = 1  # bars
    max_history: int = 500  # max bars to store
    high_correlation_threshold: float = 0.7
    max_correlation_threshold: float = 0.85  # Avoid crowded trades


@dataclass
class CorrelationResult:
    """Correlation analysis result"""
    timestamp: datetime
    correlation_matrix: pd.DataFrame
    volatility: pd.Series
    portfolio_correlation: float
    diversification_score: float
    high_correlation_pairs: List[Tuple[str, str, float]]
    clusters: List[List[str]]
    risk_contributions: Dict[str, float]


class DynamicCorrelationEngine:
    """
    Production-ready dynamic correlation engine for forex trading
    
    Key Features:
    - Fast rolling correlation calculation
    - Real-time portfolio correlation monitoring
    - Automatic clustering of correlated assets
    - Risk contribution analysis
    - Memory-efficient data storage
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = CorrelationConfig(**(config or {}))
        
        # Data storage with size limits
        self.returns_data: Dict[str, deque] = {}
        self.price_data: Dict[str, deque] = {}
        self.timestamps: deque = deque(maxlen=self.config.max_history)
        
        # Cached results
        self.last_correlation_matrix: Optional[pd.DataFrame] = None
        self.last_calculation_time: Optional[datetime] = None
        self.calculation_count: int = 0
        
        logger.info(f"DynamicCorrelationEngine initialized (lookback={self.config.lookback_period})")
    
    def update_price(self, symbol: str, price: float, timestamp: Optional[datetime] = None):
        """
        Update price for a symbol (call every tick/bar)
        
        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Initialize storage for new symbol
        if symbol not in self.price_data:
            self.price_data[symbol] = deque(maxlen=self.config.max_history)
            self.returns_data[symbol] = deque(maxlen=self.config.max_history)
        
        # Store price
        self.price_data[symbol].append(price)
        
        # Calculate return if we have previous price
        if len(self.price_data[symbol]) > 1:
            prev_price = list(self.price_data[symbol])[-2]
            ret = (price - prev_price) / prev_price if prev_price != 0 else 0
            self.returns_data[symbol].append(ret)
        
        # Store timestamp
        if len(self.timestamps) == 0 or self.timestamps[-1] != timestamp:
            self.timestamps.append(timestamp)
    
    def update_prices(self, prices: Dict[str, float], timestamp: Optional[datetime] = None):
        """Batch update multiple prices"""
        for symbol, price in prices.items():
            self.update_price(symbol, price, timestamp)
    
    def get_correlation(self, symbol1: str, symbol2: str, 
                       lookback: Optional[int] = None) -> float:
        """
        Calculate correlation between two symbols
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            lookback: Optional custom lookback period
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        # Validate data availability
        if symbol1 not in self.returns_data or symbol2 not in self.returns_data:
            return 0.0
        
        returns1 = list(self.returns_data[symbol1])
        returns2 = list(self.returns_data[symbol2])
        
        # Check minimum observations
        min_obs = self.config.min_observations
        if len(returns1) < min_obs or len(returns2) < min_obs:
            return 0.0
        
        # Use specified or default lookback
        period = lookback or self.config.lookback_period
        period = min(period, len(returns1), len(returns2))
        
        # Get recent returns
        recent1 = np.array(returns1[-period:])
        recent2 = np.array(returns2[-period:])
        
        # Calculate correlation
        try:
            if self.config.method == "pearson":
                corr, _ = pearsonr(recent1, recent2)
            elif self.config.method == "spearman":
                corr, _ = spearmanr(recent1, recent2)
            else:
                corr = np.corrcoef(recent1, recent2)[0, 1]
            
            return corr if not np.isnan(corr) else 0.0
            
        except Exception as e:
            logger.debug(f"Correlation calculation failed: {e}")
            return 0.0
    
    def get_correlation_matrix(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate full correlation matrix
        
        Args:
            symbols: List of symbols (default: all tracked)
            
        Returns:
            Correlation matrix as DataFrame
        """
        if symbols is None:
            symbols = list(self.returns_data.keys())
        
        n = len(symbols)
        if n < 2:
            return pd.DataFrame()
        
        # Initialize matrix
        corr_matrix = pd.DataFrame(np.eye(n), index=symbols, columns=symbols)
        
        # Calculate correlations
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i < j:
                    corr = self.get_correlation(sym1, sym2)
                    corr_matrix.loc[sym1, sym2] = corr
                    corr_matrix.loc[sym2, sym1] = corr
        
        self.last_correlation_matrix = corr_matrix
        self.last_calculation_time = datetime.now()
        self.calculation_count += 1
        
        return corr_matrix
    
    def get_portfolio_correlation(self, positions: Dict[str, float]) -> float:
        """
        Calculate portfolio-weighted average correlation
        
        Args:
            positions: Dict of symbol -> position size (can be negative for shorts)
            
        Returns:
            Portfolio correlation score (0-1, higher = more correlated risk)
        """
        if len(positions) < 2:
            return 0.0
        
        symbols = list(positions.keys())
        corr_matrix = self.get_correlation_matrix(symbols)
        
        if corr_matrix.empty:
            return 0.0
        
        # Calculate weighted average correlation
        total_weight = sum(abs(w) for w in positions.values())
        if total_weight == 0:
            return 0.0
        
        weighted_corr = 0.0
        pair_count = 0
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i < j:
                    weight1 = abs(positions[sym1]) / total_weight
                    weight2 = abs(positions[sym2]) / total_weight
                    corr = abs(corr_matrix.loc[sym1, sym2])
                    
                    weighted_corr += corr * weight1 * weight2
                    pair_count += 1
        
        return weighted_corr / pair_count if pair_count > 0 else 0.0
    
    def get_high_correlation_pairs(self, symbols: Optional[List[str]] = None,
                                    threshold: Optional[float] = None) -> List[Tuple[str, str, float]]:
        """
        Find pairs with high correlation
        
        Args:
            symbols: Symbols to check
            threshold: Correlation threshold (default: config.high_correlation_threshold)
            
        Returns:
            List of (symbol1, symbol2, correlation) tuples
        """
        if symbols is None:
            symbols = list(self.returns_data.keys())
        
        threshold = threshold or self.config.high_correlation_threshold
        
        corr_matrix = self.get_correlation_matrix(symbols)
        if corr_matrix.empty:
            return []
        
        high_pairs = []
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i < j:
                    corr = abs(corr_matrix.loc[sym1, sym2])
                    if corr >= threshold:
                        high_pairs.append((sym1, sym2, corr))
        
        # Sort by correlation descending
        high_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return high_pairs
    
    def get_clusters(self, symbols: Optional[List[str]] = None,
                    max_clusters: int = 3) -> List[List[str]]:
        """
        Group symbols into correlation clusters
        
        Args:
            symbols: Symbols to cluster
            max_clusters: Maximum number of clusters
            
        Returns:
            List of clusters (each cluster is a list of symbols)
        """
        if symbols is None:
            symbols = list(self.returns_data.keys())
        
        if len(symbols) < 2:
            return [[s] for s in symbols]
        
        corr_matrix = self.get_correlation_matrix(symbols)
        if corr_matrix.empty:
            return [[s] for s in symbols]
        
        # Simple clustering: group by highest correlations
        clusters = []
        remaining = set(symbols)
        
        while remaining and len(clusters) < max_clusters:
            # Find pair with highest correlation
            best_pair = None
            best_corr = 0
            
            for sym1 in remaining:
                for sym2 in remaining:
                    if sym1 != sym2:
                        corr = corr_matrix.loc[sym1, sym2]
                        if abs(corr) > best_corr:
                            best_corr = abs(corr)
                            best_pair = (sym1, sym2)
            
            if best_pair and best_corr > 0.5:
                # Create cluster with best pair
                cluster = list(best_pair)
                remaining -= set(best_pair)
                
                # Add highly correlated symbols
                for sym in list(remaining):
                    avg_corr = np.mean([abs(corr_matrix.loc[sym, c]) for c in cluster])
                    if avg_corr > 0.6:
                        cluster.append(sym)
                        remaining.remove(sym)
                
                clusters.append(cluster)
            else:
                # No more high correlations, add remaining as single clusters
                for sym in remaining:
                    clusters.append([sym])
                break
        
        return clusters
    
    def get_volatility(self, symbol: str, annualize: bool = True) -> float:
        """
        Calculate realized volatility for a symbol
        
        Args:
            symbol: Trading symbol
            annualize: Whether to annualize (multiply by sqrt(252))
            
        Returns:
            Volatility as standard deviation of returns
        """
        if symbol not in self.returns_data:
            return 0.0
        
        returns = list(self.returns_data[symbol])
        if len(returns) < self.config.min_observations:
            return 0.0
        
        vol = np.std(returns[-self.config.lookback_period:])
        
        if annualize:
            # Assuming daily data, annualize with sqrt(252)
            # For intraday, adjust accordingly
            vol *= np.sqrt(252)
        
        return vol
    
    def get_risk_contribution(self, symbol: str, 
                             positions: Dict[str, float]) -> float:
        """
        Calculate risk contribution of a symbol to portfolio
        
        Args:
            symbol: Symbol to analyze
            positions: Portfolio positions (symbol -> size)
            
        Returns:
            Risk contribution as percentage of portfolio risk
        """
        if symbol not in positions or len(positions) < 2:
            return 0.0
        
        symbols = list(positions.keys())
        corr_matrix = self.get_correlation_matrix(symbols)
        
        if corr_matrix.empty:
            return 0.0
        
        # Calculate portfolio variance
        weights = np.array([abs(positions[s]) for s in symbols])
        weights = weights / weights.sum()
        
        cov_matrix = corr_matrix.values  # Simplified: using correlation as covariance proxy
        
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        
        if port_variance == 0:
            return 0.0
        
        # Marginal contribution
        marginal = np.dot(cov_matrix, weights)
        
        # Risk contribution
        sym_idx = symbols.index(symbol)
        risk_contrib = (weights[sym_idx] * marginal[sym_idx]) / port_variance
        
        return risk_contrib
    
    def analyze_portfolio(self, positions: Dict[str, float]) -> Dict[str, Any]:
        """
        Complete portfolio correlation analysis
        
        Args:
            positions: Dict of symbol -> position size
            
        Returns:
            Analysis dictionary with risk metrics
        """
        if not positions:
            return {
                'portfolio_correlation': 0.0,
                'diversification_score': 1.0,
                'risk_level': 'low',
                'high_correlation_pairs': [],
                'clusters': [],
                'recommendations': []
            }
        
        symbols = list(positions.keys())
        
        # Calculate metrics
        port_corr = self.get_portfolio_correlation(positions)
        high_pairs = self.get_high_correlation_pairs(symbols)
        clusters = self.get_clusters(symbols)
        
        # Diversification score (1 - portfolio correlation)
        diversification = 1 - port_corr
        
        # Risk level
        if port_corr > 0.7:
            risk_level = 'high'
        elif port_corr > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        # Generate recommendations
        recommendations = []
        
        if port_corr > 0.6:
            recommendations.append({
                'type': 'reduce_correlation',
                'message': f'High portfolio correlation ({port_corr:.2f})',
                'action': 'Reduce position sizes or remove highly correlated pairs'
            })
        
        for sym1, sym2, corr in high_pairs[:3]:
            recommendations.append({
                'type': 'high_correlation',
                'symbols': [sym1, sym2],
                'correlation': corr,
                'action': f'Consider reducing exposure to {sym1} or {sym2}'
            })
        
        if diversification < 0.4:
            recommendations.append({
                'type': 'diversify',
                'message': f'Low diversification score ({diversification:.2f})',
                'action': 'Add uncorrelated assets to portfolio'
            })
        
        return {
            'portfolio_correlation': port_corr,
            'diversification_score': diversification,
            'risk_level': risk_level,
            'high_correlation_pairs': high_pairs[:5],
            'clusters': clusters,
            'recommendations': recommendations,
            'volatility': {sym: self.get_volatility(sym) for sym in symbols},
            'risk_contributions': {sym: self.get_risk_contribution(sym, positions) for sym in symbols}
        }
    
    def should_reduce_position(self, symbol: str, positions: Dict[str, float],
                               threshold: float = 0.75) -> Tuple[bool, str]:
        """
        Check if position should be reduced due to correlation risk
        
        Args:
            symbol: Symbol to check
            positions: Current positions
            threshold: Correlation threshold
            
        Returns:
            (should_reduce, reason) tuple
        """
        if symbol not in positions:
            return False, "No position"
        
        # Check correlation with existing positions
        for other_sym, size in positions.items():
            if other_sym != symbol and size != 0:
                corr = abs(self.get_correlation(symbol, other_sym))
                
                if corr > threshold:
                    return True, f"High correlation with {other_sym}: {corr:.2f}"
        
        # Check portfolio correlation
        port_corr = self.get_portfolio_correlation(positions)
        if port_corr > 0.8:
            return True, f"Portfolio correlation too high: {port_corr:.2f}"
        
        return False, "Correlation risk acceptable"
    
    def get_optimal_hedge_ratio(self, symbol1: str, symbol2: str) -> float:
        """
        Calculate optimal hedge ratio between two symbols
        
        Args:
            symbol1: Primary symbol
            symbol2: Hedge symbol
            
        Returns:
            Hedge ratio (position in symbol2 per unit of symbol1)
        """
        if symbol1 not in self.returns_data or symbol2 not in self.returns_data:
            return 0.0
        
        # Get returns
        returns1 = np.array(list(self.returns_data[symbol1])[-self.config.lookback_period:])
        returns2 = np.array(list(self.returns_data[symbol2])[-self.config.lookback_period:])
        
        if len(returns1) < self.config.min_observations or len(returns2) < self.config.min_observations:
            return 0.0
        
        # Beta = Cov(r1, r2) / Var(r2)
        try:
            covariance = np.cov(returns1, returns2)[0, 1]
            variance2 = np.var(returns2)
            
            if variance2 == 0:
                return 0.0
            
            beta = covariance / variance2
            return beta
            
        except Exception as e:
            logger.debug(f"Hedge ratio calculation failed: {e}")
            return 0.0
    
    def get_state(self) -> Dict[str, Any]:
        """Get current engine state"""
        return {
            'tracked_symbols': list(self.returns_data.keys()),
            'data_points': {sym: len(self.returns_data[sym]) for sym in self.returns_data},
            'last_calculation': self.last_calculation_time,
            'calculation_count': self.calculation_count,
            'config': {
                'method': self.config.method,
                'lookback': self.config.lookback_period
            }
        }
    
    def reset(self):
        """Clear all data"""
        self.returns_data.clear()
        self.price_data.clear()
        self.timestamps.clear()
        self.last_correlation_matrix = None
        self.last_calculation_time = None
        self.calculation_count = 0
        logger.info("DynamicCorrelationEngine reset")


# ============================================================================
# INTEGRATION WITH RISK ENGINE
# ============================================================================

class CorrelationRiskManager:
    """
    Risk manager with integrated correlation monitoring
    Simplified for production use
    """
    
    def __init__(self, config: dict):
        self.correlation_engine = DynamicCorrelationEngine(
            config.get('correlation', {})
        )
        self.max_portfolio_correlation = config.get('max_portfolio_correlation', 0.7)
        self.max_single_correlation = config.get('max_single_correlation', 0.8)
        self.max_positions_per_cluster = config.get('max_positions_per_cluster', 2)
        
        self.positions: Dict[str, float] = {}
        self.clusters: List[List[str]] = []
        
        logger.info("CorrelationRiskManager initialized")
    
    def update_price(self, symbol: str, price: float):
        """Update price data"""
        self.correlation_engine.update_price(symbol, price)
    
    def add_position(self, symbol: str, size: float) -> Tuple[bool, str]:
        """
        Add new position with correlation check
        
        Args:
            symbol: Trading symbol
            size: Position size (positive=long, negative=short)
            
        Returns:
            (allowed, reason) tuple
        """
        # Test position
        test_positions = self.positions.copy()
        test_positions[symbol] = size
        
        # Check correlation with existing positions
        for existing_sym, existing_size in self.positions.items():
            if existing_size != 0:
                corr = self.correlation_engine.get_correlation(symbol, existing_sym)
                
                if abs(corr) > self.max_single_correlation:
                    return False, f"Correlation with {existing_sym} too high: {corr:.2f}"
        
        # Check portfolio correlation
        port_corr = self.correlation_engine.get_portfolio_correlation(test_positions)
        if port_corr > self.max_portfolio_correlation:
            return False, f"Portfolio correlation would be too high: {port_corr:.2f}"
        
        # Check cluster limits
        clusters = self.correlation_engine.get_clusters(list(test_positions.keys()))
        for cluster in clusters:
            cluster_positions = [s for s in cluster if s in test_positions and test_positions[s] != 0]
            if len(cluster_positions) > self.max_positions_per_cluster:
                return False, f"Too many positions in cluster: {cluster_positions}"
        
        # Accept position
        self.positions[symbol] = size
        self.clusters = clusters
        
        return True, "Position accepted"
    
    def remove_position(self, symbol: str):
        """Remove position from tracking"""
        if symbol in self.positions:
            del self.positions[symbol]
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Get comprehensive risk report"""
        return self.correlation_engine.analyze_portfolio(self.positions)
    
    def get_hedge_recommendation(self, symbol: str) -> Optional[str]:
        """
        Suggest best hedge for a position
        
        Args:
            symbol: Symbol to hedge
            
        Returns:
            Recommended hedge symbol or None
        """
        if symbol not in self.positions:
            return None
        
        all_symbols = list(self.correlation_engine.returns_data.keys())
        best_hedge = None
        best_corr = 1.0  # Looking for lowest correlation
        
        for other in all_symbols:
            if other != symbol and other not in self.positions:
                corr = abs(self.correlation_engine.get_correlation(symbol, other))
                if corr < best_corr:
                    best_corr = corr
                    best_hedge = other
        
        return best_hedge if best_corr < 0.3 else None


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize
    config = {
        'method': 'pearson',
        'lookback_period': 30,
        'high_correlation_threshold': 0.7
    }
    
    engine = DynamicCorrelationEngine(config)
    
    # Simulate market data for forex pairs
    import random
    
    np.random.seed(42)
    
    # Generate correlated returns
    n_days = 100
    
    # DXY trend (upward)
    dxy_returns = np.random.normal(0.0005, 0.005, n_days)
    
    # EURUSD (negatively correlated with DXY)
    eur_returns = -0.7 * dxy_returns + np.random.normal(0, 0.003, n_days)
    
    # GBPUSD (negatively correlated with DXY, correlated with EURUSD)
    gbp_returns = -0.6 * dxy_returns + 0.5 * eur_returns + np.random.normal(0, 0.004, n_days)
    
    # USDJPY (positively correlated with DXY)
    jpy_returns = 0.6 * dxy_returns + np.random.normal(0, 0.004, n_days)
    
    # XAUUSD (negatively correlated with DXY)
    gold_returns = -0.5 * dxy_returns + np.random.normal(0, 0.008, n_days)
    
    # Update engine with prices
    base_prices = {'DXY': 100.0, 'EURUSD': 1.10, 'GBPUSD': 1.25, 
                   'USDJPY': 150.0, 'XAUUSD': 2000.0}
    
    for i in range(n_days):
        timestamp = datetime.now() - timedelta(days=n_days-i)
        
        # Update DXY
        base_prices['DXY'] *= (1 + dxy_returns[i])
        engine.update_price('DXY', base_prices['DXY'], timestamp)
        
        # Update forex pairs
        base_prices['EURUSD'] *= (1 + eur_returns[i])
        engine.update_price('EURUSD', base_prices['EURUSD'], timestamp)
        
        base_prices['GBPUSD'] *= (1 + gbp_returns[i])
        engine.update_price('GBPUSD', base_prices['GBPUSD'], timestamp)
        
        base_prices['USDJPY'] *= (1 + jpy_returns[i])
        engine.update_price('USDJPY', base_prices['USDJPY'], timestamp)
        
        base_prices['XAUUSD'] *= (1 + gold_returns[i])
        engine.update_price('XAUUSD', base_prices['XAUUSD'], timestamp)
    
    # Analyze correlations
    print("Correlation Matrix:")
    print("=" * 60)
    corr_matrix = engine.get_correlation_matrix()
    print(corr_matrix.round(3))
    
    print("\nHigh Correlation Pairs:")
    print("=" * 60)
    high_pairs = engine.get_high_correlation_pairs()
    for sym1, sym2, corr in high_pairs:
        print(f"{sym1} - {sym2}: {corr:.3f}")
    
    print("\nCorrelation Clusters:")
    print("=" * 60)
    clusters = engine.get_clusters()
    for i, cluster in enumerate(clusters, 1):
        print(f"Cluster {i}: {', '.join(cluster)}")
    
    # Portfolio analysis
    print("\nPortfolio Analysis:")
    print("=" * 60)
    positions = {'EURUSD': 1.0, 'GBPUSD': 0.5, 'USDJPY': -0.5}
    analysis = engine.analyze_portfolio(positions)
    
    print(f"Portfolio Correlation: {analysis['portfolio_correlation']:.3f}")
    print(f"Diversification Score: {analysis['diversification_score']:.3f}")
    print(f"Risk Level: {analysis['risk_level']}")
    
    print("\nRecommendations:")
    for rec in analysis['recommendations']:
        print(f"- {rec['type']}: {rec.get('message', rec.get('action', ''))}")
    
    # Test risk manager
    print("\nCorrelation Risk Manager:")
    print("=" * 60)
    risk_config = {
        'max_portfolio_correlation': 0.6,
        'max_single_correlation': 0.75
    }
    
    risk_mgr = CorrelationRiskManager(risk_config)
    
    # Feed data to risk manager
    for sym in ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']:
        for price in list(engine.price_data[sym])[-30:]:
            risk_mgr.update_price(sym, price)
    
    # Test position adding
    allowed, reason = risk_mgr.add_position('EURUSD', 1.0)
    print(f"Add EURUSD: {'ALLOWED' if allowed else 'REJECTED'} - {reason}")
    
    allowed, reason = risk_mgr.add_position('GBPUSD', 0.5)
    print(f"Add GBPUSD: {'ALLOWED' if allowed else 'REJECTED'} - {reason}")
    
    allowed, reason = risk_mgr.add_position('XAUUSD', 0.5)
    print(f"Add XAUUSD: {'ALLOWED' if allowed else 'REJECTED'} - {reason}")