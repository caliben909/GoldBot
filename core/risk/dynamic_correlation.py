"""
Dynamic Correlation Analysis - Calculate time-varying correlation matrices
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.covariance import LedoitWolf, OAS
from statsmodels.tsa.api import VAR
from scipy.linalg import cholesky

logger = logging.getLogger(__name__)


@dataclass
class CorrelationConfig:
    """Configuration for dynamic correlation calculation"""
    # Rolling window settings
    window_type: str = 'expanding'  # 'rolling' or 'expanding'
    window_length: int = 60  # days for expanding, periods for rolling
    min_observations: int = 20
    
    # Correlation method
    method: str = 'pearson'  # 'pearson', 'spearman', 'kendall'
    
    # Covariance estimator
    covariance_estimator: str = 'ledoit_wolf'  # 'sample', 'ledoit_wolf', 'oas', 'shrunk'
    
    # Volatility smoothing
    use_garch: bool = False
    garch_order: Tuple[int, int] = (1, 1)
    
    # Outlier handling
    remove_outliers: bool = False
    outlier_threshold: float = 3.0
    
    # Update frequency
    update_frequency: int = 1  # days
    
    # Risk management thresholds
    max_correlation_threshold: float = 0.7
    high_correlation_threshold: float = 0.5
    medium_correlation_threshold: float = 0.3
    
    # Portfolio optimization settings
    max_portfolio_correlation: float = 0.3
    max_single_correlation: float = 0.7


@dataclass
class CorrelationResult:
    """Result of correlation calculation"""
    # Time-varying correlation matrix
    correlation_matrix: pd.DataFrame
    volatility: pd.Series
    
    # Portfolio-level metrics
    portfolio_volatility: float
    portfolio_beta: float
    diversification_score: float
    
    # Risk contributions
    risk_contributions: Dict[str, float]
    
    # Correlation clusters
    clusters: List[List[str]]
    
    # High correlation pairs
    high_correlation_pairs: List[Tuple[str, str, float]]
    
    # Time series of correlation metrics
    average_correlation: pd.Series
    correlation_std: pd.Series
    max_correlation: pd.Series
    min_correlation: pd.Series
    
    # GARCH volatility forecasts (if enabled)
    conditional_volatility: Optional[pd.Series] = None
    volatility_forecast: Optional[pd.Series] = None
    forecast_error: Optional[pd.Series] = None


class DynamicCorrelationEngine:
    """
    Dynamic correlation engine with time-varying correlation analysis
    
    Features:
    - Multiple correlation methods (Pearson, Spearman, Kendall)
    - Robust covariance estimation
    - GARCH volatility modeling
    - Outlier detection and handling
    - Correlation clustering
    - Risk contribution analysis
    """
    
    def __init__(self, config: dict):
        self.config = CorrelationConfig(**config['risk_management']['dynamic_correlation'])
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.price_history: Dict[str, pd.Series] = {}
        self.returns_history: Dict[str, pd.Series] = {}
        self.correlation_history: Dict[datetime, pd.DataFrame] = {}
        self.volatility_history: Dict[datetime, pd.Series] = {}
        
        # GARCH models
        self.garch_models: Dict[str, Any] = {}
        
        logger.info("DynamicCorrelationEngine initialized")
    
    async def update_data(self, symbol: str, prices: pd.Series):
        """Update price history for a symbol"""
        # Store prices
        self.price_history[symbol] = prices
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        self.returns_history[symbol] = returns
        
        # Fit GARCH model if configured
        if self.config.use_garch:
            await self._fit_garch_model(symbol, returns)
    
    async def _fit_garch_model(self, symbol: str, returns: pd.Series):
        """Fit GARCH model to returns series"""
        try:
            from arch import arch_model
            
            # Create GARCH model
            model = arch_model(
                returns * 100,  # Convert to percentage
                vol='Garch',
                p=self.config.garch_order[0],
                q=self.config.garch_order[1],
                mean='Constant',
                dist='Normal'
            )
            
            # Fit model
            fit = model.fit(disp='off')
            
            # Store model
            self.garch_models[symbol] = fit
            
            self.logger.debug(f"GARCH model fitted for {symbol}")
            
        except Exception as e:
            self.logger.warning(f"Failed to fit GARCH model for {symbol}: {e}")
            return None
    
    async def calculate_correlation_matrix(self, symbols: List[str], 
                                         as_of: Optional[datetime] = None) -> CorrelationResult:
        """
        Calculate dynamic correlation matrix
        
        Args:
            symbols: List of symbols to include
            as_of: Date to calculate correlation for
        
        Returns:
            CorrelationResult object
        """
        if as_of is None:
            as_of = datetime.now()
        
        # Get returns data for symbols
        returns_data = []
        valid_symbols = []
        
        for symbol in symbols:
            if symbol in self.returns_history:
                returns = self.returns_history[symbol]
                
                # Filter data up to as_of
                if returns.index.tz is not None:
                    as_of = as_of.astimezone(returns.index.tz)
                
                filtered_returns = returns[returns.index <= as_of]
                
                if len(filtered_returns) >= self.config.min_observations:
                    returns_data.append(filtered_returns)
                    valid_symbols.append(symbol)
        
        if len(valid_symbols) < 2:
            self.logger.warning(f"Not enough valid symbols for correlation calculation: {len(valid_symbols)}")
            return self._empty_correlation_result(symbols)
        
        # Remove outliers if configured
        if self.config.remove_outliers:
            returns_data = [self._remove_outliers(r) for r in returns_data]
        
        # Calculate rolling window
        if self.config.window_type == 'expanding':
            window_size = None  # Use all available data
        else:
            window_size = self.config.window_length
        
        # Calculate correlations
        if self.config.method == 'pearson':
            corr_matrix = self._calculate_pearson_correlation(returns_data, valid_symbols, window_size)
        elif self.config.method == 'spearman':
            corr_matrix = self._calculate_spearman_correlation(returns_data, valid_symbols, window_size)
        elif self.config.method == 'kendall':
            corr_matrix = self._calculate_kendall_correlation(returns_data, valid_symbols, window_size)
        else:
            raise ValueError(f"Unsupported correlation method: {self.config.method}")
        
        # Calculate volatility
        volatility = self._calculate_volatility(returns_data, valid_symbols, window_size)
        
        # Calculate conditional volatility from GARCH
        conditional_volatility = None
        if self.config.use_garch:
            conditional_volatility = await self._calculate_conditional_volatility(valid_symbols)
        
        # Calculate risk contributions
        risk_contributions = await self._calculate_risk_contributions(valid_symbols, corr_matrix)
        
        # Perform clustering
        clusters = await self._cluster_assets(valid_symbols, corr_matrix)
        
        # Find high correlation pairs
        high_correlation_pairs = await self._find_high_correlation_pairs(valid_symbols, corr_matrix)
        
        # Calculate time series metrics
        avg_corr, corr_std, max_corr, min_corr = await self._calculate_correlation_metrics(valid_symbols)
        
        # Calculate portfolio-level metrics
        portfolio_volatility, portfolio_beta, diversification_score = await self._calculate_portfolio_metrics(
            valid_symbols, corr_matrix, volatility
        )
        
        # Forecast volatility if GARCH is enabled
        volatility_forecast = None
        forecast_error = None
        if self.config.use_garch:
            volatility_forecast, forecast_error = await self._forecast_volatility(valid_symbols)
        
        return CorrelationResult(
            correlation_matrix=corr_matrix,
            volatility=volatility,
            conditional_volatility=conditional_volatility,
            portfolio_volatility=portfolio_volatility,
            portfolio_beta=portfolio_beta,
            diversification_score=diversification_score,
            risk_contributions=risk_contributions,
            clusters=clusters,
            high_correlation_pairs=high_correlation_pairs,
            average_correlation=avg_corr,
            correlation_std=corr_std,
            max_correlation=max_corr,
            min_correlation=min_corr,
            volatility_forecast=volatility_forecast,
            forecast_error=forecast_error
        )
    
    def _calculate_pearson_correlation(self, returns_data: List[pd.Series], 
                                     symbols: List[str], window_size: Optional[int]) -> pd.DataFrame:
        """Calculate Pearson correlation matrix"""
        returns_df = pd.concat(returns_data, axis=1, keys=symbols).dropna()
        
        if window_size is not None:
            corr_matrix = returns_df.rolling(window=window_size).corr()
            corr_matrix = corr_matrix.groupby(level=0).last()
        else:
            corr_matrix = returns_df.corr()
        
        return corr_matrix
    
    def _calculate_spearman_correlation(self, returns_data: List[pd.Series], 
                                       symbols: List[str], window_size: Optional[int]) -> pd.DataFrame:
        """Calculate Spearman rank correlation matrix"""
        returns_df = pd.concat(returns_data, axis=1, keys=symbols).dropna()
        
        def spearman_corr(x, y):
            return spearmanr(x, y)[0]
        
        if window_size is not None:
            corr_matrix = returns_df.rolling(window=window_size).corr(spearman_corr)
            corr_matrix = corr_matrix.groupby(level=0).last()
        else:
            corr_matrix = returns_df.corr(method='spearman')
        
        return corr_matrix
    
    def _calculate_kendall_correlation(self, returns_data: List[pd.Series], 
                                     symbols: List[str], window_size: Optional[int]) -> pd.DataFrame:
        """Calculate Kendall's tau correlation matrix"""
        returns_df = pd.concat(returns_data, axis=1, keys=symbols).dropna()
        
        if window_size is not None:
            def kendall_corr(x, y):
                return kendalltau(x, y)[0]
            
            corr_matrix = returns_df.rolling(window=window_size).corr(kendall_corr)
            corr_matrix = corr_matrix.groupby(level=0).last()
        else:
            corr_matrix = returns_df.corr(method='kendall')
        
        return corr_matrix
    
    def _calculate_volatility(self, returns_data: List[pd.Series], 
                             symbols: List[str], window_size: Optional[int]) -> pd.Series:
        """Calculate annualized volatility"""
        returns_df = pd.concat(returns_data, axis=1, keys=symbols).dropna()
        
        if window_size is not None:
            volatility = returns_df.rolling(window=window_size).std() * np.sqrt(252)
        else:
            volatility = returns_df.std() * np.sqrt(252)
        
        return volatility.iloc[-1] if hasattr(volatility, 'iloc') else volatility
    
    async def _calculate_conditional_volatility(self, symbols: List[str]) -> pd.Series:
        """Calculate conditional volatility from GARCH models"""
        conditional_vol = {}
        
        for symbol in symbols:
            if symbol in self.garch_models:
                try:
                    # Get last conditional volatility from GARCH
                    last_vol = self.garch_models[symbol].conditional_volatility.iloc[-1]
                    conditional_vol[symbol] = last_vol
                except Exception as e:
                    self.logger.warning(f"Failed to get conditional volatility for {symbol}: {e}")
        
        return pd.Series(conditional_vol)
    
    async def _calculate_risk_contributions(self, symbols: List[str], 
                                           corr_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk contributions using marginal contributions to VaR"""
        # Equal weight portfolio
        weights = np.array([1 / len(symbols)] * len(symbols))
        
        # Calculate portfolio variance
        variance = np.dot(weights.T, np.dot(corr_matrix.values, weights))
        
        # Calculate marginal contributions
        marginal_contributions = np.dot(corr_matrix.values, weights)
        
        # Calculate risk contributions
        risk_contributions = (weights * marginal_contributions) / variance
        
        return dict(zip(symbols, risk_contributions))
    
    async def _cluster_assets(self, symbols: List[str], 
                            corr_matrix: pd.DataFrame) -> List[List[str]]:
        """Cluster assets based on correlation matrix"""
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        # Convert correlation matrix to distance matrix
        distance_matrix = np.sqrt(0.5 * (1 - corr_matrix.values))
        distance_matrix = squareform(distance_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distance_matrix, method='ward')
        
        # Determine clusters (0.3 correlation threshold)
        clusters = fcluster(linkage_matrix, 0.3, criterion='distance')
        
        # Group symbols by cluster
        cluster_dict = {}
        for i, (symbol, cluster_id) in enumerate(zip(symbols, clusters)):
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []
            cluster_dict[cluster_id].append(symbol)
        
        return list(cluster_dict.values())
    
    async def _find_high_correlation_pairs(self, symbols: List[str], 
                                         corr_matrix: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Find pairs with high correlation"""
        high_corr_pairs = []
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i < j:
                    corr = abs(corr_matrix.loc[symbol1, symbol2])
                    
                    if corr >= self.config.max_correlation_threshold:
                        high_corr_pairs.append((symbol1, symbol2, corr))
        
        # Sort by correlation descending
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return high_corr_pairs
    
    async def _calculate_correlation_metrics(self, symbols: List[str]) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Calculate time series of correlation metrics"""
        avg_corr = []
        corr_std = []
        max_corr = []
        min_corr = []
        dates = []
        
        # Get common dates across all symbols
        common_dates = []
        for symbol in symbols:
            if symbol in self.returns_history:
                dates.extend(self.returns_history[symbol].index.tolist())
        
        common_dates = sorted(list(set(common_dates)))
        
        # Calculate metrics for each date
        for date in common_dates:
            try:
                # Get correlation matrix for this date
                result = await self.calculate_correlation_matrix(symbols, as_of=date)
                corr_matrix = result.correlation_matrix
                
                # Calculate metrics
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), 1).astype(bool)).stack()
                
                avg_corr.append(upper_tri.mean())
                corr_std.append(upper_tri.std())
                max_corr.append(upper_tri.max())
                min_corr.append(upper_tri.min())
                dates.append(date)
            except Exception as e:
                continue
        
        return (
            pd.Series(avg_corr, index=dates),
            pd.Series(corr_std, index=dates),
            pd.Series(max_corr, index=dates),
            pd.Series(min_corr, index=dates)
        )
    
    async def _calculate_portfolio_metrics(self, symbols: List[str], 
                                           corr_matrix: pd.DataFrame,
                                           volatility: pd.Series) -> Tuple[float, float, float]:
        """Calculate portfolio-level metrics"""
        # Equal weight portfolio
        n = len(symbols)
        weights = np.array([1/n]*n)
        
        # Calculate portfolio volatility
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(corr_matrix.values * np.outer(volatility, volatility), weights)))
        
        # Calculate portfolio beta (assuming market is average of all assets)
        market_returns = np.mean([self.returns_history[symbol].values for symbol in symbols], axis=0)
        portfolio_returns = np.sum([self.returns_history[symbol].values * w for symbol, w in zip(symbols, weights)], axis=0)
        
        beta = np.cov(portfolio_returns, market_returns)[0, 1] / np.var(market_returns)
        
        # Calculate diversification score (Herfindahl-Hirschman Index)
        diversification_score = 1 - np.sum(weights ** 2)
        
        return portfolio_vol, beta, diversification_score
    
    async def _forecast_volatility(self, symbols: List[str]) -> Tuple[pd.Series, pd.Series]:
        """Forecast volatility using GARCH models"""
        forecasts = {}
        errors = {}
        
        for symbol in symbols:
            if symbol in self.garch_models:
                try:
                    # Forecast 1-step ahead volatility
                    forecast = self.garch_models[symbol].forecast(horizon=1)
                    forecasts[symbol] = forecast.variance.iloc[-1].iloc[-1] ** 0.5
                    
                    # Calculate forecast error
                    actual_vol = self.returns_history[symbol].iloc[-1:].std() * np.sqrt(252)
                    errors[symbol] = abs(forecasts[symbol] - actual_vol)
                
                except Exception as e:
                    self.logger.warning(f"Failed to forecast volatility for {symbol}: {e}")
        
        return pd.Series(forecasts), pd.Series(errors)
    
    def _remove_outliers(self, returns: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Remove outliers using Z-score method"""
        z_scores = np.abs(stats.zscore(returns.dropna()))
        return returns[z_scores < threshold]
    
    def _empty_correlation_result(self, symbols: List[str]) -> CorrelationResult:
        """Create empty correlation result when no data available"""
        # Create empty correlation matrix
        n = len(symbols)
        corr_matrix = pd.DataFrame(np.eye(n), index=symbols, columns=symbols)
        
        return CorrelationResult(
            correlation_matrix=corr_matrix,
            volatility=pd.Series([0.0]*n, index=symbols),
            conditional_volatility=None,
            portfolio_volatility=0.0,
            portfolio_beta=0.0,
            diversification_score=0.0,
            risk_contributions=dict(zip(symbols, [1/n]*n)),
            clusters=[[symbol] for symbol in symbols],
            high_correlation_pairs=[],
            average_correlation=pd.Series([0.0], index=[datetime.now()]),
            correlation_std=pd.Series([0.0], index=[datetime.now()]),
            max_correlation=pd.Series([0.0], index=[datetime.now()]),
            min_correlation=pd.Series([0.0], index=[datetime.now()]),
            volatility_forecast=None,
            forecast_error=None
        )
    
    async def detect_correlation_risk(self, current_positions: List[Dict], 
                                     correlation_result: CorrelationResult) -> List[Dict]:
        """
        Detect correlation-based risks
        
        Args:
            current_positions: List of current positions
            correlation_result: Correlation analysis result
        
        Returns:
            List of risk alerts
        """
        risks = []
        
        # Extract symbols from positions
        position_symbols = [pos['symbol'] for pos in current_positions]
        
        # Check portfolio correlation
        if correlation_result.portfolio_volatility > self.config.max_portfolio_correlation:
            risks.append({
                'type': 'portfolio_correlation',
                'level': 'high',
                'message': f"Portfolio correlation ({correlation_result.portfolio_volatility:.2f}) exceeds threshold ({self.config.max_portfolio_correlation:.2f})",
                'suggestion': 'Consider reducing exposure to highly correlated assets'
            })
        
        # Check single position correlation
        for pos in current_positions:
            symbol = pos['symbol']
            
            if symbol in correlation_result.risk_contributions:
                if correlation_result.risk_contributions[symbol] > self.config.max_single_correlation:
                    risks.append({
                        'type': 'single_position_correlation',
                        'level': 'medium',
                        'message': f"{symbol} risk contribution ({correlation_result.risk_contributions[symbol]:.2f}) exceeds threshold ({self.config.max_single_correlation:.2f})",
                        'suggestion': f"Consider reducing position size for {symbol}"
                    })
        
        # Check high correlation pairs
        for symbol1, symbol2, corr in correlation_result.high_correlation_pairs:
            if symbol1 in position_symbols and symbol2 in position_symbols:
                risks.append({
                    'type': 'high_correlation_pair',
                    'level': 'high',
                    'message': f"High correlation between {symbol1} and {symbol2}: {corr:.2f}",
                    'suggestion': 'Consider reducing exposure to one of these assets'
                })
        
        # Check diversification score
        if correlation_result.diversification_score < 0.3:
            risks.append({
                'type': 'diversification_risk',
                'level': 'medium',
                'message': f"Low diversification score ({correlation_result.diversification_score:.2f})",
                'suggestion': 'Consider adding uncorrelated assets to the portfolio'
            })
        
        return risks
    
    async def optimize_correlation_risk(self, current_positions: List[Dict], 
                                      correlation_result: CorrelationResult) -> List[Dict]:
        """
        Suggest correlation risk reduction optimizations
        
        Args:
            current_positions: List of current positions
            correlation_result: Correlation analysis result
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Calculate total portfolio value
        total_value = sum(pos['quantity'] * pos['avg_entry_price'] for pos in current_positions)
        
        # Identify highly correlated assets
        for symbol1, symbol2, corr in correlation_result.high_correlation_pairs:
            pos1 = next((p for p in current_positions if p['symbol'] == symbol1), None)
            pos2 = next((p for p in current_positions if p['symbol'] == symbol2), None)
            
            if pos1 and pos2:
                # Calculate combined exposure
                exposure1 = pos1['quantity'] * pos1['avg_entry_price'] / total_value
                exposure2 = pos2['quantity'] * pos2['avg_entry_price'] / total_value
                
                if exposure1 + exposure2 > 0.2:  # 20% combined limit
                    suggestions.append({
                        'type': 'correlation_reduction',
                        'symbols': [symbol1, symbol2],
                        'correlation': corr,
                        'current_exposure': exposure1 + exposure2,
                        'suggested_exposure': min(exposure1 + exposure2, 0.2),
                        'action': 'reduce',
                        'rationale': f"Combined exposure to highly correlated assets exceeds 20%"
                    })
        
        # Suggest diversifying assets
        if correlation_result.diversification_score < 0.3:
            # Identify potential diversifiers from available universe
            available_symbols = list(self.returns_history.keys())
            position_symbols = [pos['symbol'] for pos in current_positions]
            potential_symbols = [s for s in available_symbols if s not in position_symbols]
            
            # Calculate correlation to current portfolio
            if potential_symbols:
                portfolio_correlations = {}
                
                for symbol in potential_symbols:
                    if symbol in self.returns_history:
                        # Calculate average correlation to current positions
                        avg_corr = 0.0
                        count = 0
                        
                        for pos_symbol in position_symbols:
                            if pos_symbol in self.returns_history:
                                returns1 = self.returns_history[symbol].dropna()
                                returns2 = self.returns_history[pos_symbol].dropna()
                                
                                if len(returns1) > 0 and len(returns2) > 0:
                                    overlap = returns1.index.intersection(returns2.index)
                                    if len(overlap) > 10:
                                        returns1 = returns1.loc[overlap]
                                        returns2 = returns2.loc[overlap]
                                        avg_corr += abs(pearsonr(returns1, returns2)[0])
                                        count += 1
                        
                        if count > 0:
                            avg_corr /= count
                            portfolio_correlations[symbol] = avg_corr
                
                # Suggest symbols with lowest correlation
                if portfolio_correlations:
                    sorted_symbols = sorted(portfolio_correlations.items(), key=lambda x: x[1])
                    
                    for symbol, avg_corr in sorted_symbols[:3]:
                        suggestions.append({
                            'type': 'diversification_opportunity',
                            'symbol': symbol,
                            'correlation_to_portfolio': avg_corr,
                            'suggested_exposure': 0.05,  # 5% of portfolio
                            'rationale': f"Low correlation to current portfolio ({avg_corr:.2f})"
                        })
        
        return suggestions


# Helper functions for quick correlation calculations
def quick_correlation(returns1: pd.Series, returns2: pd.Series, 
                     method: str = 'pearson') -> float:
    """Quick correlation calculation between two return series"""
    if len(returns1) < 10 or len(returns2) < 10:
        return 0.0
    
    # Get overlapping dates
    overlap = returns1.index.intersection(returns2.index)
    
    if len(overlap) < 10:
        return 0.0
    
    returns1 = returns1.loc[overlap]
    returns2 = returns2.loc[overlap]
    
    if method == 'pearson':
        return pearsonr(returns1, returns2)[0]
    elif method == 'spearman':
        return spearmanr(returns1, returns2)[0]
    elif method == 'kendall':
        return kendalltau(returns1, returns2)[0]
    
    return 0.0


def rolling_correlation(returns1: pd.Series, returns2: pd.Series, 
                      window: int = 60, method: str = 'pearson') -> pd.Series:
    """Calculate rolling correlation between two return series"""
    returns_df = pd.concat([returns1, returns2], axis=1, keys=['returns1', 'returns2']).dropna()
    
    if method == 'pearson':
        return returns_df.rolling(window=window).corr().unstack().loc[:, 'returns1', 'returns2']
    elif method == 'spearman':
        def corr_func(x):
            return spearmanr(x['returns1'], x['returns2'])[0]
        
        return returns_df.rolling(window=window).apply(corr_func)
    elif method == 'kendall':
        def corr_func(x):
            return kendalltau(x['returns1'], x['returns2'])[0]
        
        return returns_df.rolling(window=window).apply(corr_func)
    
    return pd.Series()
