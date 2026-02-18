"""
Feature Engineering - Creates features for ML models
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from ta import add_all_ta_features
from ta.utils import dropna
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Creates and selects features for machine learning models
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.feature_names = []
        self.feature_importance = {}
        self.scaler = None
        self.normalization_params = {}
        
        logger.info("FeatureEngineer initialized")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Price-based features
            features = self._add_price_features(df, features)
            
            # Volume-based features
            features = self._add_volume_features(df, features)
            
            # Technical indicators
            features = self._add_technical_indicators(df, features)
            
            # Market microstructure
            features = self._add_microstructure_features(df, features)
            
            # SMC-specific features
            features = self._add_smc_features(df, features)
            
            # Session-based features
            features = self._add_session_features(df, features)
            
            # Lagged features
            features = self._add_lagged_features(features)
            
            # Rolling statistics
            features = self._add_rolling_statistics(features)
            
            # Remove NaN values
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(method='ffill').fillna(method='bfill')
            
            self.feature_names = features.columns.tolist()
            logger.info(f"Created {len(self.feature_names)} features")
            
            return features
            
        except Exception as e:
            logger.error(f"Feature creation failed: {e}")
            return pd.DataFrame()
    
    def _add_price_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Returns
        features['returns_1'] = df['close'].pct_change(1)
        features['returns_5'] = df['close'].pct_change(5)
        features['returns_10'] = df['close'].pct_change(10)
        features['returns_20'] = df['close'].pct_change(20)
        
        # Log returns
        features['log_returns_1'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price position
        features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Price range
        features['range'] = df['high'] - df['low']
        features['range_pct'] = features['range'] / df['close']
        
        # Candle patterns
        features['candle_body'] = abs(df['close'] - df['open'])
        features['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        features['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        return features
    
    def _add_volume_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        # Volume ratios
        features['volume_ratio_5'] = df['volume'] / df['volume'].rolling(5).mean()
        features['volume_ratio_20'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Volume price trend
        features['vpt'] = (df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))).cumsum()
        
        # On-balance volume
        obv = np.where(df['close'] > df['close'].shift(1), df['volume'],
                      np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
        features['obv'] = pd.Series(obv).cumsum()
        
        # Volume-weighted price
        features['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
        
        return features
    
    def _add_technical_indicators(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        # Moving averages
        features['sma_10'] = df['close'].rolling(10).mean()
        features['sma_20'] = df['close'].rolling(20).mean()
        features['sma_50'] = df['close'].rolling(50).mean()
        
        # EMA
        features['ema_12'] = df['close'].ewm(span=12).mean()
        features['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        features['macd'] = features['ema_12'] - features['ema_26']
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        features['bb_upper'] = bb_middle + (bb_std * 2)
        features['bb_lower'] = bb_middle - (bb_std * 2)
        features['bb_width'] = features['bb_upper'] - features['bb_lower']
        features['bb_position'] = (df['close'] - features['bb_lower']) / features['bb_width']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        features['atr'] = true_range.rolling(14).mean()
        features['atr_pct'] = features['atr'] / df['close']
        
        return features
    
    def _add_microstructure_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Spread (if available)
        if 'spread' in df.columns:
            features['spread'] = df['spread']
            features['spread_pct'] = df['spread'] / df['close']
        
        # Tick volume ratio
        if 'tick_volume' in df.columns:
            features['tick_ratio'] = df['tick_volume'] / df['volume']
        
        # Bid-ask imbalance (if available)
        if 'bid' in df.columns and 'ask' in df.columns:
            features['bid_ask_imbalance'] = (df['ask'] - df['bid']) / df['close']
        
        return features
    
    def _add_smc_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add Smart Money Concepts features"""
        # Swing points (simplified)
        features['swing_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        features['swing_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        
        # Distance to recent swings
        last_high = df['high'].where(features['swing_high']).ffill()
        last_low = df['low'].where(features['swing_low']).ffill()
        
        features['dist_to_high'] = (df['close'] - last_high) / df['close']
        features['dist_to_low'] = (df['close'] - last_low) / df['close']
        
        return features
    
    def _add_session_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add session-based features"""
        # Time-based features
        features['hour'] = df.index.hour
        features['minute'] = df.index.minute
        features['day_of_week'] = df.index.dayofweek
        
        # Session indicators
        features['is_asia'] = ((features['hour'] >= 0) & (features['hour'] < 9)).astype(int)
        features['is_london'] = ((features['hour'] >= 8) & (features['hour'] < 17)).astype(int)
        features['is_ny'] = ((features['hour'] >= 13) & (features['hour'] < 22)).astype(int)
        features['is_overlap'] = ((features['hour'] >= 13) & (features['hour'] < 17)).astype(int)
        
        return features
    
    def _add_lagged_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features"""
        lag_periods = [1, 2, 3, 5, 10]
        
        for col in ['returns_1', 'volume_ratio_5', 'rsi', 'macd']:
            if col in features.columns:
                for lag in lag_periods:
                    features[f'{col}_lag_{lag}'] = features[col].shift(lag)
        
        return features
    
    def _add_rolling_statistics(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics"""
        windows = [5, 10, 20]
        
        for col in ['returns_1', 'volume_ratio_5']:
            if col in features.columns:
                for window in windows:
                    features[f'{col}_mean_{window}'] = features[col].rolling(window).mean()
                    features[f'{col}_std_{window}'] = features[col].rolling(window).std()
                    features[f'{col}_skew_{window}'] = features[col].rolling(window).skew()
                    features[f'{col}_kurt_{window}'] = features[col].rolling(window).kurt()
        
        return features
    
    def normalize_features(self, features: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize features to zero mean and unit variance"""
        if fit:
            self.normalization_params = {
                'mean': features.mean(),
                'std': features.std()
            }
        
        if self.normalization_params:
            normalized = (features - self.normalization_params['mean']) / self.normalization_params['std']
            normalized = normalized.replace([np.inf, -np.inf], np.nan)
            normalized = normalized.fillna(0)
            
            return normalized
        
        return features
    
    def select_features(self, features: pd.DataFrame, target: pd.Series, 
                       n_features: int = 20) -> pd.DataFrame:
        """Select most important features"""
        from sklearn.ensemble import RandomForestRegressor
        
        try:
            # Train random forest for feature importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(features.fillna(0), target)
            
            # Get feature importance
            importance = pd.DataFrame({
                'feature': features.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Select top features
            top_features = importance.head(n_features)['feature'].tolist()
            self.feature_importance = importance.set_index('feature')['importance'].to_dict()
            
            logger.info(f"Selected {len(top_features)} features")
            
            return features[top_features]
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names