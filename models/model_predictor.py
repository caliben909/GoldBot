"""
Model Predictor - Real-time prediction service for trading signals
"""
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

from models.feature_engineering import FeatureEngineer
from models.xgboost_model import XGBoostModel
from models.lstm_model import LSTMModel
from models.ensemble_model import EnsembleModel

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Real-time prediction service that:
    - Handles feature engineering
    - Manages model loading
    - Provides predictions with confidence scores
    - Supports multiple model types
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.models = {}
        self.enabled = config.get('enabled', False)
        
        # Initialize models
        self._init_models()
        
        logger.info("ModelPredictor initialized")
    
    def _init_models(self):
        """Initialize available models"""
        try:
            # XGBoost
            if self.config.get('xgboost', {}).get('enabled', True):
                self.models['xgboost'] = XGBoostModel(self.config.get('xgboost', {}))
            
            # LSTM
            if self.config.get('lstm', {}).get('enabled', False):
                self.models['lstm'] = LSTMModel(self.config.get('lstm', {}))
            
            # Ensemble
            if self.config.get('ensemble', {}).get('enabled', False):
                self.models['ensemble'] = EnsembleModel(self.config.get('ensemble', {}))
            
            # Load trained models
            self._load_models()
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
    
    def _load_models(self):
        """Load trained models from disk"""
        for name, model in self.models.items():
            config = self.config.get(name, {})
            model_path = config.get('model_path')
            scaler_path = config.get('scaler_path')
            
            if model_path:
                success = model.load(model_path, scaler_path)
                if not success:
                    logger.warning(f"Failed to load {name} model")
    
    def predict(self, df: pd.DataFrame, model_name: str = 'xgboost') -> Tuple[np.ndarray, float]:
        """
        Make prediction from OHLCV data
        
        Args:
            df: OHLCV DataFrame
            model_name: Model to use (xgboost, lstm, or ensemble)
            
        Returns:
            Predictions and confidence scores
        """
        try:
            if model_name not in self.models:
                raise Exception(f"Model {model_name} not available")
            
            model = self.models[model_name]
            
            # Create features
            features = self.feature_engineer.create_features(df)
            if features.empty:
                logger.warning("No features created")
                return np.array([]), 0.0
            
            # Normalize features
            features_normalized = self.feature_engineer.normalize_features(features)
            
            # Make predictions
            predictions, probabilities = model.predict(features_normalized.values)
            
            # Calculate confidence
            if probabilities is not None and len(probabilities) > 0:
                confidence = np.mean(np.max(probabilities, axis=1))
            else:
                confidence = 0.5
            
            logger.debug(f"Prediction: {predictions[-1] if len(predictions) > 0 else 'none'}, "
                        f"Confidence: {confidence:.2f}")
            
            return predictions, confidence
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return np.array([]), 0.0
    
    def predict_signal(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Predict if current market conditions are favorable for a trade
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Signal (buy/sell) and confidence
        """
        try:
            # Get predictions from primary model
            predictions, confidence = self.predict(df, 'xgboost')
            
            if len(predictions) == 0 or confidence < self.config.get('confidence_threshold', 0.6):
                return False, 0.0
            
            # Determine signal direction
            last_prediction = predictions[-1]
            
            return last_prediction == 1, confidence
            
        except Exception as e:
            logger.error(f"Signal prediction failed: {e}")
            return False, 0.0
    
    def get_model_info(self) -> Dict:
        """Get information about available models"""
        info = {
            'enabled': self.enabled,
            'models': []
        }
        
        for name, model in self.models.items():
            model_info = {
                'name': name,
                'type': type(model).__name__,
                'available': True
            }
            
            if hasattr(model, 'get_feature_importance'):
                model_info['feature_importance'] = model.get_feature_importance()
            
            info['models'].append(model_info)
        
        return info
    
    def update_model(self, model_name: str, model_path: str, 
                    scaler_path: Optional[str] = None):
        """Update model with new version"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return False
        
        try:
            success = self.models[model_name].load(model_path, scaler_path)
            if success:
                logger.info(f"Updated {model_name} model")
                return True
            else:
                logger.error(f"Failed to update {model_name} model")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update {model_name} model: {e}")
            return False
    
    async def warmup(self, symbols: List[str], timeframe: str = '1h'):
        """Warmup models by loading historical data and making test predictions"""
        try:
            logger.info("Warming up models...")
            
            # For each symbol, load some historical data and make a test prediction
            for symbol in symbols:
                logger.debug(f"Warming up for {symbol}")
                
                # This would typically load real historical data
                # For now, create a dummy DataFrame
                date_range = pd.date_range(start='2023-01-01', periods=50, freq=timeframe)
                dummy_df = pd.DataFrame({
                    'open': np.random.uniform(1700, 1800, 50),
                    'high': np.random.uniform(1700, 1800, 50),
                    'low': np.random.uniform(1700, 1800, 50),
                    'close': np.random.uniform(1700, 1800, 50),
                    'volume': np.random.uniform(1000, 10000, 50)
                }, index=date_range)
                
                # Make a test prediction
                await asyncio.to_thread(self.predict, dummy_df)
                
            logger.info("Models warmed up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model warmup failed: {e}")
            return False
