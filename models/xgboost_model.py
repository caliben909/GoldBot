"""
XGBoost Model - XGBoost classifier for trading signal prediction
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import joblib
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix)

from models.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    XGBoost classifier for trading signal prediction
    
    Features:
    - Hyperparameter optimization
    - Feature importance tracking
    - Model persistence
    - Performance monitoring
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_engineer = FeatureEngineer(config)
        self.feature_names = []
        self.model_path = Path(config.get('model_path', 'models/xgboost_model.joblib'))
        self.scaler_path = Path(config.get('scaler_path', 'models/xgboost_scaler.joblib'))
        
        logger.info("XGBoostModel initialized")
    
    def train(self, X: np.ndarray, y: np.ndarray, 
             params: Optional[Dict] = None) -> Dict:
        """
        Train XGBoost model with hyperparameter optimization
        
        Args:
            X: Features
            y: Labels
            params: Optional parameters
            
        Returns:
            Training results and metrics
        """
        if params is None:
            params = self._get_default_params()
        
        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_scaled, y)
        
        # Make predictions on training data
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y, y_pred, y_proba)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_engineer.get_feature_names(),
                self.model.feature_importances_
            ))
        
        logger.info(f"XGBoost model trained with accuracy: {metrics['accuracy']:.3f}")
        
        return metrics
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Args:
            features: Feature array
            
        Returns:
            Predictions and probabilities
        """
        if self.model is None or self.scaler is None:
            raise Exception("Model not trained")
        
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        return predictions, probabilities
    
    def load(self, model_path: Optional[Path] = None, 
            scaler_path: Optional[Path] = None):
        """Load trained model from disk"""
        if model_path is None:
            model_path = self.model_path
        
        if scaler_path is None:
            scaler_path = self.scaler_path
        
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded XGBoost model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            return False
    
    def save(self, model_path: Optional[Path] = None, 
            scaler_path: Optional[Path] = None):
        """Save trained model to disk"""
        if self.model is None or self.scaler is None:
            raise Exception("Model not trained")
        
        if model_path is None:
            model_path = self.model_path
        
        if scaler_path is None:
            scaler_path = self.scaler_path
        
        try:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved XGBoost model to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save XGBoost model: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(
                self.feature_engineer.get_feature_names(),
                self.model.feature_importances_
            ))
        return {}
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_proba: np.ndarray) -> Dict:
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_proba[:, 1]),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        return metrics
    
    def _get_default_params(self) -> Dict:
        """Get default XGBoost parameters"""
        return {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'random_state': 42
        }
    
    def optimize_params(self, X: np.ndarray, y: np.ndarray, 
                      n_trials: int = 50) -> Dict:
        """Optimize hyperparameters using Optuna"""
        from optuna import create_study
        from optuna.integration import XGBoostPruningCallback
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_uniform('gamma', 0, 5)
            }
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_val_scaled)
            return accuracy_score(y_val, y_pred)
        
        study = create_study(direction='maximize', study_name='xgboost_optimization')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best XGBoost params: {study.best_params}")
        logger.info(f"Best accuracy: {study.best_value:.3f}")
        
        return study.best_params
