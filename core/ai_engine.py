"""
AI Engine - Machine learning with automatic retraining and model versioning
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import asyncio
import logging
import pickle
import json
from pathlib import Path
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler, RobustScaler
import optuna
import mlflow
import shap
import warnings

from models.feature_engineering import FeatureEngineer
from models.model_registry import ModelRegistry

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ModelType(Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE = "ensemble"


class AIConfig:
    """AI model configuration"""
    def __init__(self, config: dict):
        self.enabled = config['strategy']['ai_filter']['enabled']
        self.model_path = Path(config['strategy']['ai_filter']['model_path'])
        self.confidence_threshold = config['strategy']['ai_filter']['confidence_threshold']
        self.features = config['strategy']['ai_filter']['features']
        
        # Training settings
        self.retrain_enabled = config['model']['training']['enabled']
        self.retrain_schedule = config['model']['training']['schedule']
        self.retrain_threshold = config['model']['training']['retrain_threshold']
        self.cv_folds = config['model']['training']['cross_validation_folds']
        self.test_size = config['model']['training']['test_size']
        self.validation_size = config['model']['training']['validation_size']
        
        # AutoML settings
        self.automl_enabled = config['model']['training']['automl']['enabled']
        self.automl_time_budget = config['model']['training']['automl']['time_budget']
        self.automl_metric = config['model']['training']['automl']['metric']
        
        # Registry settings
        self.registry_enabled = config['model']['registry']['enabled']
        self.registry_path = Path(config['model']['registry']['path'])
        self.min_accuracy = config['model']['registry']['min_accuracy']
        self.min_precision = config['model']['registry']['min_precision']


class AIEngine:
    """
    Advanced AI engine with:
    - Multiple model support (XGBoost, LightGBM, Random Forest)
    - Automatic retraining pipeline
    - Model versioning and registry
    - Hyperparameter optimization (Optuna)
    - Feature importance tracking
    - Model performance monitoring
    - A/B testing support
    - SHAP explanations
    """
    
    def __init__(self, config: dict):
        self.config = AIConfig(config)
        self.logger = logging.getLogger(__name__)
        
        # Models
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        
        # Feature engineering
        self.feature_engineer = FeatureEngineer(config)
        
        # Model registry
        self.registry = ModelRegistry(self.config.registry_path) if self.config.registry_enabled else None
        
        # Performance tracking
        self.model_performance: Dict[str, Dict] = {}
        self.prediction_history: List[Dict] = []
        self.feature_importance_history: Dict[str, List] = {}
        
        # Training state
        self.last_training_time: Dict[str, datetime] = {}
        self.training_in_progress = False
        
        # SHAP explainer
        self.explainers: Dict[str, Any] = {}
        
        logger.info("AIEngine initialized with auto-retraining")
    
    async def initialize(self):
        """Initialize AI engine and load models"""
        logger.info("Initializing AI Engine...")
        
        # Create registry directory
        if self.config.registry_enabled:
            self.config.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Load models from registry
        if self.config.registry_enabled:
            await self._load_models_from_registry()
        else:
            await self._load_single_model()
        
        # Start background tasks
        if self.config.retrain_enabled:
            asyncio.create_task(self._retraining_scheduler())
        
        logger.info("✅ AI Engine initialized")
    
    async def _load_models_from_registry(self):
        """Load best models from registry"""
        try:
            # Get best model for each type
            for model_type in ModelType:
                best_model = self.registry.get_best_model(model_type.value)
                if best_model:
                    self.models[model_type.value] = best_model['model']
                    self.scalers[model_type.value] = best_model.get('scaler')
                    self.model_metadata[model_type.value] = best_model.get('metadata', {})
                    logger.info(f"Loaded {model_type.value} model from registry")
            
            # Load ensemble if available
            ensemble = self.registry.get_best_model('ensemble')
            if ensemble:
                self.models['ensemble'] = ensemble['model']
                logger.info("Loaded ensemble model from registry")
                
        except Exception as e:
            logger.error(f"Failed to load models from registry: {e}")
            await self._load_single_model()
    
    async def _load_single_model(self):
        """Load single model from path"""
        try:
            if self.config.model_path.exists():
                model_data = joblib.load(self.config.model_path)
                
                if isinstance(model_data, dict):
                    self.models['default'] = model_data.get('model')
                    self.scalers['default'] = model_data.get('scaler')
                    self.model_metadata['default'] = model_data.get('metadata', {})
                else:
                    self.models['default'] = model_data
                
                logger.info(f"Loaded model from {self.config.model_path}")
            else:
                logger.warning(f"Model not found at {self.config.model_path}")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    async def predict(self, features: np.ndarray, model_name: str = 'ensemble') -> Tuple[np.ndarray, float]:
        """
        Make predictions with confidence scores
        
        Args:
            features: Feature array
            model_name: Model to use
        
        Returns:
            Predictions and confidence scores
        """
        if not self.models:
            logger.error("No models loaded")
            return np.array([]), 0.0
        
        # Scale features
        if model_name in self.scalers and self.scalers[model_name]:
            features_scaled = self.scalers[model_name].transform(features)
        else:
            features_scaled = features
        
        # Get predictions from ensemble or single model
        if model_name == 'ensemble' and len(self.models) > 1:
            predictions = await self._ensemble_predict(features_scaled)
            confidence = self._calculate_ensemble_confidence(predictions)
        elif model_name in self.models:
            model = self.models[model_name]
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_scaled)
                predictions = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                confidence = np.mean(np.max(proba, axis=1))
            else:
                predictions = model.predict(features_scaled)
                confidence = 0.5
        else:
            logger.error(f"Model {model_name} not found")
            return np.array([]), 0.0
        
        # Store prediction
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'model': model_name,
            'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            'confidence': confidence
        })
        
        return predictions, confidence
    
    async def _ensemble_predict(self, features: np.ndarray) -> np.ndarray:
        """Ensemble prediction from multiple models"""
        all_predictions = []
        
        for name, model in self.models.items():
            if name == 'ensemble':
                continue
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)
                pred = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            else:
                pred = model.predict(features)
            
            # Weight by model performance
            weight = self.model_performance.get(name, {}).get('weight', 1.0)
            all_predictions.append(pred * weight)
        
        if all_predictions:
            return np.mean(all_predictions, axis=0)
        return np.array([])
    
    def _calculate_ensemble_confidence(self, predictions: np.ndarray) -> float:
        """Calculate confidence based on model agreement"""
        if len(predictions) == 0:
            return 0.0
        
        # Standard deviation as measure of disagreement
        std = np.std(predictions)
        confidence = 1.0 - min(std, 1.0)
        
        return confidence
    
    async def train(self, X: np.ndarray, y: np.ndarray, 
                   model_type: ModelType = ModelType.XGBOOST) -> Dict:
        """
        Train a new model
        
        Args:
            X: Features
            y: Labels
            model_type: Type of model to train
        
        Returns:
            Training results and metrics
        """
        self.training_in_progress = True
        start_time = datetime.now()
        
        try:
            # Split data
            split_idx = int(len(X) * (1 - self.config.test_size - self.config.validation_size))
            val_idx = int(len(X) * (1 - self.config.validation_size))
            
            X_train = X[:split_idx]
            y_train = y[:split_idx]
            X_val = X[split_idx:val_idx]
            y_val = y[split_idx:val_idx]
            X_test = X[val_idx:]
            y_test = y[val_idx:]
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if model_type == ModelType.XGBOOST:
                model, params = await self._train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val)
            elif model_type == ModelType.LIGHTGBM:
                model, params = await self._train_lightgbm(X_train_scaled, y_train, X_val_scaled, y_val)
            elif model_type == ModelType.RANDOM_FOREST:
                model, params = await self._train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val)
            else:
                model, params = await self._train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val)
            
            # Evaluate
            metrics = await self._evaluate_model(model, X_test_scaled, y_test)
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(model, X_train_scaled)
            
            # Create model metadata
            metadata = {
                'type': model_type.value,
                'params': params,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'training_date': datetime.now().isoformat(),
                'training_duration': (datetime.now() - start_time).total_seconds(),
                'n_samples': len(X),
                'n_features': X.shape[1]
            }
            
            # Store model
            model_name = f"{model_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            self.model_metadata[model_name] = metadata
            self.model_performance[model_name] = metrics
            
            # Save to registry
            if self.config.registry_enabled:
                await self._save_to_registry(model_name, model, scaler, metadata)
            
            # Update ensemble if needed
            if len(self.models) > 1:
                await self._update_ensemble()
            
            # Calculate SHAP values
            await self._calculate_shap(model, X_train_scaled[:100], model_name)
            
            training_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Model {model_name} trained in {training_time:.1f}s with accuracy {metrics['accuracy']:.3f}")
            
            return {
                'model_name': model_name,
                'metrics': metrics,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {}
        
        finally:
            self.training_in_progress = False
            self.last_training_time[model_type.value] = datetime.now()
    
    async def _train_xgboost(self, X_train, y_train, X_val, y_val) -> Tuple[Any, Dict]:
        """Train XGBoost with hyperparameter optimization"""
        if self.config.automl_enabled:
            # Optuna optimization
            study = optuna.create_study(direction='maximize', study_name='xgboost_optimization')
            study.optimize(
                lambda trial: self._xgboost_objective(trial, X_train, y_train, X_val, y_val),
                n_trials=50,
                timeout=self.config.automl_time_budget
            )
            best_params = study.best_params
        else:
            # Default parameters
            best_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1
            }
        
        # Train final model
        model = xgb.XGBClassifier(
            **best_params,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        return model, best_params
    
    def _xgboost_objective(self, trial, X_train, y_train, X_val, y_val) -> float:
        """Optuna objective for XGBoost"""
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_uniform('gamma', 0, 5)
        }
        
        model = xgb.XGBClassifier(**params, objective='binary:logistic', random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
        
        y_pred = model.predict(X_val)
        return accuracy_score(y_val, y_pred)
    
    async def _train_lightgbm(self, X_train, y_train, X_val, y_val) -> Tuple[Any, Dict]:
        """Train LightGBM model"""
        if self.config.automl_enabled:
            study = optuna.create_study(direction='maximize', study_name='lightgbm_optimization')
            study.optimize(
                lambda trial: self._lightgbm_objective(trial, X_train, y_train, X_val, y_val),
                n_trials=50,
                timeout=self.config.automl_time_budget
            )
            best_params = study.best_params
        else:
            best_params = {
                'num_leaves': 31,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        
        model = lgb.LGBMClassifier(**best_params, random_state=42)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        return model, best_params
    
    def _lightgbm_objective(self, trial, X_train, y_train, X_val, y_val) -> float:
        """Optuna objective for LightGBM"""
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50)
        }
        
        model = lgb.LGBMClassifier(**params, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
        
        y_pred = model.predict(X_val)
        return accuracy_score(y_val, y_pred)
    
    async def _train_random_forest(self, X_train, y_train, X_val, y_val) -> Tuple[Any, Dict]:
        """Train Random Forest model"""
        if self.config.automl_enabled:
            study = optuna.create_study(direction='maximize', study_name='rf_optimization')
            study.optimize(
                lambda trial: self._random_forest_objective(trial, X_train, y_train, X_val, y_val),
                n_trials=50,
                timeout=self.config.automl_time_budget
            )
            best_params = study.best_params
        else:
            best_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        
        model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        return model, best_params
    
    def _random_forest_objective(self, trial, X_train, y_train, X_val, y_val) -> float:
        """Optuna objective for Random Forest"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        }
        
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        return accuracy_score(y_val, y_pred)
    
    async def _evaluate_model(self, model, X_test, y_test) -> Dict:
        """Comprehensive model evaluation"""
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_proba[:, 1]) if y_proba.shape[1] > 1 else roc_auc_score(y_test, y_proba[:, 0])
        else:
            auc = 0.0
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'auc': auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return metrics
    
    def _calculate_feature_importance(self, model, X: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}
        return {}
    
    async def _calculate_shap(self, model, X_sample: np.ndarray, model_name: str):
        """Calculate SHAP values for model explainability"""
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            self.explainers[model_name] = explainer
            
            # Store average SHAP values
            self.model_metadata[model_name]['shap_mean'] = np.abs(shap_values).mean(axis=0).tolist()
            
        except Exception as e:
            logger.warning(f"SHAP calculation failed: {e}")
    
    async def _save_to_registry(self, name: str, model, scaler, metadata: Dict):
        """Save model to registry"""
        try:
            model_path = self.config.registry_path / f"{name}.joblib"
            scaler_path = self.config.registry_path / f"{name}_scaler.joblib"
            metadata_path = self.config.registry_path / f"{name}_metadata.json"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Register in model registry
            if self.registry:
                self.registry.register_model(
                    name=name,
                    model=model,
                    scaler=scaler,
                    metadata=metadata
                )
            
            logger.info(f"Model {name} saved to registry")
            
        except Exception as e:
            logger.error(f"Failed to save model to registry: {e}")
    
    async def _update_ensemble(self):
        """Update ensemble model with weighted voting"""
        if len(self.models) < 2:
            return
        
        # Calculate weights based on performance
        total_performance = sum(m.get('accuracy', 0) for m in self.model_performance.values())
        
        if total_performance > 0:
            for name in self.model_performance:
                self.model_performance[name]['weight'] = self.model_performance[name].get('accuracy', 0) / total_performance
        
        # Create ensemble metadata
        ensemble_metadata = {
            'type': 'ensemble',
            'models': list(self.models.keys()),
            'weights': {name: m.get('weight', 1.0) for name, m in self.model_performance.items()},
            'created_at': datetime.now().isoformat()
        }
        
        self.model_metadata['ensemble'] = ensemble_metadata
    
    async def _retraining_scheduler(self):
        """Background task for automatic retraining"""
        while True:
            try:
                # Check if retraining is needed
                for model_name, metadata in self.model_metadata.items():
                    if model_name == 'ensemble':
                        continue
                    
                    last_train = self.last_training_time.get(model_name)
                    if last_train:
                        # Check schedule
                        if self.config.retrain_schedule == 'daily':
                            should_retrain = (datetime.now() - last_train).days >= 1
                        elif self.config.retrain_schedule == 'weekly':
                            should_retrain = (datetime.now() - last_train).days >= 7
                        elif self.config.retrain_schedule == 'monthly':
                            should_retrain = (datetime.now() - last_train).days >= 30
                        else:
                            should_retrain = False
                        
                        # Check performance degradation
                        if not should_retrain and 'metrics' in metadata:
                            current_perf = metadata['metrics'].get('accuracy', 0)
                            if current_perf < self.config.min_accuracy - self.config.retrain_threshold:
                                should_retrain = True
                                logger.info(f"Performance degradation detected for {model_name}")
                        
                        if should_retrain and not self.training_in_progress:
                            logger.info(f"Starting scheduled retraining for {model_name}")
                            # Would need to fetch new training data here
                            # await self.train(X_new, y_new, ModelType(model_name))
                
                # Sleep based on schedule
                if self.config.retrain_schedule == 'daily':
                    await asyncio.sleep(3600)  # Check hourly
                else:
                    await asyncio.sleep(3600 * 6)  # Check every 6 hours
                
            except Exception as e:
                logger.error(f"Retraining scheduler error: {e}")
                await asyncio.sleep(3600)
    
    def get_feature_importance(self, model_name: str = 'ensemble') -> Dict:
        """Get feature importance for model"""
        if model_name in self.model_metadata:
            return self.model_metadata[model_name].get('feature_importance', {})
        return {}
    
    def get_model_performance(self, model_name: Optional[str] = None) -> Dict:
        """Get model performance metrics"""
        if model_name:
            return self.model_performance.get(model_name, {})
        return self.model_performance
    
    def get_shap_explanation(self, features: np.ndarray, model_name: str) -> Optional[np.ndarray]:
        """Get SHAP explanation for prediction"""
        if model_name in self.explainers:
            return self.explainers[model_name].shap_values(features)
        return None
    
    async def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down AI Engine...")
        
        # Save final state
        if self.config.registry_enabled:
            final_metadata = {
                'timestamp': datetime.now().isoformat(),
                'models': list(self.models.keys()),
                'performance': self.model_performance,
                'prediction_count': len(self.prediction_history)
            }
            
            metadata_path = self.config.registry_path / 'final_state.json'
            with open(metadata_path, 'w') as f:
                json.dump(final_metadata, f, indent=2)
        
        logger.info("AI Engine shutdown complete")