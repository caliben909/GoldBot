"""
Model Validator - Comprehensive model validation and performance monitoring
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

from models.xgboost_model import XGBoostModel
from models.lstm_model import LSTMModel
from models.ensemble_model import EnsembleModel
from models.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Comprehensive model validation service that:
    - Validates model performance on new data
    - Checks for concept drift
    - Monitors model degradation
    - Ensures reliability and consistency
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.validation_history = []
        
        # Validation thresholds
        self.min_accuracy = config.get('min_accuracy', 0.6)
        self.min_precision = config.get('min_precision', 0.5)
        self.min_recall = config.get('min_recall', 0.5)
        self.min_auc = config.get('min_auc', 0.6)
        
        logger.info("ModelValidator initialized")
    
    def validate_model(self, model: Any, X: np.ndarray, y: np.ndarray, 
                      model_name: str = 'unknown') -> Dict:
        """
        Validate model performance
        
        Args:
            model: Trained model
            X: Features
            y: Labels
            model_name: Model identifier
            
        Returns:
            Validation results
        """
        try:
            # Make predictions
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X)
                y_pred = np.argmax(y_proba, axis=1)
            else:
                y_pred = model.predict(X)
                y_proba = None
            
            # Calculate metrics
            metrics = self._calculate_metrics(y, y_pred, y_proba)
            
            # Check if model meets minimum performance requirements
            passes, reasons = self._check_performance_requirements(metrics)
            
            # Record validation history
            self._record_validation(model_name, metrics, passes, reasons)
            
            # Log results
            self._log_validation_result(model_name, metrics, passes, reasons)
            
            return {
                'model': model_name,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'passes_validation': passes,
                'fail_reasons': reasons,
                'status': 'pass' if passes else 'fail'
            }
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return {
                'model': model_name,
                'timestamp': datetime.now().isoformat(),
                'metrics': {},
                'passes_validation': False,
                'fail_reasons': [str(e)],
                'status': 'error'
            }
    
    def validate_signal_prediction(self, model: Any, df: pd.DataFrame, 
                                 actual_outcome: bool) -> Dict:
        """Validate signal prediction against actual outcome"""
        try:
            features = self.feature_engineer.create_features(df)
            features_normalized = self.feature_engineer.normalize_features(features)
            
            predictions, probabilities = model.predict(features_normalized.values)
            
            if len(predictions) == 0:
                return {
                    'success': False,
                    'prediction': None,
                    'actual': actual_outcome,
                    'confidence': 0.0,
                    'correct': False
                }
            
            prediction = predictions[-1]
            confidence = np.max(probabilities[-1]) if probabilities is not None else 0.5
            correct = bool(prediction) == actual_outcome
            
            return {
                'success': True,
                'prediction': bool(prediction),
                'actual': actual_outcome,
                'confidence': confidence,
                'correct': correct
            }
            
        except Exception as e:
            logger.error(f"Signal prediction validation failed: {e}")
            return {
                'success': False,
                'prediction': None,
                'actual': actual_outcome,
                'confidence': 0.0,
                'correct': False
            }
    
    def check_model_degradation(self, recent_metrics: List[Dict], 
                              window: int = 10) -> Tuple[bool, float]:
        """
        Check if model performance is degraded
        
        Args:
            recent_metrics: List of recent validation results
            window: Number of recent results to consider
            
        Returns:
            Degradation status and average performance drop
        """
        if len(recent_metrics) < window:
            return False, 0.0
        
        # Get recent metrics
        recent = recent_metrics[-window:]
        older = recent_metrics[:-window]
        
        # Calculate average performance
        recent_accuracy = np.mean([m['accuracy'] for m in recent if 'accuracy' in m])
        older_accuracy = np.mean([m['accuracy'] for m in older if 'accuracy' in m])
        
        if older_accuracy == 0:
            return False, 0.0
        
        performance_drop = (older_accuracy - recent_accuracy) / older_accuracy
        
        # Consider performance degraded if drop is more than 10%
        is_degraded = performance_drop > 0.1
        
        if is_degraded:
            logger.warning(f"Model performance degraded by {performance_drop:.1%}")
        
        return is_degraded, performance_drop
    
    def detect_concept_drift(self, historical_features: np.ndarray,
                            recent_features: np.ndarray) -> Tuple[bool, float]:
        """
        Detect concept drift in features
        
        Args:
            historical_features: Historical feature distribution
            recent_features: Recent feature distribution
            
        Returns:
            Drift status and drift score
        """
        try:
            # Calculate feature distribution statistics
            hist_mean = np.mean(historical_features, axis=0)
            hist_std = np.std(historical_features, axis=0)
            
            recent_mean = np.mean(recent_features, axis=0)
            recent_std = np.std(recent_features, axis=0)
            
            # Calculate drift score for each feature (Z-score)
            drift_scores = []
            for i in range(len(hist_mean)):
                if hist_std[i] > 0:
                    z_score = abs(recent_mean[i] - hist_mean[i]) / hist_std[i]
                    drift_scores.append(z_score)
            
            avg_drift_score = np.mean(drift_scores)
            
            # Consider drift significant if average Z-score > 2
            has_drift = avg_drift_score > 2
            
            if has_drift:
                logger.warning(f"Concept drift detected with score: {avg_drift_score:.2f}")
            
            return has_drift, avg_drift_score
            
        except Exception as e:
            logger.error(f"Concept drift detection failed: {e}")
            return False, 0.0
    
    def get_validation_summary(self, model_name: str = None) -> Dict:
        """Get comprehensive validation summary"""
        if model_name:
            history = [h for h in self.validation_history if h['model'] == model_name]
        else:
            history = self.validation_history
        
        if not history:
            return {
                'total_validations': 0,
                'pass_count': 0,
                'fail_count': 0,
                'avg_accuracy': 0.0,
                'avg_precision': 0.0,
                'avg_recall': 0.0,
                'avg_auc': 0.0,
                'pass_rate': 0.0
            }
        
        pass_count = sum(1 for h in history if h['passes_validation'])
        fail_count = len(history) - pass_count
        
        avg_accuracy = np.mean([h['metrics']['accuracy'] for h in history if 'accuracy' in h['metrics']])
        avg_precision = np.mean([h['metrics']['precision'] for h in history if 'precision' in h['metrics']])
        avg_recall = np.mean([h['metrics']['recall'] for h in history if 'recall' in h['metrics']])
        avg_auc = np.mean([h['metrics']['auc'] for h in history if 'auc' in h['metrics']])
        
        return {
            'total_validations': len(history),
            'pass_count': pass_count,
            'fail_count': fail_count,
            'avg_accuracy': avg_accuracy,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_auc': avg_auc,
            'pass_rate': pass_count / len(history)
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_proba: Optional[np.ndarray]) -> Dict:
        """Calculate comprehensive validation metrics"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix,
            classification_report
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
        
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        return metrics
    
    def _check_performance_requirements(self, metrics: Dict) -> Tuple[bool, List[str]]:
        """Check if metrics meet minimum requirements"""
        fail_reasons = []
        
        if 'accuracy' in metrics and metrics['accuracy'] < self.min_accuracy:
            fail_reasons.append(f"Accuracy {metrics['accuracy']:.2f} < {self.min_accuracy}")
        
        if 'precision' in metrics and metrics['precision'] < self.min_precision:
            fail_reasons.append(f"Precision {metrics['precision']:.2f} < {self.min_precision}")
        
        if 'recall' in metrics and metrics['recall'] < self.min_recall:
            fail_reasons.append(f"Recall {metrics['recall']:.2f} < {self.min_recall}")
        
        if 'auc' in metrics and metrics['auc'] < self.min_auc:
            fail_reasons.append(f"AUC {metrics['auc']:.2f} < {self.min_auc}")
        
        return len(fail_reasons) == 0, fail_reasons
    
    def _record_validation(self, model_name: str, metrics: Dict, 
                         passes: bool, fail_reasons: List[str]):
        """Record validation in history"""
        self.validation_history.append({
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'passes_validation': passes,
            'fail_reasons': fail_reasons
        })
        
        # Keep only last 100 validation records
        if len(self.validation_history) > 100:
            self.validation_history = self.validation_history[-100:]
    
    def _log_validation_result(self, model_name: str, metrics: Dict, 
                              passes: bool, fail_reasons: List[str]):
        """Log validation result"""
        status = "✅ PASSED" if passes else "❌ FAILED"
        
        logger.info(
            f"Validation {status} for {model_name}\n"
            f"  Accuracy: {metrics.get('accuracy', 0):.3f}\n"
            f"  Precision: {metrics.get('precision', 0):.3f}\n"
            f"  Recall: {metrics.get('recall', 0):.3f}\n"
            f"  F1 Score: {metrics.get('f1_score', 0):.3f}\n"
            f"  AUC: {metrics.get('auc', 0):.3f}"
        )
        
        if fail_reasons:
            for reason in fail_reasons:
                logger.warning(f"  Reason: {reason}")
