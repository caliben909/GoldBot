"""
Ensemble Model - Combines multiple models for robust predictions
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class EnsembleModel:
    """
    Ensemble of multiple models with weighted voting
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.models = []
        self.weights = []
        self.model_names = []
        self.performance_history = defaultdict(list)
        
        logger.info("EnsembleModel initialized")
    
    def add_model(self, model, name: str, weight: float = 1.0):
        """Add model to ensemble"""
        self.models.append(model)
        self.model_names.append(name)
        self.weights.append(weight)
        
        logger.info(f"Added model {name} with weight {weight}")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Ensemble prediction"""
        if not self.models:
            logger.error("No models in ensemble")
            return np.array([])
        
        predictions = []
        
        for i, model in enumerate(self.models):
            try:
                pred = model.predict(features)
                if len(pred) > 0:
                    predictions.append(pred * self.weights[i])
            except Exception as e:
                logger.error(f"Model {self.model_names[i]} failed: {e}")
        
        if not predictions:
            return np.array([])
        
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0) / sum(self.weights)
        
        return ensemble_pred
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Ensemble probability prediction"""
        if not self.models:
            return np.array([])
        
        probas = []
        
        for i, model in enumerate(self.models):
            try:
                proba = model.predict_proba(features)
                if len(proba) > 0:
                    probas.append(proba * self.weights[i])
            except Exception as e:
                logger.error(f"Model {self.model_names[i]} failed: {e}")
        
        if not probas:
            return np.array([])
        
        # Weighted average of probabilities
        ensemble_proba = np.sum(probas, axis=0) / sum(self.weights)
        
        return ensemble_proba
    
    def update_weights(self, performance_scores: Dict[str, float]):
        """Update model weights based on performance"""
        total_score = sum(performance_scores.values())
        
        if total_score > 0:
            for i, name in enumerate(self.model_names):
                if name in performance_scores:
                    self.weights[i] = performance_scores[name] / total_score
                    
            logger.info(f"Updated ensemble weights: {dict(zip(self.model_names, self.weights))}")
    
    def get_model_count(self) -> int:
        """Get number of models in ensemble"""
        return len(self.models)