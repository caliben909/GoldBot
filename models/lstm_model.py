"""
LSTM Model - Deep learning for sequence prediction
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class LSTMModel:
    """
    LSTM neural network for time series prediction
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.sequence_length = 60  # Lookback period
        self.n_features = 10        # Number of features
        self.n_outputs = 1          # Binary classification
        
        # Model architecture
        self.lstm_units = [64, 32]
        self.dropout_rate = 0.2
        self.learning_rate = 0.001
        
        logger.info("LSTMModel initialized")
    
    def build_model(self):
        """Build LSTM model architecture"""
        try:
            model = models.Sequential()
            
            # First LSTM layer
            model.add(layers.LSTM(
                self.lstm_units[0],
                return_sequences=True,
                input_shape=(self.sequence_length, self.n_features)
            ))
            model.add(layers.Dropout(self.dropout_rate))
            
            # Second LSTM layer
            model.add(layers.LSTM(self.lstm_units[1], return_sequences=False))
            model.add(layers.Dropout(self.dropout_rate))
            
            # Dense layers
            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.Dropout(self.dropout_rate))
            model.add(layers.Dense(16, activation='relu'))
            
            # Output layer
            model.add(layers.Dense(self.n_outputs, activation='sigmoid'))
            
            # Compile model
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC()]
            )
            
            self.model = model
            logger.info("LSTM model built successfully")
            
        except Exception as e:
            logger.error(f"Failed to build LSTM model: {e}")
    
    def load_model(self, model_path: str):
        """Load trained model"""
        try:
            self.model = keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def save_model(self, model_path: str):
        """Save trained model"""
        try:
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32):
        """Train LSTM model"""
        if self.model is None:
            self.build_model()
        
        try:
            # Callbacks
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            logger.info(f"Model trained for {len(history.history['loss'])} epochs")
            
            return history
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            logger.error("Model not loaded")
            return np.array([])
        
        try:
            predictions = self.model.predict(X)
            return predictions.flatten()
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return np.array([])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        return self.predict(X)
    
    def prepare_sequences(self, data: np.ndarray) -> np.ndarray:
        """Prepare sequences for LSTM input"""
        X = []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
        
        return np.array(X)