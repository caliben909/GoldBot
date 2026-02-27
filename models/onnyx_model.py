"""
ONNX Model - Production-grade ONNX runtime inference
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import onnxruntime as ort
import logging
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)

class ONNXModel:
    """
    ONNX runtime model for high-performance inference
    Supports CPU, CUDA, and TensorRT execution providers
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.model_path = Path(config['strategy']['ai_filter']['model_path'])
        self.session = None
        self.input_names = []
        self.output_names = []
        self.model_metadata = {}
        self.execution_provider = 'CPUExecutionProvider'
        
        logger.info(f"ONNXModel initialized with path: {self.model_path}")
    
    def load_model(self):
        """Load ONNX model with optimal execution provider"""
        try:
            if not self.model_path.exists():
                logger.error(f"Model not found: {self.model_path}")
                return False
            
            # Check for CUDA availability
            providers = ort.get_available_providers()
            logger.info(f"Available providers: {providers}")
            
            if 'CUDAExecutionProvider' in providers:
                self.execution_provider = 'CUDAExecutionProvider'
                logger.info("Using CUDA execution provider")
            elif 'TensorrtExecutionProvider' in providers:
                self.execution_provider = 'TensorrtExecutionProvider'
                logger.info("Using TensorRT execution provider")
            else:
                logger.info("Using CPU execution provider")
            
            # Create inference session
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=[self.execution_provider]
            )
            
            # Get model metadata
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # Load metadata if available
            metadata_path = self.model_path.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Inputs: {self.input_names}")
            logger.info(f"Outputs: {self.output_names}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Run inference on features"""
        if self.session is None:
            logger.error("Model not loaded")
            return np.array([])
        
        try:
            # Prepare input dict
            input_dict = {}
            for i, name in enumerate(self.input_names):
                if i < features.shape[1]:
                    input_dict[name] = features[:, i:i+1].astype(np.float32)
                else:
                    input_dict[name] = features.astype(np.float32)
            
            # Run inference
            start_time = time.time()
            outputs = self.session.run(self.output_names, input_dict)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            logger.debug(f"Inference time: {inference_time:.2f}ms")
            
            return outputs[0]
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return np.array([])
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        outputs = self.predict(features)
        
        if len(outputs) == 0:
            return np.array([])
        
        # Apply softmax if needed
        if outputs.ndim == 2 and outputs.shape[1] > 1:
            exp_outputs = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
            return exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
        
        return outputs
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'path': str(self.model_path),
            'provider': self.execution_provider,
            'inputs': self.input_names,
            'outputs': self.output_names,
            'metadata': self.model_metadata
        }
