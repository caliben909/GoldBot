"""
Model Registry - Version control and management for ML models
"""
import json
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import shutil
import hashlib

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Model version control and registry system
    
    Features:
    - Model versioning
    - Performance tracking
    - Automatic rollback
    - Model metadata storage
    - A/B testing support
    """
    
    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Registry structure
        self.models_path = self.registry_path / 'models'
        self.metadata_path = self.registry_path / 'metadata'
        self.backup_path = self.registry_path / 'backup'
        
        self.models_path.mkdir(exist_ok=True)
        self.metadata_path.mkdir(exist_ok=True)
        self.backup_path.mkdir(exist_ok=True)
        
        # Load registry index
        self.index_file = self.registry_path / 'index.json'
        self.index = self._load_index()
        
        logger.info(f"ModelRegistry initialized at {registry_path}")
    
    def _load_index(self) -> Dict:
        """Load registry index"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {
            'models': {},
            'production': None,
            'staging': None,
            'archive': []
        }
    
    def _save_index(self):
        """Save registry index"""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def register_model(self, name: str, model: Any, scaler: Any = None, 
                      metadata: Dict = None) -> str:
        """
        Register a new model version
        
        Args:
            name: Model name
            model: Model object
            scaler: Scaler object
            metadata: Model metadata
        
        Returns:
            Model version ID
        """
        # Generate version ID
        version = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_id = f"{name}_{version}"
        
        # Save model
        model_path = self.models_path / f"{version_id}.joblib"
        joblib.dump(model, model_path)
        
        # Save scaler if provided
        scaler_path = None
        if scaler:
            scaler_path = self.models_path / f"{version_id}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
        
        # Calculate model hash
        with open(model_path, 'rb') as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Prepare metadata
        model_metadata = {
            'version_id': version_id,
            'name': name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'model_path': str(model_path),
            'scaler_path': str(scaler_path) if scaler_path else None,
            'model_hash': model_hash,
            'metrics': metadata.get('metrics', {}) if metadata else {},
            'params': metadata.get('params', {}) if metadata else {},
            'feature_importance': metadata.get('feature_importance', {}),
            'training_date': metadata.get('training_date'),
            'training_duration': metadata.get('training_duration'),
            'n_samples': metadata.get('n_samples'),
            'n_features': metadata.get('n_features'),
            'status': 'staging'
        }
        
        # Save metadata
        metadata_path = self.metadata_path / f"{version_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Update index
        if name not in self.index['models']:
            self.index['models'][name] = []
        
        self.index['models'][name].append({
            'version_id': version_id,
            'created_at': datetime.now().isoformat(),
            'metrics': model_metadata['metrics'],
            'status': 'staging'
        })
        
        # Set as staging if first model
        if self.index['staging'] is None:
            self.index['staging'] = version_id
        
        self._save_index()
        
        logger.info(f"Registered model {version_id}")
        return version_id
    
    def promote_to_production(self, version_id: str) -> bool:
        """
        Promote model to production
        
        Args:
            version_id: Model version ID
        
        Returns:
            Success status
        """
        # Find model
        model_info = self._find_model(version_id)
        if not model_info:
            logger.error(f"Model {version_id} not found")
            return False
        
        # Archive current production model
        if self.index['production']:
            self._archive_model(self.index['production'])
        
        # Promote new model
        self.index['production'] = version_id
        self._update_model_status(version_id, 'production')
        
        self._save_index()
        logger.info(f"Promoted {version_id} to production")
        return True
    
    def promote_to_staging(self, version_id: str) -> bool:
        """Promote model to staging"""
        self.index['staging'] = version_id
        self._update_model_status(version_id, 'staging')
        self._save_index()
        logger.info(f"Promoted {version_id} to staging")
        return True
    
    def _archive_model(self, version_id: str):
        """Archive a model"""
        model_info = self._find_model(version_id)
        if not model_info:
            return
        
        # Move to archive
        self.index['archive'].append({
            'version_id': version_id,
            'archived_at': datetime.now().isoformat()
        })
        
        # Update status
        self._update_model_status(version_id, 'archived')
        
        # Move files to backup
        src_model = self.models_path / f"{version_id}.joblib"
        dst_model = self.backup_path / f"{version_id}.joblib"
        if src_model.exists():
            shutil.copy2(src_model, dst_model)
        
        src_meta = self.metadata_path / f"{version_id}.json"
        dst_meta = self.backup_path / f"{version_id}.json"
        if src_meta.exists():
            shutil.copy2(src_meta, dst_meta)
        
        logger.info(f"Archived model {version_id}")
    
    def _find_model(self, version_id: str) -> Optional[Dict]:
        """Find model in registry"""
        for name, versions in self.index['models'].items():
            for v in versions:
                if v['version_id'] == version_id:
                    return v
        return None
    
    def _update_model_status(self, version_id: str, status: str):
        """Update model status"""
        for name, versions in self.index['models'].items():
            for v in versions:
                if v['version_id'] == version_id:
                    v['status'] = status
                    break
    
    def get_best_model(self, metric: str = 'accuracy') -> Optional[Dict]:
        """Get best performing model by metric"""
        best_score = -1
        best_model = None
        
        for name, versions in self.index['models'].items():
            for v in versions:
                score = v['metrics'].get(metric, 0)
                if score > best_score:
                    best_score = score
                    best_model = self.load_model(v['version_id'])
        
        return best_model
    
    def load_model(self, version_id: str) -> Optional[Dict]:
        """Load model by version ID"""
        model_path = self.models_path / f"{version_id}.joblib"
        scaler_path = self.models_path / f"{version_id}_scaler.joblib"
        metadata_path = self.metadata_path / f"{version_id}.json"
        
        if not model_path.exists():
            logger.error(f"Model {version_id} not found")
            return None
        
        # Load model
        model = joblib.load(model_path)
        
        # Load scaler
        scaler = None
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        
        # Load metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return {
            'model': model,
            'scaler': scaler,
            'metadata': metadata
        }
    
    def get_model_history(self, name: str) -> List[Dict]:
        """Get history of model versions"""
        return self.index['models'].get(name, [])
    
    def compare_models(self, version_id_1: str, version_id_2: str) -> Dict:
        """Compare two model versions"""
        model1 = self.load_model(version_id_1)
        model2 = self.load_model(version_id_2)
        
        if not model1 or not model2:
            return {}
        
        comparison = {
            'version_1': version_id_1,
            'version_2': version_id_2,
            'metrics_delta': {}
        }
        
        # Compare metrics
        metrics1 = model1['metadata'].get('metrics', {})
        metrics2 = model2['metadata'].get('metrics', {})
        
        for metric in set(metrics1.keys()) | set(metrics2.keys()):
            val1 = metrics1.get(metric, 0)
            val2 = metrics2.get(metric, 0)
            comparison['metrics_delta'][metric] = val2 - val1
        
        return comparison
    
    def rollback(self, name: str, version_id: Optional[str] = None) -> bool:
        """Rollback to previous version"""
        history = self.get_model_history(name)
        
        if len(history) < 2:
            logger.error(f"Not enough history to rollback {name}")
            return False
        
        if version_id:
            # Rollback to specific version
            target = version_id
        else:
            # Rollback to previous version
            target = history[-2]['version_id']
        
        # Promote target to production
        return self.promote_to_production(target)
    
    def cleanup_old_versions(self, keep_last: int = 5):
        """Cleanup old model versions"""
        for name, versions in self.index['models'].items():
            if len(versions) > keep_last:
                to_remove = versions[:-keep_last]
                for v in to_remove:
                    if v['status'] != 'production':
                        self._archive_model(v['version_id'])
    
    def get_registry_summary(self) -> Dict:
        """Get registry summary"""
        return {
            'total_models': sum(len(v) for v in self.index['models'].values()),
            'model_names': list(self.index['models'].keys()),
            'production_model': self.index['production'],
            'staging_model': self.index['staging'],
            'archived_count': len(self.index['archive'])
        }