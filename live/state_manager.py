"""
State Manager - Persistent state management for trading bot
"""
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import asyncio
import aiofiles
import hashlib

logger = logging.getLogger(__name__)


class StateManager:
    """
    Persistent state management with backup and recovery
    
    Features:
    - Automatic state saving
    - Version control
    - Backup rotation
    - Crash recovery
    - State validation
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.state_dir = Path('state')
        self.backup_dir = Path('state/backups')
        
        self.state_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        
        # State files
        self.positions_file = self.state_dir / 'positions.json'
        self.orders_file = self.state_dir / 'orders.json'
        self.equity_file = self.state_dir / 'equity.json'
        self.metrics_file = self.state_dir / 'metrics.json'
        self.system_file = self.state_dir / 'system.json'
        
        # Current state
        self.state = {
            'positions': {},
            'orders': {},
            'equity_history': [],
            'metrics': {},
            'system': {
                'last_start': None,
                'last_shutdown': None,
                'restart_count': 0,
                'version': '2.0.0'
            }
        }
        
        # Auto-save task
        self.auto_save_interval = 300  # 5 minutes
        self._save_task = None
        self._running = False
        
        logger.info("StateManager initialized")
    
    async def start(self):
        """Start state manager"""
        self._running = True
        self._save_task = asyncio.create_task(self._auto_save())
        
        # Load previous state
        await self.load_state()
        
        # Update system state
        self.state['system']['last_start'] = datetime.now().isoformat()
        self.state['system']['restart_count'] += 1
        
        logger.info("StateManager started")
    
    async def load_state(self):
        """Load state from disk"""
        try:
            # Load positions
            if self.positions_file.exists():
                async with aiofiles.open(self.positions_file, 'r') as f:
                    content = await f.read()
                    self.state['positions'] = json.loads(content)
            
            # Load orders
            if self.orders_file.exists():
                async with aiofiles.open(self.orders_file, 'r') as f:
                    content = await f.read()
                    self.state['orders'] = json.loads(content)
            
            # Load equity history
            if self.equity_file.exists():
                async with aiofiles.open(self.equity_file, 'r') as f:
                    content = await f.read()
                    self.state['equity_history'] = json.loads(content)
            
            # Load metrics
            if self.metrics_file.exists():
                async with aiofiles.open(self.metrics_file, 'r') as f:
                    content = await f.read()
                    self.state['metrics'] = json.loads(content)
            
            # Load system state
            if self.system_file.exists():
                async with aiofiles.open(self.system_file, 'r') as f:
                    content = await f.read()
                    self.state['system'].update(json.loads(content))
            
            logger.info("State loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    async def save_state(self):
        """Save current state to disk"""
        try:
            # Create backup if needed
            await self._create_backup()
            
            # Save positions
            async with aiofiles.open(self.positions_file, 'w') as f:
                await f.write(json.dumps(self.state['positions'], indent=2, default=str))
            
            # Save orders
            async with aiofiles.open(self.orders_file, 'w') as f:
                await f.write(json.dumps(self.state['orders'], indent=2, default=str))
            
            # Save equity history (keep last 1000 points)
            equity_to_save = self.state['equity_history'][-1000:]
            async with aiofiles.open(self.equity_file, 'w') as f:
                await f.write(json.dumps(equity_to_save, indent=2, default=str))
            
            # Save metrics
            async with aiofiles.open(self.metrics_file, 'w') as f:
                await f.write(json.dumps(self.state['metrics'], indent=2, default=str))
            
            # Save system state
            async with aiofiles.open(self.system_file, 'w') as f:
                await f.write(json.dumps(self.state['system'], indent=2, default=str))
            
            logger.debug("State saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    async def _auto_save(self):
        """Auto-save state periodically"""
        while self._running:
            await asyncio.sleep(self.auto_save_interval)
            await self.save_state()
    
    async def _create_backup(self):
        """Create backup of current state"""
        try:
            backup_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.backup_dir / f"state_backup_{backup_time}.json"
            
            # Create backup
            async with aiofiles.open(backup_file, 'w') as f:
                await f.write(json.dumps(self.state, indent=2, default=str))
            
            # Clean old backups (keep last 10)
            backups = sorted(self.backup_dir.glob('state_backup_*.json'))
            if len(backups) > 10:
                for old_backup in backups[:-10]:
                    old_backup.unlink()
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
    
    async def update_positions(self, positions: List[Dict]):
        """Update positions state"""
        positions_dict = {}
        for pos in positions:
            positions_dict[pos['symbol']] = {
                **pos,
                'timestamp': datetime.now().isoformat()
            }
        
        self.state['positions'] = positions_dict
    
    async def update_orders(self, orders: List[Dict]):
        """Update orders state"""
        orders_dict = {}
        for order in orders:
            orders_dict[order['id']] = {
                **order,
                'timestamp': datetime.now().isoformat()
            }
        
        self.state['orders'] = orders_dict
    
    async def add_equity_point(self, equity: float):
        """Add equity curve point"""
        self.state['equity_history'].append({
            'timestamp': datetime.now().isoformat(),
            'equity': equity
        })
        
        # Keep only last 10000 points
        if len(self.state['equity_history']) > 10000:
            self.state['equity_history'] = self.state['equity_history'][-10000:]
    
    async def update_metrics(self, metrics: Dict):
        """Update performance metrics"""
        self.state['metrics'] = {
            **self.state['metrics'],
            **metrics,
            'last_update': datetime.now().isoformat()
        }
    
    def get_positions(self) -> Dict:
        """Get current positions"""
        return self.state['positions']
    
    def get_orders(self) -> Dict:
        """Get current orders"""
        return self.state['orders']
    
    def get_equity_history(self) -> List:
        """Get equity history"""
        return self.state['equity_history']
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        return self.state['metrics']
    
    def get_system_info(self) -> Dict:
        """Get system information"""
        return {
            **self.state['system'],
            'uptime': self._get_uptime()
        }
    
    def _get_uptime(self) -> str:
        """Calculate uptime"""
        if not self.state['system']['last_start']:
            return "0s"
        
        start = datetime.fromisoformat(self.state['system']['last_start'])
        uptime = datetime.now() - start
        
        days = uptime.days
        hours = uptime.seconds // 3600
        minutes = (uptime.seconds % 3600) // 60
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    async def save_checkpoint(self, name: str):
        """Save manual checkpoint"""
        checkpoint_file = self.backup_dir / f"checkpoint_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        async with aiofiles.open(checkpoint_file, 'w') as f:
            await f.write(json.dumps(self.state, indent=2, default=str))
        
        logger.info(f"Checkpoint saved: {checkpoint_file}")
    
    async def restore_checkpoint(self, checkpoint_file: Path) -> bool:
        """Restore from checkpoint"""
        try:
            async with aiofiles.open(checkpoint_file, 'r') as f:
                content = await f.read()
                self.state = json.loads(content)
            
            await self.save_state()
            logger.info(f"Restored from checkpoint: {checkpoint_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return False
    
    async def shutdown(self):
        """Graceful shutdown"""
        self._running = False
        
        if self._save_task:
            self._save_task.cancel()
        
        # Final state save
        self.state['system']['last_shutdown'] = datetime.now().isoformat()
        await self.save_state()
        
        logger.info("StateManager shutdown complete")