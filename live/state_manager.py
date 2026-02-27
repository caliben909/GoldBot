"""
State Manager v2.0 - Production-Ready Implementation
Persistent state management with crash recovery, validation, and journaling
Optimized for institutional trading systems
"""

import json
import pickle
import gzip
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, is_dataclass
import logging
import threading
import time
from contextlib import contextmanager
import copy

logger = logging.getLogger(__name__)


@dataclass
class PositionState:
    """Serializable position state"""
    ticket: str
    symbol: str
    direction: str
    quantity: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    open_time: str
    strategy: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TradeJournalEntry:
    """Journal entry for complete audit trail"""
    timestamp: str
    action: str  # 'open', 'close', 'modify', 'cancel'
    ticket: str
    symbol: str
    direction: str
    quantity: float
    price: float
    pnl: Optional[float] = None
    pnl_pips: Optional[float] = None
    commission: float = 0.0
    swap: float = 0.0
    balance_before: float = 0.0
    balance_after: float = 0.0
    reason: str = ""
    strategy: str = ""


@dataclass
class SystemState:
    """System state metadata"""
    version: str = "2.0.0"
    last_start: Optional[str] = None
    last_shutdown: Optional[str] = None
    last_save: Optional[str] = None
    restart_count: int = 0
    save_count: int = 0
    checksum: Optional[str] = None


class StateValidator:
    """State validation and checksum verification"""
    
    @staticmethod
    def calculate_checksum(data: Dict) -> str:
        """Calculate SHA-256 checksum of state"""
        # Create canonical JSON string (sorted keys)
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
    
    @staticmethod
    def validate_state(state: Dict) -> Tuple[bool, str]:
        """Validate state integrity"""
        required_keys = ['positions', 'orders', 'equity_history', 'metrics', 'system', 'journal']
        
        # Check required keys
        for key in required_keys:
            if key not in state:
                return False, f"Missing required key: {key}"
        
        # Validate system state
        system = state.get('system', {})
        if 'checksum' in system and system['checksum']:
            # Verify checksum
            test_state = copy.deepcopy(state)
            test_state['system'] = {k: v for k, v in system.items() if k != 'checksum'}
            expected_checksum = StateValidator.calculate_checksum(test_state)
            
            if system['checksum'] != expected_checksum:
                return False, f"Checksum mismatch: expected {expected_checksum}, got {system['checksum']}"
        
        # Validate positions
        positions = state.get('positions', {})
        for ticket, pos in positions.items():
            required_pos_keys = ['symbol', 'direction', 'quantity', 'entry_price']
            for key in required_pos_keys:
                if key not in pos:
                    return False, f"Position {ticket} missing {key}"
        
        return True, "State valid"


class StateManager:
    """
    Production-ready state manager with:
    - Synchronous file I/O (faster for small files)
    - Automatic backup rotation
    - State validation and recovery
    - Complete trade journaling
    - Memory-efficient equity tracking
    - Thread-safe operations
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Directories
        self.state_dir = Path(self.config.get('state_dir', 'state'))
        self.backup_dir = Path(self.config.get('backup_dir', 'state/backups'))
        self.journal_dir = Path(self.config.get('journal_dir', 'state/journal'))
        
        # Create directories
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.state_file = self.state_dir / 'trading_state.json'
        self.journal_file = self.journal_dir / f"journal_{datetime.now().strftime('%Y%m')}.jsonl"
        self.equity_file = self.state_dir / 'equity.csv'
        
        # State
        self.state: Dict[str, Any] = {
            'positions': {},
            'orders': {},
            'equity_history': [],  # Last 100 points in memory
            'metrics': {},
            'system': asdict(SystemState()),
            'journal_cursor': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        self._running = False
        self._save_thread: Optional[threading.Thread] = None
        
        # Auto-save settings
        self.auto_save_interval = self.config.get('auto_save_interval', 60)  # 1 minute
        self.max_equity_memory = self.config.get('max_equity_memory', 100)
        self.max_backups = self.config.get('max_backups', 20)
        
        # Callbacks
        self.save_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        
        # Performance tracking
        self._last_save_time = 0
        self._save_count = 0
        
        logger.info(f"StateManager initialized (dir={self.state_dir})")
    
    def register_save_callback(self, callback: Callable):
        """Register callback to be called after save"""
        self.save_callbacks.append(callback)
    
    def register_recovery_callback(self, callback: Callable):
        """Register callback to be called after recovery"""
        self.recovery_callbacks.append(callback)
    
    def start(self):
        """Start state manager"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            
            # Load previous state
            self._load_state()
            
            # Update system state
            self.state['system']['last_start'] = datetime.now().isoformat()
            self.state['system']['restart_count'] = self.state['system'].get('restart_count', 0) + 1
            
            # Start auto-save thread
            self._save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
            self._save_thread.start()
            
            # Rotate journal file if needed
            self._rotate_journal()
            
            logger.info("StateManager started")
    
    def stop(self):
        """Stop state manager"""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            
            # Final save
            self._save_state(force=True)
            
            # Update shutdown time
            self.state['system']['last_shutdown'] = datetime.now().isoformat()
            self._save_state(force=True)
            
            logger.info("StateManager stopped")
    
    def _auto_save_loop(self):
        """Background auto-save thread"""
        while self._running:
            time.sleep(self.auto_save_interval)
            if self._running:
                try:
                    self._save_state()
                except Exception as e:
                    logger.error(f"Auto-save failed: {e}")
    
    def _load_state(self) -> bool:
        """Load state from disk with recovery"""
        try:
            # Try primary state file
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    loaded_state = json.load(f)
                
                # Validate
                is_valid, message = StateValidator.validate_state(loaded_state)
                
                if is_valid:
                    self.state = loaded_state
                    logger.info("State loaded successfully")
                    
                    # Trigger recovery callbacks
                    for callback in self.recovery_callbacks:
                        try:
                            callback(self.state)
                        except Exception as e:
                            logger.error(f"Recovery callback error: {e}")
                    
                    return True
                else:
                    logger.warning(f"State validation failed: {message}")
                    return self._recover_from_backup()
            
            # Try to recover from backup
            return self._recover_from_backup()
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return self._recover_from_backup()
    
    def _recover_from_backup(self) -> bool:
        """Attempt recovery from backup files"""
        try:
            backups = sorted(self.backup_dir.glob('state_*.json.gz'), reverse=True)
            
            for backup_file in backups[:5]:  # Try last 5 backups
                try:
                    with gzip.open(backup_file, 'rt') as f:
                        loaded_state = json.load(f)
                    
                    is_valid, message = StateValidator.validate_state(loaded_state)
                    
                    if is_valid:
                        self.state = loaded_state
                        logger.info(f"State recovered from backup: {backup_file}")
                        
                        # Trigger recovery callbacks
                        for callback in self.recovery_callbacks:
                            try:
                                callback(self.state)
                            except Exception as e:
                                logger.error(f"Recovery callback error: {e}")
                        
                        return True
                    
                except Exception as e:
                    logger.warning(f"Failed to load backup {backup_file}: {e}")
                    continue
            
            logger.error("No valid backup found")
            return False
            
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False
    
    def _save_state(self, force: bool = False):
        """Save state to disk"""
        current_time = time.time()
        
        # Throttle saves (max 1 per second unless forced)
        if not force and current_time - self._last_save_time < 1:
            return
        
        with self._lock:
            try:
                # Calculate checksum
                state_to_save = copy.deepcopy(self.state)
                state_to_save['system']['checksum'] = None
                checksum = StateValidator.calculate_checksum(state_to_save)
                state_to_save['system']['checksum'] = checksum
                state_to_save['system']['last_save'] = datetime.now().isoformat()
                
                # Write to temp file first (atomic write)
                temp_file = self.state_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(state_to_save, f, indent=2, default=str)
                
                # Atomic rename
                temp_file.replace(self.state_file)
                
                # Update tracking
                self._last_save_time = current_time
                self._save_count += 1
                self.state['system']['save_count'] = self._save_count
                
                # Create backup periodically (every 10 saves)
                if self._save_count % 10 == 0:
                    self._create_backup()
                
                # Trigger callbacks
                for callback in self.save_callbacks:
                    try:
                        callback(state_to_save)
                    except Exception as e:
                        logger.error(f"Save callback error: {e}")
                
                logger.debug(f"State saved (count={self._save_count})")
                
            except Exception as e:
                logger.error(f"Failed to save state: {e}")
    
    def _create_backup(self):
        """Create compressed backup"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.backup_dir / f"state_{timestamp}.json.gz"
            
            # Compress and save
            with gzip.open(backup_file, 'wt') as f:
                json.dump(self.state, f, default=str)
            
            # Clean old backups
            backups = sorted(self.backup_dir.glob('state_*.json.gz'))
            if len(backups) > self.max_backups:
                for old_backup in backups[:-self.max_backups]:
                    old_backup.unlink()
                    logger.debug(f"Removed old backup: {old_backup}")
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
    
    def _rotate_journal(self):
        """Rotate journal file monthly"""
        current_month = datetime.now().strftime('%Y%m')
        self.journal_file = self.journal_dir / f"journal_{current_month}.jsonl"
    
    # ========================================================================
    # POSITION MANAGEMENT
    # ========================================================================
    
    def add_position(self, position: Union[PositionState, Dict]):
        """Add or update position"""
        with self._lock:
            if isinstance(position, dict):
                position = PositionState(**position)
            
            self.state['positions'][position.ticket] = position.to_dict()
            logger.debug(f"Position added: {position.ticket}")
    
    def update_position(self, ticket: str, updates: Dict):
        """Update position fields"""
        with self._lock:
            if ticket in self.state['positions']:
                self.state['positions'][ticket].update(updates)
                self.state['positions'][ticket]['last_update'] = datetime.now().isoformat()
    
    def remove_position(self, ticket: str, pnl: float = 0, reason: str = ""):
        """Remove position and journal the closure"""
        with self._lock:
            if ticket in self.state['positions']:
                pos = self.state['positions'][ticket]
                
                # Journal the close
                self._journal_trade(
                    action='close',
                    ticket=ticket,
                    symbol=pos['symbol'],
                    direction=pos['direction'],
                    quantity=pos['quantity'],
                    price=pos.get('current_price', pos['entry_price']),
                    pnl=pnl,
                    reason=reason,
                    strategy=pos.get('strategy', '')
                )
                
                # Remove from active positions
                del self.state['positions'][ticket]
                logger.debug(f"Position removed: {ticket}")
    
    def get_position(self, ticket: str) -> Optional[Dict]:
        """Get position by ticket"""
        return self.state['positions'].get(ticket)
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all positions"""
        return copy.deepcopy(self.state['positions'])
    
    def get_positions_by_symbol(self, symbol: str) -> List[Dict]:
        """Get positions for a symbol"""
        return [
            pos for pos in self.state['positions'].values()
            if pos['symbol'] == symbol
        ]
    
    # ========================================================================
    # ORDER MANAGEMENT
    # ========================================================================
    
    def add_order(self, order_id: str, order_data: Dict):
        """Add pending order"""
        with self._lock:
            self.state['orders'][order_id] = {
                **order_data,
                'created_at': datetime.now().isoformat()
            }
    
    def update_order(self, order_id: str, updates: Dict):
        """Update order"""
        with self._lock:
            if order_id in self.state['orders']:
                self.state['orders'][order_id].update(updates)
    
    def remove_order(self, order_id: str, reason: str = ""):
        """Remove order"""
        with self._lock:
            if order_id in self.state['orders']:
                order = self.state['orders'][order_id]
                
                # Journal cancellation
                self._journal_trade(
                    action='cancel',
                    ticket=order_id,
                    symbol=order.get('symbol', ''),
                    direction=order.get('direction', ''),
                    quantity=order.get('quantity', 0),
                    price=order.get('price', 0),
                    reason=reason
                )
                
                del self.state['orders'][order_id]
    
    def get_order(self, order_id: str) -> Optional[Dict]:
        """Get order by ID"""
        return self.state['orders'].get(order_id)
    
    # ========================================================================
    # EQUITY TRACKING
    # ========================================================================
    
    def add_equity_point(self, equity: float, balance: Optional[float] = None,
                        open_pnl: Optional[float] = None):
        """Add equity curve point (memory-efficient)"""
        with self._lock:
            point = {
                'timestamp': datetime.now().isoformat(),
                'equity': equity
            }
            
            if balance is not None:
                point['balance'] = balance
            if open_pnl is not None:
                point['open_pnl'] = open_pnl
            
            self.state['equity_history'].append(point)
            
            # Trim memory cache
            if len(self.state['equity_history']) > self.max_equity_memory:
                self._flush_equity_to_disk()
                self.state['equity_history'] = self.state['equity_history'][-self.max_equity_memory:]
    
    def _flush_equity_to_disk(self):
        """Flush equity history to CSV for long-term storage"""
        try:
            # Append to CSV
            mode = 'a' if self.equity_file.exists() else 'w'
            
            with open(self.equity_file, mode) as f:
                if mode == 'w':
                    f.write("timestamp,equity,balance,open_pnl\n")
                
                for point in self.state['equity_history'][:-self.max_equity_memory]:
                    line = f"{point['timestamp']},{point.get('equity',0)},{point.get('balance',0)},{point.get('open_pnl',0)}\n"
                    f.write(line)
            
        except Exception as e:
            logger.error(f"Equity flush failed: {e}")
    
    def get_equity_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get equity history"""
        history = self.state['equity_history']
        if limit:
            history = history[-limit:]
        return copy.deepcopy(history)
    
    # ========================================================================
    # TRADE JOURNALING
    # ========================================================================
    
    def _journal_trade(self, **kwargs):
        """Write to trade journal"""
        try:
            entry = TradeJournalEntry(
                timestamp=datetime.now().isoformat(),
                **kwargs
            )
            
            # Append to journal file
            with open(self.journal_file, 'a') as f:
                f.write(json.dumps(asdict(entry), default=str) + '\n')
            
        except Exception as e:
            logger.error(f"Journal write failed: {e}")
    
    def journal_entry(self, action: str, ticket: str, symbol: str,
                     direction: str, quantity: float, price: float,
                     **kwargs):
        """Manual journal entry"""
        self._journal_trade(
            action=action,
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            price=price,
            **kwargs
        )
    
    def get_journal(self, start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None,
                   symbol: Optional[str] = None) -> List[Dict]:
        """Read trade journal with filtering"""
        entries = []
        
        # Find relevant journal files
        journal_files = sorted(self.journal_dir.glob('journal_*.jsonl'))
        
        for journal_file in journal_files:
            try:
                with open(journal_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        
                        # Apply filters
                        entry_time = datetime.fromisoformat(entry['timestamp'])
                        
                        if start_date and entry_time < start_date:
                            continue
                        if end_date and entry_time > end_date:
                            continue
                        if symbol and entry.get('symbol') != symbol:
                            continue
                        
                        entries.append(entry)
                        
            except Exception as e:
                logger.warning(f"Error reading journal {journal_file}: {e}")
        
        return entries
    
    # ========================================================================
    # METRICS AND SYSTEM
    # ========================================================================
    
    def update_metrics(self, metrics: Dict):
        """Update performance metrics"""
        with self._lock:
            self.state['metrics'].update(metrics)
            self.state['metrics']['last_update'] = datetime.now().isoformat()
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        return copy.deepcopy(self.state['metrics'])
    
    def get_system_info(self) -> Dict:
        """Get system information"""
        info = copy.deepcopy(self.state['system'])
        info['uptime'] = self._calculate_uptime()
        info['positions_count'] = len(self.state['positions'])
        info['orders_count'] = len(self.state['orders'])
        return info
    
    def _calculate_uptime(self) -> str:
        """Calculate system uptime"""
        last_start = self.state['system'].get('last_start')
        if not last_start:
            return "0s"
        
        start = datetime.fromisoformat(last_start)
        uptime = datetime.now() - start
        
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m {seconds}s"
    
    # ========================================================================
    # CHECKPOINTS
    # ========================================================================
    
    def create_checkpoint(self, name: str) -> Path:
        """Create manual checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_file = self.backup_dir / f"checkpoint_{name}_{timestamp}.json.gz"
        
        with self._lock:
            with gzip.open(checkpoint_file, 'wt') as f:
                json.dump(self.state, f, default=str)
        
        logger.info(f"Checkpoint created: {checkpoint_file}")
        return checkpoint_file
    
    def list_checkpoints(self) -> List[Path]:
        """List available checkpoints"""
        return sorted(self.backup_dir.glob('checkpoint_*.json.gz'))
    
    def restore_checkpoint(self, checkpoint_file: Union[str, Path]) -> bool:
        """Restore from checkpoint"""
        checkpoint_path = Path(checkpoint_file)
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            with gzip.open(checkpoint_path, 'rt') as f:
                loaded_state = json.load(f)
            
            is_valid, message = StateValidator.validate_state(loaded_state)
            
            if is_valid:
                with self._lock:
                    self.state = loaded_state
                    self._save_state(force=True)
                
                logger.info(f"Restored from checkpoint: {checkpoint_path}")
                return True
            else:
                logger.error(f"Checkpoint validation failed: {message}")
                return False
                
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def get_state_summary(self) -> Dict:
        """Get human-readable state summary"""
        return {
            'positions': {
                'count': len(self.state['positions']),
                'symbols': list(set(p['symbol'] for p in self.state['positions'].values())),
                'total_volume': sum(p['quantity'] for p in self.state['positions'].values())
            },
            'orders': {
                'count': len(self.state['orders']),
                'pending': len(self.state['orders'])
            },
            'equity_points_in_memory': len(self.state['equity_history']),
            'system': self.get_system_info(),
            'last_save_ago': time.time() - self._last_save_time if self._last_save_time > 0 else None
        }


# ============================================================================
# CONTEXT MANAGER
# ============================================================================

@contextmanager
def managed_state(config: Optional[Dict] = None):
    """Context manager for state management"""
    manager = StateManager(config)
    try:
        manager.start()
        yield manager
    finally:
        manager.stop()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Configuration
    config = {
        'state_dir': 'state',
        'auto_save_interval': 30,
        'max_equity_memory': 50
    }
    
    # Using context manager
    with managed_state(config) as state:
        print("State Manager Demo")
        print("=" * 50)
        
        # Add positions
        state.add_position(PositionState(
            ticket="12345",
            symbol="EURUSD",
            direction="long",
            quantity=0.5,
            entry_price=1.0850,
            current_price=1.0860,
            stop_loss=1.0800,
            take_profit=1.0950,
            open_time=datetime.now().isoformat(),
            strategy="SMC_Contrarian",
            metadata={'confidence': 0.85}
        ))
        
        # Add equity points
        for i in range(10):
            state.add_equity_point(
                equity=10000 + i * 10,
                balance=10000,
                open_pnl=i * 10
            )
        
        # Update metrics
        state.update_metrics({
            'win_rate': 0.72,
            'profit_factor': 2.5,
            'sharpe_ratio': 1.8
        })
        
        # Print summary
        summary = state.get_state_summary()
        print(f"\nPositions: {summary['positions']['count']}")
        print(f"Equity Points: {summary['equity_points_in_memory']}")
        print(f"Uptime: {summary['system']['uptime']}")
        
        # Create checkpoint
        checkpoint = state.create_checkpoint("demo")
        print(f"\nCheckpoint created: {checkpoint}")
        
        # Simulate position close
        state.remove_position("12345", pnl=50.0, reason="TP hit")
        
        print("\nFinal state saved")