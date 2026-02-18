"""
Recovery Manager - Automatic recovery from failures
"""
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import traceback
import signal
import os
import sys

logger = logging.getLogger(__name__)


class RecoveryManager:
    """
    Automatic recovery from system failures
    
    Features:
    - Crash detection
    - Automatic restart
    - Position recovery
    - Connection reestablishment
    - State consistency checks
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Recovery settings
        self.max_restarts = 5
        self.restart_window = 3600  # 1 hour
        self.cool_down_period = 60  # 1 minute
        
        # Failure tracking
        self.failure_count = 0
        self.failures = []
        self.last_restart = None
        
        # Component health
        self.component_health = {}
        
        logger.info("RecoveryManager initialized")
    
    async def start(self):
        """Start recovery manager"""
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("RecoveryManager started")
    
    async def monitor_component(self, component_name: str, health_check_func):
        """Monitor component health"""
        while True:
            try:
                health = await health_check_func()
                self.component_health[component_name] = {
                    'status': 'healthy' if health else 'unhealthy',
                    'last_check': datetime.now(),
                    'error': None
                }
                
                if not health:
                    await self.handle_failure(component_name, "Health check failed")
                
            except Exception as e:
                self.component_health[component_name] = {
                    'status': 'error',
                    'last_check': datetime.now(),
                    'error': str(e)
                }
                await self.handle_failure(component_name, str(e))
            
            await asyncio.sleep(60)  # Check every minute
    
    async def handle_failure(self, component: str, error: str):
        """Handle component failure"""
        self.logger.error(f"Failure detected in {component}: {error}")
        
        # Record failure
        self.failure_count += 1
        self.failures.append({
            'timestamp': datetime.now(),
            'component': component,
            'error': error,
            'traceback': traceback.format_exc()
        })
        
        # Check if too many failures
        if self._too_many_failures():
            self.logger.critical("Too many failures, initiating emergency shutdown")
            await self.emergency_shutdown()
            return
        
        # Attempt recovery based on component
        if component == 'data_engine':
            await self.recover_data_engine()
        elif component == 'execution_engine':
            await self.recover_execution_engine()
        elif component == 'strategy_engine':
            await self.recover_strategy_engine()
        elif component == 'ai_engine':
            await self.recover_ai_engine()
        else:
            await self.general_recovery()
    
    def _too_many_failures(self) -> bool:
        """Check if too many failures occurred"""
        # Count failures in last hour
        recent = [f for f in self.failures 
                 if f['timestamp'] > datetime.now() - timedelta(seconds=self.restart_window)]
        
        return len(recent) >= self.max_restarts
    
    async def recover_data_engine(self):
        """Recover data engine"""
        self.logger.info("Attempting to recover data engine...")
        
        try:
            # Reinitialize data connections
            from core.data_engine import DataEngine
            new_engine = DataEngine(self.config)
            await new_engine.initialize()
            
            # Swap instances
            self.component_health['data_engine']['instance'] = new_engine
            self.component_health['data_engine']['status'] = 'recovered'
            
            self.logger.info("Data engine recovered successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to recover data engine: {e}")
            await self.handle_failure('data_engine', f"Recovery failed: {e}")
    
    async def recover_execution_engine(self):
        """Recover execution engine"""
        self.logger.info("Attempting to recover execution engine...")
        
        try:
            from core.execution_engine import ExecutionEngine
            new_engine = ExecutionEngine(self.config)
            await new_engine.initialize()
            
            self.component_health['execution_engine']['instance'] = new_engine
            self.component_health['execution_engine']['status'] = 'recovered'
            
            self.logger.info("Execution engine recovered successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to recover execution engine: {e}")
            await self.handle_failure('execution_engine', f"Recovery failed: {e}")
    
    async def recover_strategy_engine(self):
        """Recover strategy engine"""
        self.logger.info("Attempting to recover strategy engine...")
        
        try:
            from core.strategy_engine import StrategyEngine
            new_engine = StrategyEngine(self.config)
            await new_engine.initialize()
            
            self.component_health['strategy_engine']['instance'] = new_engine
            self.component_health['strategy_engine']['status'] = 'recovered'
            
            self.logger.info("Strategy engine recovered successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to recover strategy engine: {e}")
            await self.handle_failure('strategy_engine', f"Recovery failed: {e}")
    
    async def recover_ai_engine(self):
        """Recover AI engine"""
        self.logger.info("Attempting to recover AI engine...")
        
        try:
            from core.ai_engine import AIEngine
            new_engine = AIEngine(self.config)
            await new_engine.initialize()
            
            self.component_health['ai_engine']['instance'] = new_engine
            self.component_health['ai_engine']['status'] = 'recovered'
            
            self.logger.info("AI engine recovered successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to recover AI engine: {e}")
            await self.handle_failure('ai_engine', f"Recovery failed: {e}")
    
    async def general_recovery(self):
        """General recovery procedure"""
        self.logger.info("Attempting general recovery...")
        
        # Wait for cool-down
        await asyncio.sleep(self.cool_down_period)
        
        # Try to restart the bot
        await self.restart_bot()
    
    async def restart_bot(self):
        """Restart the entire bot"""
        self.logger.warning("Restarting bot...")
        
        self.last_restart = datetime.now()
        
        # Save state before restart
        if 'state_manager' in self.component_health:
            await self.component_health['state_manager']['instance'].save_state()
        
        # Restart the process
        os.execv(sys.executable, ['python'] + sys.argv)
    
    async def emergency_shutdown(self):
        """Emergency shutdown procedure"""
        self.logger.critical("EMERGENCY SHUTDOWN INITIATED")
        
        # Close all positions
        if 'execution_engine' in self.component_health:
            engine = self.component_health['execution_engine']['instance']
            await engine.close_all_positions()
        
        # Save final state
        if 'state_manager' in self.component_health:
            await self.component_health['state_manager']['instance'].save_checkpoint('emergency')
        
        # Shutdown
        sys.exit(1)
    
    def _signal_handler(self, signum, frame):
        """Handle system signals"""
        self.logger.info(f"Received signal {signum}")
        
        if signum in [signal.SIGINT, signal.SIGTERM]:
            asyncio.create_task(self.graceful_shutdown())
    
    async def graceful_shutdown(self):
        """Graceful shutdown on signal"""
        self.logger.info("Graceful shutdown initiated")
        
        # Close all positions if configured
        if self.config['trading'].get('close_on_shutdown', False):
            if 'execution_engine' in self.component_health:
                await self.component_health['execution_engine']['instance'].close_all_positions()
        
        # Save state
        if 'state_manager' in self.component_health:
            await self.component_health['state_manager']['instance'].save_state()
        
        self.logger.info("Shutdown complete")
        sys.exit(0)
    
    def get_health_report(self) -> Dict:
        """Get comprehensive health report"""
        return {
            'failure_count': self.failure_count,
            'recent_failures': len([f for f in self.failures 
                                   if f['timestamp'] > datetime.now() - timedelta(hours=1)]),
            'component_health': self.component_health,
            'last_restart': self.last_restart.isoformat() if self.last_restart else None,
            'status': 'healthy' if self.failure_count < self.max_restarts else 'critical'
        }