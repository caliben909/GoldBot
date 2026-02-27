"""
Recovery Manager - Institutional-Grade Failure Recovery System
Implements circuit breakers, graceful degradation, and guaranteed state persistence
"""

import asyncio
import signal
import os
import sys
import json
import pickle
import logging
import traceback
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from pathlib import Path
import aiofiles
import functools

logger = logging.getLogger(__name__)


class ComponentState(Enum):
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    RECOVERING = auto()
    FAILED = auto()
    SHUTDOWN = auto()


class RecoveryAction(Enum):
    RESTART = auto()
    RECONNECT = auto()
    RESET_STATE = auto()
    FAILOVER = auto()
    GRACEFUL_DEGRADE = auto()
    EMERGENCY_SHUTDOWN = auto()


@dataclass
class FailureEvent:
    """Immutable failure record"""
    timestamp: datetime
    component: str
    error_type: str
    error_message: str
    traceback: str
    recovery_attempts: int = 0
    resolved: bool = False


@dataclass
class ComponentConfig:
    """Per-component recovery configuration"""
    name: str
    max_recovery_attempts: int = 3
    recovery_timeout: float = 30.0
    cooldown_seconds: float = 5.0
    critical: bool = False  # If True, failure triggers emergency shutdown
    dependencies: List[str] = field(default_factory=list)
    recovery_action: RecoveryAction = RecoveryAction.RESTART
    health_check_interval: float = 30.0


class CircuitBreaker:
    """
    Prevents cascading failures by blocking recovery attempts after threshold
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures: List[datetime] = []
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time: Optional[datetime] = None
        self._lock = asyncio.Lock()
    
    async def record_failure(self) -> bool:
        """Record failure and check if circuit should open"""
        async with self._lock:
            now = datetime.now()
            self.failures.append(now)
            self.last_failure_time = now
            
            # Remove old failures outside window
            window_start = now - timedelta(seconds=self.recovery_timeout)
            self.failures = [f for f in self.failures if f > window_start]
            
            if len(self.failures) >= self.failure_threshold:
                if self.state == "CLOSED":
                    self.state = "OPEN"
                    logger.critical(f"Circuit breaker OPENED after {len(self.failures)} failures")
                    return False
                return False
            
            return True
    
    async def can_execute(self) -> bool:
        """Check if operation allowed"""
        async with self._lock:
            if self.state == "CLOSED":
                return True
            
            if self.state == "OPEN":
                # Check if timeout elapsed
                if self.last_failure_time:
                    elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                    if elapsed > self.recovery_timeout:
                        self.state = "HALF_OPEN"
                        logger.info("Circuit breaker entering HALF_OPEN state")
                        return True
                return False
            
            return True  # HALF_OPEN allows one test
    
    async def record_success(self):
        """Record successful operation, close circuit if half-open"""
        async with self._lock:
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failures = []
                logger.info("Circuit breaker CLOSED - recovery successful")


class StatePersistence:
    """
    Guaranteed state persistence with atomic writes and verification
    """
    
    def __init__(self, state_dir: str = "state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        self._lock = asyncio.Lock()
    
    async def save(self, name: str, state: Dict) -> Path:
        """Atomic state save with backup"""
        async with self._lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.state"
            filepath = self.state_dir / filename
            temp_path = filepath.with_suffix('.tmp')
            
            try:
                # Serialize
                data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Write to temp file atomically
                async with aiofiles.open(temp_path, 'wb') as f:
                    await f.write(data)
                
                # Atomic rename
                temp_path.rename(filepath)
                
                # Verify by reading back
                async with aiofiles.open(filepath, 'rb') as f:
                    verify_data = await f.read()
                    pickle.loads(verify_data)  # Verify deserialization
                
                # Cleanup old states (keep last 10)
                await self._cleanup_old_states(name)
                
                logger.info(f"State saved: {filepath}")
                return filepath
                
            except Exception as e:
                logger.error(f"State save failed: {e}")
                if temp_path.exists():
                    temp_path.unlink()
                raise
    
    async def load_latest(self, name: str) -> Optional[Dict]:
        """Load most recent valid state"""
        try:
            pattern = f"{name}_*.state"
            states = sorted(self.state_dir.glob(pattern), reverse=True)
            
            for state_file in states[:3]:  # Try last 3
                try:
                    async with aiofiles.open(state_file, 'rb') as f:
                        data = await f.read()
                        state = pickle.loads(data)
                        logger.info(f"State loaded from {state_file}")
                        return state
                except Exception as e:
                    logger.warning(f"Failed to load {state_file}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"State load error: {e}")
            return None
    
    async def _cleanup_old_states(self, name: str, keep: int = 10):
        """Remove old state files"""
        pattern = f"{name}_*.state"
        states = sorted(self.state_dir.glob(pattern), reverse=True)
        
        for old_file in states[keep:]:
            try:
                old_file.unlink()
            except Exception:
                pass


class RecoveryManager:
    """
    Production-grade recovery manager with circuit breakers, dependency management,
    and guaranteed state persistence
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.components: Dict[str, Any] = {}
        self.component_configs: Dict[str, ComponentConfig] = {}
        self.component_states: Dict[str, ComponentState] = {}
        self.health_checks: Dict[str, Callable] = {}
        
        # Failure tracking
        self.failure_history: List[FailureEvent] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # State persistence
        self.state_persistence = StatePersistence(
            config.get('state_dir', 'state')
        )
        
        # Recovery tracking
        self.recovery_in_progress: Set[str] = set()
        self.global_failure_count = 0
        self.last_global_restart: Optional[datetime] = None
        
        # Shutdown
        self._shutdown_event = asyncio.Event()
        self._monitor_tasks: List[asyncio.Task] = []
        
        logger.info("RecoveryManager initialized")
    
    def register_component(self, name: str, instance: Any, 
                          config: ComponentConfig,
                          health_check: Callable):
        """
        Register component for monitoring
        
        Args:
            name: Component identifier
            instance: Component instance
            config: Recovery configuration
            health_check: Async function returning bool
        """
        self.components[name] = instance
        self.component_configs[name] = config
        self.component_states[name] = ComponentState.HEALTHY
        self.health_checks[name] = health_check
        self.circuit_breakers[name] = CircuitBreaker()
        
        logger.info(f"Registered component: {name}")
    
    async def start(self):
        """Start recovery monitoring"""
        # Use signal handlers compatible with asyncio
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, 
                    functools.partial(self._signal_handler, sig))
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            logger.warning("Async signal handlers not supported, using fallback")
        
        # Start health monitoring for each component
        for name in self.components:
            task = asyncio.create_task(
                self._monitor_component(name),
                name=f"health_monitor_{name}"
            )
            self._monitor_tasks.append(task)
        
        # Start global watchdog
        watchdog_task = asyncio.create_task(
            self._global_watchdog(),
            name="global_watchdog"
        )
        self._monitor_tasks.append(watchdog_task)
        
        logger.info("RecoveryManager monitoring started")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("RecoveryManager shutting down...")
        self._shutdown_event.set()
        
        # Cancel all monitor tasks
        for task in self._monitor_tasks:
            task.cancel()
        
        if self._monitor_tasks:
            await asyncio.gather(*self._monitor_tasks, return_exceptions=True)
        
        # Final state save
        await self._emergency_save_state()
        
        logger.info("RecoveryManager shutdown complete")
    
    # ==================== MONITORING ====================
    
    async def _monitor_component(self, name: str):
        """Monitor single component health"""
        config = self.component_configs[name]
        
        while not self._shutdown_event.is_set():
            try:
                # Check if recovering - skip health check
                if name in self.recovery_in_progress:
                    await asyncio.sleep(1)
                    continue
                
                # Check circuit breaker
                breaker = self.circuit_breakers[name]
                if not await breaker.can_execute():
                    logger.warning(f"Circuit breaker open for {name}, skipping health check")
                    await asyncio.sleep(10)
                    continue
                
                # Perform health check with timeout
                try:
                    healthy = await asyncio.wait_for(
                        self.health_checks[name](),
                        timeout=config.health_check_interval
                    )
                except asyncio.TimeoutError:
                    healthy = False
                    logger.error(f"Health check timeout for {name}")
                
                # Update state
                if healthy:
                    self.component_states[name] = ComponentState.HEALTHY
                    await breaker.record_success()
                else:
                    self.component_states[name] = ComponentState.UNHEALTHY
                    await self._handle_component_failure(name, "Health check failed")
                
            except Exception as e:
                logger.error(f"Monitor error for {name}: {e}")
                await self._handle_component_failure(name, f"Monitor exception: {e}")
            
            # Wait for next check
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=config.health_check_interval
                )
            except asyncio.TimeoutError:
                continue
    
    async def _global_watchdog(self):
        """Global system health monitoring"""
        while not self._shutdown_event.is_set():
            try:
                # Check critical component failures
                critical_failures = [
                    name for name, state in self.component_states.items()
                    if state == ComponentState.FAILED 
                    and self.component_configs[name].critical
                ]
                
                if critical_failures:
                    logger.critical(f"Critical components failed: {critical_failures}")
                    await self._emergency_shutdown("Critical component failure")
                    return
                
                # Check dependency chain failures
                await self._check_dependencies()
                
                # Save periodic state snapshot
                if self.global_failure_count == 0:
                    await self._save_periodic_state()
                
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
            
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=60  # Check every minute
                )
            except asyncio.TimeoutError:
                continue
    
    async def _check_dependencies(self):
        """Ensure dependency chains are healthy"""
        for name, config in self.component_configs.items():
            if self.component_states[name] != ComponentState.HEALTHY:
                continue
            
            # Check if dependencies are healthy
            for dep in config.dependencies:
                if self.component_states.get(dep) != ComponentState.HEALTHY:
                    logger.warning(
                        f"{name} dependency {dep} unhealthy, "
                        f"degrading {name}"
                    )
                    self.component_states[name] = ComponentState.DEGRADED
    
    # ==================== FAILURE HANDLING ====================
    
    async def _handle_component_failure(self, name: str, error: str):
        """Handle component failure with circuit breaker"""
        # Record failure
        failure = FailureEvent(
            timestamp=datetime.now(),
            component=name,
            error_type=type(error).__name__ if isinstance(error, Exception) else "Unknown",
            error_message=str(error),
            traceback=traceback.format_exc()
        )
        self.failure_history.append(failure)
        self.global_failure_count += 1
        
        # Check circuit breaker
        breaker = self.circuit_breakers[name]
        if not await breaker.record_failure():
            logger.critical(f"Circuit breaker preventing recovery of {name}")
            self.component_states[name] = ComponentState.FAILED
            return
        
        # Start recovery
        asyncio.create_task(self._recover_component(name))
    
    async def _recover_component(self, name: str):
        """Execute component-specific recovery"""
        if name in self.recovery_in_progress:
            return
        
        self.recovery_in_progress.add(name)
        self.component_states[name] = ComponentState.RECOVERING
        
        config = self.component_configs[name]
        failure = self.failure_history[-1] if self.failure_history else None
        
        logger.info(f"Starting recovery of {name} "
                   f"(attempt {failure.recovery_attempts + 1 if failure else 1})")
        
        try:
            # Execute recovery with timeout
            success = await asyncio.wait_for(
                self._execute_recovery_action(name, config.recovery_action),
                timeout=config.recovery_timeout
            )
            
            if success:
                self.component_states[name] = ComponentState.HEALTHY
                if failure:
                    failure.resolved = True
                logger.info(f"Recovery of {name} successful")
                
                # Notify dependent components to check health
                await self._notify_dependents(name)
            else:
                raise RuntimeError("Recovery action returned False")
                
        except asyncio.TimeoutError:
            logger.error(f"Recovery timeout for {name}")
            failure.recovery_attempts += 1
            
            if failure.recovery_attempts >= config.max_recovery_attempts:
                self.component_states[name] = ComponentState.FAILED
                if config.critical:
                    await self._emergency_shutdown(f"Critical component {name} failed recovery")
            else:
                # Retry after cooldown
                await asyncio.sleep(config.cooldown_seconds)
                asyncio.create_task(self._recover_component(name))
                
        except Exception as e:
            logger.error(f"Recovery failed for {name}: {e}")
            failure.recovery_attempts += 1
            self.component_states[name] = ComponentState.FAILED
            
            if config.critical:
                await self._emergency_shutdown(f"Critical component {name} recovery error")
        
        finally:
            self.recovery_in_progress.discard(name)
    
    async def _execute_recovery_action(self, name: str, action: RecoveryAction) -> bool:
        """Execute specific recovery action"""
        if action == RecoveryAction.RESTART:
            return await self._restart_component(name)
        elif action == RecoveryAction.RECONNECT:
            return await self._reconnect_component(name)
        elif action == RecoveryAction.RESET_STATE:
            return await self._reset_component_state(name)
        elif action == RecoveryAction.FAILOVER:
            return await self._failover_component(name)
        elif action == RecoveryAction.GRACEFUL_DEGRADE:
            return await self._degrade_component(name)
        else:
            logger.error(f"Unknown recovery action: {action}")
            return False
    
    async def _restart_component(self, name: str) -> bool:
        """Restart component instance"""
        try:
            # Save state before restart
            await self._save_component_state(name)
            
            # Get component class and recreate
            old_instance = self.components[name]
            component_class = old_instance.__class__
            
            # Create new instance
            new_instance = component_class(self.config)
            
            # Initialize with retry
            for attempt in range(3):
                try:
                    if hasattr(new_instance, 'initialize'):
                        await asyncio.wait_for(
                            new_instance.initialize(),
                            timeout=30.0
                        )
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"Init retry {attempt + 1} for {name}")
                    await asyncio.sleep(2 ** attempt)
            
            # Update references
            self.components[name] = new_instance
            
            # Restore state if applicable
            await self._restore_component_state(name)
            
            return True
            
        except Exception as e:
            logger.error(f"Component restart failed for {name}: {e}")
            return False
    
    async def _reconnect_component(self, name: str) -> bool:
        """Reconnect component (for connection-based components)"""
        instance = self.components[name]
        
        try:
            if hasattr(instance, 'reconnect'):
                await instance.reconnect()
            elif hasattr(instance, 'disconnect') and hasattr(instance, 'connect'):
                await instance.disconnect()
                await asyncio.sleep(1)
                await instance.connect()
            else:
                # Fallback to restart
                return await self._restart_component(name)
            
            return True
            
        except Exception as e:
            logger.error(f"Reconnect failed for {name}: {e}")
            return False
    
    async def _reset_component_state(self, name: str) -> bool:
        """Reset component to clean state"""
        try:
            instance = self.components[name]
            
            if hasattr(instance, 'reset_state'):
                await instance.reset_state()
            else:
                # Clear internal caches/data
                if hasattr(instance, 'cache'):
                    instance.cache.clear()
            
            return True
            
        except Exception as e:
            logger.error(f"State reset failed for {name}: {e}")
            return False
    
    async def _failover_component(self, name: str) -> bool:
        """Failover to backup instance"""
        # Implementation depends on specific failover strategy
        logger.warning(f"Failover not implemented for {name}")
        return False
    
    async def _degrade_component(self, name: str) -> bool:
        """Gracefully degrade component functionality"""
        try:
            instance = self.components[name]
            
            if hasattr(instance, 'degrade'):
                await instance.degrade()
                self.component_states[name] = ComponentState.DEGRADED
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Degradation failed for {name}: {e}")
            return False
    
    async def _notify_dependents(self, component_name: str):
        """Notify dependent components to recheck health"""
        for name, config in self.component_configs.items():
            if component_name in config.dependencies:
                if self.component_states[name] == ComponentState.DEGRADED:
                    # Trigger health check
                    asyncio.create_task(self._trigger_health_check(name))
    
    async def _trigger_health_check(self, name: str):
        """Manually trigger health check"""
        try:
            healthy = await self.health_checks[name]()
            if healthy:
                self.component_states[name] = ComponentState.HEALTHY
        except Exception as e:
            logger.error(f"Triggered health check failed for {name}: {e}")
    
    # ==================== STATE MANAGEMENT ====================
    
    async def _save_component_state(self, name: str):
        """Save component state before recovery"""
        instance = self.components[name]
        
        try:
            if hasattr(instance, 'get_state'):
                state = await instance.get_state()
                await self.state_persistence.save(f"component_{name}", state)
        except Exception as e:
            logger.warning(f"Could not save state for {name}: {e}")
    
    async def _restore_component_state(self, name: str):
        """Restore component state after recovery"""
        try:
            state = await self.state_persistence.load_latest(f"component_{name}")
            if state and hasattr(self.components[name], 'set_state'):
                await self.components[name].set_state(state)
                logger.info(f"State restored for {name}")
        except Exception as e:
            logger.warning(f"Could not restore state for {name}: {e}")
    
    async def _save_periodic_state(self):
        """Save periodic full system state"""
        try:
            full_state = {
                'timestamp': datetime.now().isoformat(),
                'components': {
                    name: {
                        'state': self.component_states[name].name,
                        'health': self.component_states[name] == ComponentState.HEALTHY
                    }
                    for name in self.components
                },
                'failures_24h': len([
                    f for f in self.failure_history 
                    if f.timestamp > datetime.now() - timedelta(hours=24)
                ])
            }
            
            await self.state_persistence.save("system", full_state)
            
        except Exception as e:
            logger.error(f"Periodic state save failed: {e}")
    
    async def _emergency_save_state(self):
        """Emergency state preservation"""
        try:
            emergency_state = {
                'timestamp': datetime.now().isoformat(),
                'reason': 'emergency_shutdown',
                'open_positions': [],  # Would come from position tracker
                'component_states': {
                    name: state.name for name, state in self.component_states.items()
                }
            }
            
            path = await self.state_persistence.save("emergency", emergency_state)
            logger.info(f"Emergency state saved to {path}")
            
        except Exception as e:
            logger.critical(f"Failed to save emergency state: {e}")
    
    # ==================== SHUTDOWN ====================
    
    def _signal_handler(self, signum: int):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        asyncio.create_task(self._initiate_shutdown())
    
    async def _initiate_shutdown(self):
        """Initiate graceful shutdown"""
        logger.info("Initiating graceful shutdown...")
        self._shutdown_event.set()
        
        # Stop all components gracefully
        shutdown_tasks = []
        for name, instance in self.components.items():
            if hasattr(instance, 'shutdown'):
                task = asyncio.create_task(
                    self._shutdown_component(name, instance),
                    name=f"shutdown_{name}"
                )
                shutdown_tasks.append(task)
        
        if shutdown_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*shutdown_tasks, return_exceptions=True),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.warning("Component shutdown timeout, forcing exit")
        
        await self.shutdown()
        sys.exit(0)
    
    async def _shutdown_component(self, name: str, instance: Any):
        """Shutdown single component"""
        try:
            await asyncio.wait_for(instance.shutdown(), timeout=10.0)
            self.component_states[name] = ComponentState.SHUTDOWN
        except Exception as e:
            logger.error(f"Error shutting down {name}: {e}")
    
    async def _emergency_shutdown(self, reason: str):
        """Emergency shutdown with position protection"""
        logger.critical(f"EMERGENCY SHUTDOWN: {reason}")
        
        # Save state immediately
        await self._emergency_save_state()
        
        # Attempt to close positions if execution engine available
        if 'execution' in self.components and self.component_states['execution'] != ComponentState.FAILED:
            try:
                execution = self.components['execution']
                if hasattr(execution, 'emergency_close_all'):
                    await asyncio.wait_for(
                        execution.emergency_close_all(),
                        timeout=10.0
                    )
                    logger.info("Emergency position close executed")
            except Exception as e:
                logger.critical(f"Could not close positions: {e}")
        
        # Force exit
        sys.exit(1)
    
    # ==================== REPORTING ====================
    
    def get_health_report(self) -> Dict:
        """Get comprehensive health report"""
        now = datetime.now()
        
        recent_failures = [
            f for f in self.failure_history
            if f.timestamp > now - timedelta(hours=24)
        ]
        
        return {
            'timestamp': now.isoformat(),
            'overall_status': 'healthy' if all(
                s == ComponentState.HEALTHY for s in self.component_states.values()
            ) else 'degraded' if any(
                s == ComponentState.DEGRADED for s in self.component_states.values()
            ) else 'critical',
            'components': {
                name: {
                    'state': state.name,
                    'circuit_breaker': self.circuit_breakers[name].state,
                    'config': {
                        'critical': self.component_configs[name].critical,
                        'dependencies': self.component_configs[name].dependencies
                    }
                }
                for name, state in self.component_states.items()
            },
            'failures_24h': len(recent_failures),
            'failures_details': [
                {
                    'time': f.timestamp.isoformat(),
                    'component': f.component,
                    'error': f.error_message,
                    'resolved': f.resolved
                }
                for f in recent_failures[-10:]  # Last 10
            ],
            'recovery_stats': {
                'global_failure_count': self.global_failure_count,
                'last_restart': self.last_global_restart.isoformat() if self.last_global_restart else None
            }
        }


# ==================== USAGE EXAMPLE ====================

class MockComponent:
    """Example component for testing"""
    
    def __init__(self, config):
        self.config = config
        self.healthy = True
        self.restart_count = 0
    
    async def initialize(self):
        self.restart_count += 1
        await asyncio.sleep(0.1)
    
    async def shutdown(self):
        await asyncio.sleep(0.1)
    
    async def get_state(self):
        return {'restart_count': self.restart_count}
    
    async def set_state(self, state):
        self.restart_count = state.get('restart_count', 0)


async def example_usage():
    """Example recovery manager setup"""
    
    config = {
        'state_dir': 'recovery_state',
        'max_restarts': 3
    }
    
    recovery = RecoveryManager(config)
    
    # Create mock components
    data_engine = MockComponent(config)
    strategy_engine = MockComponent(config)
    execution_engine = MockComponent(config)
    
    # Register with dependencies (strategy depends on data)
    recovery.register_component(
        'data',
        data_engine,
        ComponentConfig(
            name='data',
            critical=True,
            recovery_action=RecoveryAction.RECONNECT,
            health_check_interval=10.0
        ),
        lambda: data_engine.healthy
    )
    
    recovery.register_component(
        'strategy',
        strategy_engine,
        ComponentConfig(
            name='strategy',
            critical=True,
            dependencies=['data'],
            recovery_action=RecoveryAction.RESTART
        ),
        lambda: strategy_engine.healthy
    )
    
    recovery.register_component(
        'execution',
        execution_engine,
        ComponentConfig(
            name='execution',
            critical=True,
            recovery_action=RecoveryAction.RESTART
        ),
        lambda: execution_engine.healthy
    )
    
    # Start monitoring
    await recovery.start()
    
    # Simulate failure
    print("Simulating data engine failure...")
    data_engine.healthy = False
    
    # Wait for recovery
    await asyncio.sleep(15)
    
    # Check health report
    report = recovery.get_health_report()
    print(f"\nHealth Report: {json.dumps(report, indent=2, default=str)}")
    
    # Shutdown
    await recovery.shutdown()


if __name__ == "__main__":
    asyncio.run(example_usage())