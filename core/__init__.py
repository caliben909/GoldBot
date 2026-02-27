"""
Core Trading Bot Modules
=======================

Institutional-grade trading system components.

Modules:
    data_engine: Market data ingestion and management
    execution_engine: Order execution and broker integration
    risk_engine: Position sizing and risk management
    strategy_engine: Trading strategy orchestration
    session_engine: Market session detection and liquidity analysis
    ai_engine: Machine learning predictions (optional)
    liquidity_engine: Order book analysis and slippage estimation

Example:
    >>> from core import DataEngine, RiskEngine
    >>> data = DataEngine(config)
    >>> risk = RiskEngine(config)
"""

__version__ = "2.1.0"
__author__ = "Institutional Trading Team"
__license__ = "Proprietary"

import importlib
import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# Type checking imports (no runtime overhead)
if TYPE_CHECKING:
    from core.data_engine import DataEngine
    from core.execution_engine import ExecutionEngine
    from core.risk_engine import RiskEngine
    from core.strategy_engine import StrategyEngine
    from core.session_engine import SessionEngine
    from core.liquidity_engine import LiquidityEngine

# Public API - lazy loaded to prevent circular imports and speed up startup
__all__ = [
    'DataEngine',
    'ExecutionEngine', 
    'RiskEngine',
    'StrategyEngine',
    'SessionEngine',
    'LiquidityEngine',
    'get_engine',  # Factory function
    'initialize_engines',  # Bulk initialization
]

# Module availability flags
_AVAILABLE_ENGINES: dict[str, bool] = {
    'data': True,
    'execution': True,
    'risk': True,
    'strategy': True,
    'session': True,
    'liquidity': True,
    'ai': False,  # Disabled by default
}


def _lazy_import(module_name: str, class_name: str):
    """
    Lazy import factory to prevent circular dependencies
    and reduce startup time.
    """
    def import_factory(*args, **kwargs):
        try:
            module = importlib.import_module(f"core.{module_name}")
            cls = getattr(module, class_name)
            return cls(*args, **kwargs)
        except ImportError as e:
            logger.error(f"Failed to import {class_name}: {e}")
            raise
        except AttributeError as e:
            logger.error(f"Class {class_name} not found in {module_name}: {e}")
            raise
    return import_factory


# Factory functions for lazy instantiation
def DataEngine(*args, **kwargs):  # type: ignore
    """Factory for DataEngine - imported on first call"""
    return _lazy_import('data_engine', 'DataEngine')(*args, **kwargs)


def ExecutionEngine(*args, **kwargs):  # type: ignore
    """Factory for ExecutionEngine - imported on first call"""
    return _lazy_import('execution_engine', 'ExecutionEngine')(*args, **kwargs)


def RiskEngine(*args, **kwargs):  # type: ignore
    """Factory for RiskEngine - imported on first call"""
    return _lazy_import('risk_engine', 'RiskEngine')(*args, **kwargs)


def StrategyEngine(*args, **kwargs):  # type: ignore
    """Factory for StrategyEngine - imported on first call"""
    return _lazy_import('strategy_engine', 'StrategyEngine')(*args, **kwargs)


def SessionEngine(*args, **kwargs):  # type: ignore
    """Factory for SessionEngine - imported on first call"""
    return _lazy_import('session_engine', 'SessionEngine')(*args, **kwargs)


def LiquidityEngine(*args, **kwargs):  # type: ignore
    """Factory for LiquidityEngine - imported on first call"""
    return _lazy_import('liquidity_engine', 'LiquidityEngine')(*args, **kwargs)


def AIEngine(*args, **kwargs):  # type: ignore
    """
    Factory for AIEngine - imported on first call.
    Raises RuntimeError if AI module not available.
    """
    if not _AVAILABLE_ENGINES.get('ai'):
        raise RuntimeError(
            "AIEngine is disabled. Enable by setting core._AVAILABLE_ENGINES['ai'] = True "
            "and ensuring AI dependencies are installed."
        )
    return _lazy_import('ai_engine', 'AIEngine')(*args, **kwargs)


def get_engine(engine_type: str, *args, **kwargs):
    """
    Generic engine factory.
    
    Args:
        engine_type: One of 'data', 'execution', 'risk', 'strategy', 
                     'session', 'liquidity', 'ai'
        *args, **kwargs: Passed to engine constructor
    
    Returns:
        Engine instance
    
    Raises:
        ValueError: If engine_type unknown
        RuntimeError: If engine disabled
    """
    factories = {
        'data': DataEngine,
        'execution': ExecutionEngine,
        'risk': RiskEngine,
        'strategy': StrategyEngine,
        'session': SessionEngine,
        'liquidity': LiquidityEngine,
        'ai': AIEngine,
    }
    
    factory = factories.get(engine_type.lower())
    if not factory:
        raise ValueError(f"Unknown engine type: {engine_type}. "
                        f"Available: {list(factories.keys())}")
    
    return factory(*args, **kwargs)


async def initialize_engines(config: dict, 
                            engine_types: list[str] | None = None) -> dict:
    """
    Bulk initialize multiple engines with dependency ordering.
    
    Args:
        config: Configuration dictionary
        engine_types: List of engines to initialize, or None for all
    
    Returns:
        Dictionary of initialized engines {name: instance}
    
    Example:
        >>> engines = await initialize_engines(config, 
        ...                                    ['data', 'risk', 'execution'])
        >>> data_engine = engines['data']
    """
    # Default: initialize all available
    if engine_types is None:
        engine_types = [k for k, v in _AVAILABLE_ENGINES.items() if v]
    
    # Define initialization order (dependencies first)
    init_order = ['data', 'session', 'liquidity', 'risk', 'ai', 'strategy', 'execution']
    ordered = [e for e in init_order if e in engine_types]
    
    # Add any not in default order (append to end)
    ordered.extend([e for e in engine_types if e not in init_order])
    
    engines = {}
    
    for engine_type in ordered:
        try:
            logger.info(f"Initializing {engine_type} engine...")
            engine = get_engine(engine_type, config)
            
            # Initialize if async method exists
            if hasattr(engine, 'initialize'):
                if asyncio.iscoroutinefunction(engine.initialize):
                    await engine.initialize()
                else:
                    engine.initialize()
            
            engines[engine_type] = engine
            logger.info(f"✓ {engine_type} engine ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize {engine_type} engine: {e}")
            if engine_type in ['data', 'risk']:  # Critical engines
                raise RuntimeError(f"Critical engine {engine_type} failed: {e}")
            continue
    
    return engines


def enable_ai_engine():
    """Enable AI engine (requires ML dependencies)"""
    try:
        import xgboost
        import sklearn
        _AVAILABLE_ENGINES['ai'] = True
        __all__.append('AIEngine')
        logger.info("AI Engine enabled")
    except ImportError as e:
        logger.warning(f"Cannot enable AI Engine - missing dependencies: {e}")


def list_available_engines() -> dict[str, bool]:
    """Get dictionary of available engines and their status"""
    return _AVAILABLE_ENGINES.copy()


# Optional: Import ai_engine if dependencies available
try:
    import xgboost
    import sklearn
    _AVAILABLE_ENGINES['ai'] = True
    __all__.append('AIEngine')
except ImportError:
    pass  # AI engine remains disabled