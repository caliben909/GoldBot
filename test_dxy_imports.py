#!/usr/bin/env python3
"""Simple test to verify DXY correlation filter imports"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_dxy_correlation_imports():
    """Test if DXY correlation filter modules are importable"""
    logger.info("Testing DXY correlation filter imports...")
    
    try:
        from core.risk.dxy_correlation_filter import DXYCorrelationFilter, DXYCorrelationConfig
        logger.debug("✅ DXYCorrelationFilter imported successfully")
    except Exception as e:
        logger.error(f"❌ DXYCorrelationFilter import failed: {e}")
        return False
    
    try:
        from core.risk_engine import RiskEngine
        logger.debug("✅ RiskEngine imported successfully")
    except Exception as e:
        logger.error(f"❌ RiskEngine import failed: {e}")
        return False
    
    try:
        from core.strategy_engine import StrategyEngine
        logger.debug("✅ StrategyEngine imported successfully")
    except Exception as e:
        logger.error(f"❌ StrategyEngine import failed: {e}")
        return False
    
    logger.info("✅ All DXY correlation filter related modules imported successfully!")
    return True

if __name__ == "__main__":
    try:
        success = test_dxy_correlation_imports()
        
        if success:
            logger.info("\n✨ DXY correlation filter implementation is available and importable")
            sys.exit(0)
        else:
            logger.error("\n❌ DXY correlation filter implementation has import errors")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Test execution failed: {e}", exc_info=True)
        sys.exit(1)
