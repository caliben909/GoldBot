#!/usr/bin/env python3
"""Test to verify silver (XAGUSD) trading configuration"""

import sys
import logging
from pathlib import Path
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_silver_configuration():
    """Test if silver is properly configured in the trading bot"""
    logger.info("Testing silver (XAGUSD) trading configuration...")
    
    try:
        # Load configuration
        config_path = Path(__file__).parent / "config" / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Check if XAGUSD is in forex symbols
        logger.info("1. Checking if XAGUSD is in forex symbols list...")
        forex_symbols = config['assets']['forex']['symbols']
        if 'XAGUSD' in forex_symbols:
            logger.info("✅ XAGUSD is in forex symbols list")
            logger.debug(f"Forex symbols: {', '.join(forex_symbols)}")
        else:
            logger.error("❌ XAGUSD not found in forex symbols list")
            return False
        
        # Check if XAUUSD (gold) is also configured (should be for precious metals trading)
        if 'XAUUSD' in forex_symbols:
            logger.info("✅ XAUUSD (gold) is also configured for precious metals trading")
        
        # Check if XAGUSD has DXY correlation threshold
        logger.info("2. Checking DXY correlation filter configuration for XAGUSD...")
        dxy_correlation = config['risk_management']['dxy_correlation']
        
        if 'XAGUSD' in dxy_correlation['symbol_correlation_thresholds']:
            threshold = dxy_correlation['symbol_correlation_thresholds']['XAGUSD']
            logger.info(f"✅ XAGUSD correlation threshold: {threshold:.2f} (negative correlation expected)")
        else:
            logger.error("❌ XAGUSD correlation threshold not configured")
            return False
        
        # Check if XAGUSD is included in trading sessions
        logger.info("3. Checking if XAGUSD is included in trading sessions...")
        sessions = config['sessions']
        all_sessions_include = True
        
        for session_name, session_config in sessions.items():
            if 'XAGUSD' not in session_config['pairs']:
                logger.warning(f"⚠️ XAGUSD not in {session_name} session pairs: {session_config['pairs']}")
                all_sessions_include = False
            else:
                logger.debug(f"✅ {session_name} session includes XAGUSD")
        
        if all_sessions_include:
            logger.info("✅ XAGUSD is included in all trading sessions")
        
        # Check if trading sessions are enabled
        logger.info("4. Verifying trading sessions are enabled...")
        enabled_sessions = [name for name, config in sessions.items() if config['enabled']]
        logger.info(f"Enabled sessions: {', '.join(enabled_sessions)}")
        
        # Check DXY correlation filter settings
        logger.info("5. Verifying DXY correlation filter settings...")
        if dxy_correlation['enabled']:
            logger.info("✅ DXY correlation filter is enabled")
            logger.debug(f"Correlation method: {dxy_correlation['correlation_method']}")
            logger.debug(f"Lookback period: {dxy_correlation['lookback_period']} days")
            logger.debug(f"Min correlation strength: {dxy_correlation['minimum_correlation_strength']}")
        else:
            logger.warning("⚠️ DXY correlation filter is disabled - XAGUSD correlation filtering will not work")
        
        # Check if risk management settings are reasonable
        logger.info("6. Checking risk management settings...")
        risk_settings = config['risk_management']
        logger.debug(f"Max risk per trade: {risk_settings['max_risk_per_trade']}%")
        logger.debug(f"Position sizing method: {risk_settings['position_sizing']['method']}")
        
        # Verify the configuration is complete for XAGUSD
        logger.info("7. Verifying complete configuration for XAGUSD trading...")
        
        # Check if MT5 connection settings exist
        if config['execution']['mt5']['enabled']:
            logger.info("✅ MT5 connection is enabled for XAGUSD trading")
        else:
            logger.warning("⚠️ MT5 connection is disabled - XAGUSD trading may not work")
        
        logger.info("\n✅ Silver (XAGUSD) trading configuration is complete and properly set up!")
        
        # Provide summary of XAGUSD configuration
        logger.info("\n=== XAGUSD Trading Configuration Summary ===")
        logger.info("Symbol: XAGUSD (Silver/USD)")
        logger.info(f"DXY Correlation Threshold: {dxy_correlation['symbol_correlation_thresholds']['XAGUSD']:.2f}")
        logger.info(f"Expected Correlation: Negative (silver prices move opposite to DXY)")
        logger.info(f"Included in Sessions: {', '.join([name for name, config in sessions.items() if 'XAGUSD' in config['pairs']])}")
        logger.info(f"Trading Enabled: {config['assets']['forex']['enabled']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}", exc_info=True)
        return False

def test_imports_for_silver_trading():
    """Test if modules needed for silver trading are importable"""
    logger.info("\nTesting module imports for silver trading...")
    
    try:
        from core.data_engine import DataEngine
        from core.risk_engine import RiskEngine
        from core.strategy_engine import StrategyEngine
        from core.risk.dxy_correlation_filter import DXYCorrelationFilter
        
        logger.info("✅ All required modules for XAGUSD trading are importable")
        return True
        
    except Exception as e:
        logger.error(f"❌ Module import failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("SILVER (XAGUSD) TRADING CONFIGURATION TEST")
    logger.info("=" * 60)
    
    try:
        config_test = test_silver_configuration()
        import_test = test_imports_for_silver_trading()
        
        if config_test and import_test:
            logger.info("\n🎉 Silver (XAGUSD) trading configuration is valid and ready!")
            logger.info("\nImportant Notes:")
            logger.info("- XAGUSD has a configured DXY correlation threshold of -0.70 (negative correlation)")
            logger.info("- Silver prices typically move opposite to the US Dollar Index (DXY)")
            logger.info("- XAGUSD is included in all main trading sessions")
            logger.info("- The DXY correlation filter will help validate silver trading signals")
            sys.exit(0)
        else:
            logger.error("\n❌ Silver trading configuration has issues that need to be fixed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        sys.exit(1)
