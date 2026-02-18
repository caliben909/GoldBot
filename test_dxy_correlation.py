#!/usr/bin/env python3
"""
Test script for DXY correlation filter functionality
"""

import asyncio
import sys
import logging
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.risk.dxy_correlation_filter import DXYCorrelationFilter, DXYCorrelationConfig
from core.risk_engine import RiskEngine
from core.strategy_engine import StrategyEngine
from core.data_engine import DataEngine
import yaml
from pathlib import Path

def load_config():
    """Load configuration from file"""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_dxy_correlation_filter():
    """Test the DXY correlation filter functionality"""
    logger.info("=" * 60)
    logger.info("TESTING DXY CORRELATION FILTER")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config = load_config()
        
        # Test 1: Initialize DXY correlation filter
        logger.info("\n1. Testing DXY correlation filter initialization...")
        filter_engine = DXYCorrelationFilter(config)
        logger.debug("✅ DXY correlation filter initialized successfully")
        
        # Test 2: Create test data
        logger.info("\n2. Creating test price data...")
        
        # Generate DXY price history (simulating a downtrend)
        dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='H')
        base_price = 105.0
        dxy_prices = []
        for i, date in enumerate(dates):
            trend = -0.0001 * i  # Downward trend
            noise = np.random.normal(0, 0.005)
            price = base_price + trend + noise
            dxy_prices.append(price)
        
        dxy_series = pd.Series(dxy_prices, index=dates)
        await filter_engine.update_dxy_data(dxy_series)
        logger.debug(f"✅ DXY price history created with {len(dxy_series)} points")
        
        # Generate EURUSD price history (negative correlation with DXY)
        eur_usd_prices = []
        base_eur = 1.08
        for i, date in enumerate(dates):
            correlation = -0.85
            trend = 0.0001 * i  # Upward trend (negative correlation to DXY)
            noise = np.random.normal(0, 0.0005)
            price = base_eur + trend + noise
            eur_usd_prices.append(price)
        
        eur_series = pd.Series(eur_usd_prices, index=dates)
        await filter_engine.update_symbol_data('EURUSD', eur_series)
        logger.debug(f"✅ EURUSD price history created with {len(eur_series)} points")
        
        # Generate USDJPY price history (positive correlation with DXY)
        usd_jpy_prices = []
        base_jpy = 145.0
        for i, date in enumerate(dates):
            correlation = 0.65
            trend = -0.0001 * i  # Downward trend (positive correlation to DXY)
            noise = np.random.normal(0, 0.005)
            price = base_jpy + trend + noise
            usd_jpy_prices.append(price)
        
        jpy_series = pd.Series(usd_jpy_prices, index=dates)
        await filter_engine.update_symbol_data('USDJPY', jpy_series)
        logger.debug(f"✅ USDJPY price history created with {len(jpy_series)} points")
        
        # Test 3: Calculate correlations
        logger.info("\n3. Calculating DXY correlations...")
        result = await filter_engine.calculate_correlations(['EURUSD', 'USDJPY'])
        
        logger.debug(f"✅ DXY price: {result.dxy_price:.2f}")
        logger.debug(f"✅ DXY trend: {result.dxy_trend}")
        logger.debug(f"✅ DXY momentum: {result.dxy_momentum:.4f}")
        logger.debug(f"✅ DXY volatility: {result.dxy_volatility:.2%}")
        
        logger.debug(f"\nCorrelations:")
        for symbol, corr in result.correlations.items():
            logger.debug(f"  {symbol}: {corr:.3f} (strength: {result.correlation_strengths[symbol]:.3f})")
        
        logger.debug(f"\nEligible symbols: {result.eligible_symbols}")
        logger.debug(f"Filtered symbols: {result.filtered_symbols}")
        
        logger.debug(f"\nPosition size adjustments:")
        for symbol, adj in result.position_size_adjustments.items():
            logger.debug(f"  {symbol}: {adj:.2f}x")
        
        logger.debug(f"\nPortfolio correlation: {result.portfolio_correlation:.3f}")
        logger.debug(f"Diversification score: {result.diversification_score:.3f}")
        
        # Test 4: Check if signals should be filtered
        logger.info("\n4. Testing signal filtering...")
        
        # Test EURUSD long signal (should be eligible)
        eur_long_signal = {
            'symbol': 'EURUSD',
            'direction': 'long',
            'timestamp': datetime.now(),
            'entry_price': eur_series.iloc[-1],
            'stop_loss': eur_series.iloc[-1] * 0.995,
            'take_profit': eur_series.iloc[-1] * 1.01
        }
        
        should_filter, reason = await filter_engine.should_filter_signal(eur_long_signal)
        logger.debug(f"EURUSD long signal: {'❌ Filtered' if should_filter else '✅ Passed'} - {reason}")
        
        # Test EURUSD short signal (should be filtered due to trend conflict)
        eur_short_signal = {
            'symbol': 'EURUSD',
            'direction': 'short',
            'timestamp': datetime.now(),
            'entry_price': eur_series.iloc[-1],
            'stop_loss': eur_series.iloc[-1] * 1.005,
            'take_profit': eur_series.iloc[-1] * 0.99
        }
        
        should_filter, reason = await filter_engine.should_filter_signal(eur_short_signal)
        logger.debug(f"EURUSD short signal: {'❌ Filtered' if should_filter else '✅ Passed'} - {reason}")
        
        # Test USDJPY short signal (should be eligible)
        jpy_short_signal = {
            'symbol': 'USDJPY',
            'direction': 'short',
            'timestamp': datetime.now(),
            'entry_price': jpy_series.iloc[-1],
            'stop_loss': jpy_series.iloc[-1] * 1.005,
            'take_profit': jpy_series.iloc[-1] * 0.99
        }
        
        should_filter, reason = await filter_engine.should_filter_signal(jpy_short_signal)
        logger.debug(f"USDJPY short signal: {'❌ Filtered' if should_filter else '✅ Passed'} - {reason}")
        
        # Test 5: Position size adjustment
        logger.info("\n5. Testing position size adjustment...")
        base_size = 0.1
        for symbol in ['EURUSD', 'USDJPY']:
            adjusted_size = await filter_engine.adjust_position_size(symbol, base_size)
            logger.debug(f"{symbol}: {base_size:.2f} -> {adjusted_size:.4f}")
        
        # Test 6: Extreme correlation alert
        logger.info("\n6. Testing extreme correlation alerts...")
        extreme_events = await filter_engine.check_extreme_correlation()
        logger.debug(f"Extreme correlation events: {len(extreme_events)}")
        
        for event in extreme_events:
            logger.debug(f"  ⚠️ {event['symbol']}: Correlation strength {event['correlation_strength']:.2f}")
        
        # Test 7: Verify configuration
        logger.info("\n7. Verifying DXY correlation filter configuration...")
        config_check = config['risk_management']['dxy_correlation']
        
        logger.debug(f"Enabled: {config_check['enabled']}")
        logger.debug(f"Method: {config_check['correlation_method']}")
        logger.debug(f"Lookback: {config_check['lookback_period']} days")
        logger.debug(f"Min observations: {config_check['min_observations']}")
        
        logger.debug("\nSymbol correlation thresholds:")
        for symbol, threshold in config_check['symbol_correlation_thresholds'].items():
            logger.debug(f"  {symbol}: {threshold:.2f}")
        
        logger.debug(f"\nFilter on correlation strength: {config_check['filter_on_correlation_strength']}")
        logger.debug(f"Min strength: {config_check['minimum_correlation_strength']}")
        logger.debug(f"Max strength: {config_check['maximum_correlation_strength']}")
        
        logger.debug(f"\nTrend confirmation: {config_check['require_dxy_trend_confirmation']}")
        logger.debug(f"Trend threshold: {config_check['trend_strength_threshold']}")
        
        logger.debug(f"\nPosition sizing adjustment: {config_check['adjust_position_size_by_correlation']}")
        logger.debug(f"Sizing multiplier: {config_check['correlation_sizing_multiplier']}")
        logger.debug(f"Max reduction: {config_check['max_correlation_sizing_reduction']}")
        
        logger.debug(f"\nAlert on extreme correlation: {config_check['alert_on_extreme_correlation']}")
        logger.debug(f"Extreme threshold: {config_check['extreme_correlation_threshold']}")
        
        logger.info("\n✅ DXY correlation filter functionality test completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False

@pytest.mark.asyncio
async def test_integration_with_risk_engine():
    """Test integration with RiskEngine"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING RISK ENGINE INTEGRATION")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config = load_config()
        
        # Create risk engine
        risk_engine = RiskEngine(config)
        
        # Test updating DXY data
        dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='H')
        dxy_prices = 105.0 + np.cumsum(np.random.normal(0, 0.001, len(dates)))
        dxy_series = pd.Series(dxy_prices, index=dates)
        await risk_engine.update_dxy_data(dxy_series)
        logger.debug("✅ DXY data updated in risk engine")
        
        # Test symbol data update
        eur_prices = 1.08 + np.cumsum(np.random.normal(0, 0.0005, len(dates)))
        eur_series = pd.Series(eur_prices, index=dates)
        await risk_engine.update_symbol_data_for_correlation('EURUSD', eur_series)
        logger.debug("✅ Symbol data updated in risk engine")
        
        # Test signal filtering through risk engine
        signal = {
            'symbol': 'EURUSD',
            'direction': 'long',
            'timestamp': datetime.now(),
            'entry_price': 1.0850,
            'stop_loss': 1.0800,
            'take_profit': 1.0950
        }
        
        should_filter, reason = await risk_engine.should_filter_signal_by_dxy_correlation(signal)
        logger.debug(f"Signal filtering: {'❌ Filtered' if should_filter else '✅ Passed'} - {reason}")
        
        # Test position size adjustment
        base_size = 0.1
        adjusted_size = await risk_engine.adjust_position_size_by_dxy_correlation('EURUSD', base_size)
        logger.debug(f"Position adjustment: {base_size:.2f} -> {adjusted_size:.4f}")
        
        logger.info("\n✅ Risk engine integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}", exc_info=True)
        return False

async def main():
    """Main test function"""
    # Run all tests
    logger.info("Starting comprehensive DXY correlation filter tests...")
    
    # Test filter functionality
    filter_test = await test_dxy_correlation_filter()
    
    # Test risk engine integration
    integration_test = await test_integration_with_risk_engine()
    
    if filter_test and integration_test:
        logger.info("\n🎉 All DXY correlation filter tests passed!")
        return 0
    else:
        logger.error("\n❌ Some tests failed")
        return 1

if __name__ == "__main__":
    try:
        # Run tests
        success = asyncio.run(main())
        
        if success == 0:
            logger.info("\n✨ DXY correlation filter implementation is working correctly")
        else:
            logger.error("\n❌ DXY correlation filter implementation has issues")
            
        sys.exit(success)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Test execution failed: {e}", exc_info=True)
        sys.exit(1)
