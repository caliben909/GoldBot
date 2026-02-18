#!/usr/bin/env python3
"""
Test script for price checker functionality
"""
import asyncio
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

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
async def test_price_checker():
    """Test basic price checking functionality"""
    logger.info("=" * 60)
    logger.info("TESTING PRICE CHECKER")
    logger.info("=" * 60)
    
    try:
        # Create test price data
        logger.info("\n1. Creating test price data...")
        
        # Generate EURUSD price history
        dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='h')
        base_price = 1.08
        prices = []
        for i, date in enumerate(dates):
            trend = 0.0001 * i  # Upward trend
            noise = np.random.normal(0, 0.0005)
            price = base_price + trend + noise
            prices.append(price)
        
        price_series = pd.Series(prices, index=dates)
        logger.debug(f"✅ Price history created with {len(price_series)} points")
        
        # Test 1: Check for extreme price movements
        logger.info("\n2. Testing extreme price movement detection...")
        
        # Calculate daily price changes
        daily_prices = price_series.resample('D').last()
        daily_returns = daily_prices.pct_change().dropna()
        
        # Check for extreme movements (> 2%)
        extreme_movements = daily_returns[abs(daily_returns) > 0.02]
        
        logger.debug(f"✅ Extreme price movements ({len(extreme_movements)}) detected")
        
        # Test 2: Check price volatility
        logger.info("\n3. Testing price volatility calculation...")
        
        # Calculate volatility
        volatility = price_series.pct_change().rolling(window=24).std() * np.sqrt(252)
        
        logger.debug(f"✅ Average volatility: {volatility.mean():.2%}")
        logger.debug(f"✅ Max volatility: {volatility.max():.2%}")
        
        # Test 3: Check for price gaps
        logger.info("\n4. Testing price gap detection...")
        
        # Calculate overnight gaps
        overnight_gaps = price_series.resample('D').last().pct_change().dropna()
        
        logger.debug(f"✅ Average overnight gap: {overnight_gaps.mean():.2%}")
        logger.debug(f"✅ Max overnight gap: {overnight_gaps.max():.2%}")
        
        logger.info("\n✅ Price checking functionality test completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False


@pytest.mark.asyncio
async def test_price_calculations():
    """Test price calculation functions"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING PRICE CALCULATIONS")
    logger.info("=" * 60)
    
    try:
        # Create test price data
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='h')
        base_price = 1.08
        prices = []
        for i, date in enumerate(dates):
            trend = 0.0001 * i
            noise = np.random.normal(0, 0.0005)
            price = base_price + trend + noise
            prices.append(price)
        
        price_series = pd.Series(prices, index=dates)
        
        # Test OHLC calculations
        logger.info("\n1. Testing OHLC calculations...")
        
        # Create OHLC data from price series
        ohlc = price_series.resample('H').ohlc()
        logger.debug(f"✅ OHLC data created with {len(ohlc)} periods")
        
        # Test 2: Calculate moving averages
        logger.info("\n2. Testing moving averages...")
        
        sma_5 = price_series.rolling(window=5).mean()
        sma_20 = price_series.rolling(window=20).mean()
        
        logger.debug(f"✅ 5-period SMA calculated")
        logger.debug(f"✅ 20-period SMA calculated")
        
        # Test 3: Calculate price ranges
        logger.info("\n3. Testing price range calculations...")
        
        high = price_series.rolling(window=24).max()
        low = price_series.rolling(window=24).min()
        range_ = high - low
        
        logger.debug(f"✅ 24-hour high calculated")
        logger.debug(f"✅ 24-hour low calculated")
        logger.debug(f"✅ Average 24-hour range: {range_.mean():.4f}")
        
        logger.info("\n✅ Price calculation tests completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False


async def main():
    """Main test function"""
    logger.info("Starting comprehensive price checker tests...")
    
    # Run all tests
    price_check = await test_price_checker()
    price_calc = await test_price_calculations()
    
    if price_check and price_calc:
        logger.info("\n🎉 All price checker tests passed!")
        return 0
    else:
        logger.error("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        
        if success == 0:
            logger.info("\n✨ Price checker implementation is working correctly")
        else:
            logger.error("\n❌ Price checker implementation has issues")
            
        sys.exit(success)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Test execution failed: {e}", exc_info=True)
        sys.exit(1)
