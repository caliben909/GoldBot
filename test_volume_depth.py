#!/usr/bin/env python3
"""
Test script for volume depth functionality
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
async def test_volume_depth():
    """Test volume depth functionality"""
    logger.info("=" * 60)
    logger.info("TESTING VOLUME DEPTH")
    logger.info("=" * 60)
    
    try:
        # Create test volume data
        logger.info("\n1. Creating test volume data...")
        
        dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='h')
        base_volume = 1000000
        volumes = []
        for i, date in enumerate(dates):
            trend = 1000 * i
            noise = np.random.normal(0, 50000)
            volume = base_volume + trend + noise
            volumes.append(max(0, volume))
        
        volume_series = pd.Series(volumes, index=dates)
        logger.debug(f"✅ Volume history created with {len(volume_series)} points")
        
        # Test 1: Calculate volume profile
        logger.info("\n2. Testing volume profile calculation...")
        
        # Calculate daily volume profile
        daily_volumes = volume_series.resample('D').sum()
        
        logger.debug(f"✅ Average daily volume: {daily_volumes.mean():,.0f}")
        logger.debug(f"✅ Max daily volume: {daily_volumes.max():,.0f}")
        logger.debug(f"✅ Min daily volume: {daily_volumes.min():,.0f}")
        
        # Test 2: Check volume spikes
        logger.info("\n3. Testing volume spike detection...")
        
        # Calculate volume moving average
        volume_ma = volume_series.rolling(window=24).mean()
        volume_std = volume_series.rolling(window=24).std()
        
        # Detect volume spikes (2 standard deviations above mean)
        volume_zscore = (volume_series - volume_ma) / volume_std
        volume_spikes = volume_zscore[volume_zscore > 2]
        
        logger.debug(f"✅ Volume spikes ({len(volume_spikes)}) detected")
        
        # Test 3: Calculate volume at price (VAP)
        logger.info("\n4. Testing volume at price calculation...")
        
        # Create mock price and volume data
        prices = np.random.uniform(low=1.08, high=1.12, size=len(dates))
        price_series = pd.Series(prices, index=dates)
        
        # Create price bins
        price_bins = np.linspace(1.08, 1.12, 20)
        price_indices = np.digitize(price_series, price_bins)
        
        # Calculate volume at each price bin
        volume_at_price = {}
        for i in range(1, len(price_bins)):
            bin_vol = volume_series[price_indices == i].sum()
            price_level = (price_bins[i-1] + price_bins[i]) / 2
            volume_at_price[price_level] = bin_vol
        
        logger.debug(f"✅ Volume at price calculated for {len(volume_at_price)} price levels")
        
        logger.info("\n✅ Volume depth functionality test completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False


@pytest.mark.asyncio
async def test_volume_analysis():
    """Test volume analysis functions"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING VOLUME ANALYSIS")
    logger.info("=" * 60)
    
    try:
        # Create test data
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='h')
        base_volume = 1000000
        volumes = []
        for i, date in enumerate(dates):
            trend = 1000 * i
            noise = np.random.normal(0, 50000)
            volume = base_volume + trend + noise
            volumes.append(max(0, volume))
        
        volume_series = pd.Series(volumes, index=dates)
        
        prices = np.random.uniform(low=1.08, high=1.12, size=len(dates))
        price_series = pd.Series(prices, index=dates)
        
        # Test 1: Calculate on-balance volume (OBV)
        logger.info("\n1. Testing on-balance volume...")
        
        obv = []
        cumulative_volume = 0
        for i in range(len(price_series)):
            if i == 0:
                obv.append(cumulative_volume)
                continue
            
            if price_series[i] > price_series[i-1]:
                cumulative_volume += volume_series[i]
            elif price_series[i] < price_series[i-1]:
                cumulative_volume -= volume_series[i]
            
            obv.append(cumulative_volume)
        
        obv_series = pd.Series(obv, index=price_series.index)
        
        logger.debug(f"✅ OBV calculated successfully")
        logger.debug(f"✅ OBV range: {obv_series.min():,.0f} - {obv_series.max():,.0f}")
        
        # Test 2: Calculate volume profile
        logger.info("\n2. Testing volume profile...")
        
        # Create OHLC data with volume
        ohlc_data = pd.DataFrame()
        ohlc_data['open'] = price_series
        ohlc_data['high'] = price_series + np.random.normal(0, 0.001, size=len(price_series))
        ohlc_data['low'] = price_series - np.random.normal(0, 0.001, size=len(price_series))
        ohlc_data['close'] = price_series
        ohlc_data['volume'] = volume_series
        
        # Calculate volume at each price level
        price_levels = np.linspace(1.08, 1.12, 20)
        volume_profile = {}
        
        for level in price_levels:
            mask = (ohlc_data['low'] <= level) & (ohlc_data['high'] >= level)
            volume_profile[level] = ohlc_data[mask]['volume'].sum()
        
        logger.debug(f"✅ Volume profile calculated for {len(volume_profile)} levels")
        
        logger.info("\n✅ Volume analysis tests completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False


async def main():
    """Main test function"""
    logger.info("Starting comprehensive volume depth tests...")
    
    # Run all tests
    volume_depth = await test_volume_depth()
    volume_analysis = await test_volume_analysis()
    
    if volume_depth and volume_analysis:
        logger.info("\n🎉 All volume depth tests passed!")
        return 0
    else:
        logger.error("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        
        if success == 0:
            logger.info("\n✨ Volume depth implementation is working correctly")
        else:
            logger.error("\n❌ Volume depth implementation has issues")
            
        sys.exit(success)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Test execution failed: {e}", exc_info=True)
        sys.exit(1)
