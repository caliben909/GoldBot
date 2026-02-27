import asyncio
import pandas as pd
from core.data_engine import DataEngine
from core.liquidity_engine import LiquidityEngine

async def main():
    # Load config
    config = {
        'strategy': {
            'smc': {
                'swing_length': 2,
                'fvg_min_size': 0.1,
                'liquidity_lookback': 20,
                'order_block_lookback': 20
            }
        },
        'data_quality': {
            'sources': {
                'yfinance': {'enabled': True, 'priority': 1},
                'csv': {'enabled': False, 'priority': 2},
                'mt5': {'enabled': False, 'priority': 3},
                'binance': {'enabled': False, 'priority': 4}
            }
        },
        'execution': {
            'mt5': {'enabled': False},
            'binance': {'enabled': False}
        }
    }
    
    # Initialize data engine
    data_engine = DataEngine(config)
    await data_engine.initialize()
    
    # Download XAUUSD data
    df = await data_engine.get_historical_data(
        symbol='XAUUSD',
        timeframe='1d',
        start=pd.Timestamp('2024-01-01'),
        end=pd.Timestamp('2024-02-01')
    )
    
    # Initialize liquidity engine
    liquidity_engine = LiquidityEngine(config)
    
    # Analyze market
    analysis = liquidity_engine.analyze_market(df)
    
    # Print results
    print("=== Liquidity Engine Analysis ===")
    print(f"Data shape: {df.shape}")
    print()
    
    print("=== Swing Points ===")
    print(f"Swing highs: {len(analysis['structure']['swing_highs'])}")
    print(f"Swing lows: {len(analysis['structure']['swing_lows'])}")
    if analysis['structure']['swing_highs']:
        print(f"First swing high: {analysis['structure']['swing_highs'][0]}")
    if analysis['structure']['swing_lows']:
        print(f"First swing low: {analysis['structure']['swing_lows'][0]}")
    print()
    
    print("=== Order Blocks ===")
    print(f"Order blocks: {len(analysis['order_blocks'])}")
    if analysis['order_blocks']:
        print(f"First order block: {analysis['order_blocks'][0]}")
    print()
    
    print("=== FVG Zones ===")
    print(f"FVG zones: {len(analysis['fvg_zones'])}")
    if analysis['fvg_zones']:
        print(f"First FVG zone: {analysis['fvg_zones'][0]}")
    print()
    
    print("=== Liquidity ===")
    print(f"Buy-side liquidity: {len(analysis['liquidity']['buy_side'])}")
    print(f"Sell-side liquidity: {len(analysis['liquidity']['sell_side'])}")
    if analysis['liquidity']['buy_side']:
        print(f"First buy-side zone: {analysis['liquidity']['buy_side'][0]}")
    if analysis['liquidity']['sell_side']:
        print(f"First sell-side zone: {analysis['liquidity']['sell_side'][0]}")
    print()
    
    print("=== Trend ===")
    print(f"Current trend: {analysis['structure']['current_trend']}")
    print(f"Structure strength: {analysis['structure']['structure_strength']:.2f}")
    print()
    
    # Print data head for debugging
    print("=== Data Head ===")
    print(df.head())
    print()
    
    await data_engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
