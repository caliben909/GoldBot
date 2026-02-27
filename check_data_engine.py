#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_engine import DataEngine
import yaml
from datetime import datetime

async def main():
    with open('config/config_optimized.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_engine = DataEngine(config)
    
    try:
        await data_engine.initialize()
        
        df = await data_engine.get_historical_data(
            'XAUUSD', 
            '1D', 
            start=datetime(2023, 1, 1), 
            end=datetime(2023, 6, 1)
        )
        
        print('Data loaded successfully')
        print(f'Number of bars: {len(df)}')
        print('\nFirst 10 bars:')
        print(df[['open', 'high', 'low', 'close']].head(10))
        
        print('\nLast 10 bars:')
        print(df[['open', 'high', 'low', 'close']].tail(10))
        
        print(f'\nPrice range: {df["low"].min():.2f} - {df["high"].max():.2f}')
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        await data_engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())