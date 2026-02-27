#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_engine import DataEngine
from core.strategy_engine import StrategyEngine
from core.risk_engine import RiskEngine
import yaml
from datetime import datetime

async def debug_entry_price():
    with open('config/config_optimized.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_engine = DataEngine(config)
    strategy_engine = StrategyEngine(config)
    risk_engine = RiskEngine(config)
    
    strategy_engine.set_risk_engine(risk_engine)
    
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
        print(f'Price range: {df["low"].min():.2f} - {df["high"].max():.2f}')
        
        # Test signal generation for first 30 days
        signals = []
        for i in range(30, len(df)):
            sub_df = df.iloc[:i+1]
            
            # Generate signals
            try:
                daily_signals = await strategy_engine.generate_trading_signals(sub_df, symbol="XAUUSD")
                
                for signal in daily_signals:
                    signals.append({
                        'timestamp': sub_df.index[-1],
                        'signal_type': signal.signal_type.value,
                        'direction': signal.direction,
                        'entry_price': signal.entry_price,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'regime': signal.regime.value
                    })
                    
                    print(f"Signal at {sub_df.index[-1]}: "
                          f"{signal.signal_type} - {signal.direction} - "
                          f"Entry: {signal.entry_price:.2f} - "
                          f"SL: {signal.stop_loss:.2f} - "
                          f"TP: {signal.take_profit:.2f}")
                    
            except Exception as e:
                print(f"Error generating signals for {sub_df.index[-1]}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        await data_engine.shutdown()

if __name__ == "__main__":
    asyncio.run(debug_entry_price())