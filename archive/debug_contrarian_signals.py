import asyncio
import pandas as pd
import numpy as np
import logging
import yaml
from datetime import datetime
from core.data_engine import DataEngine
from core.strategy_engine import StrategyEngine
from core.risk_engine import RiskEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    # Load configuration
    with open('config/config_optimized.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize data engine
    data_engine = DataEngine(config)
    
    await data_engine.initialize()

    # Get XAUUSD data
    df = await data_engine.get_historical_data(
        symbol="XAUUSD",
        timeframe="1D",
        start=datetime(2023, 1, 1),
        end=datetime(2023, 6, 1)
    )
    logger.info(f"Loaded {len(df)} days of data")

    strategy_engine = StrategyEngine(config)
    risk_engine = RiskEngine(config)
    strategy_engine.set_risk_engine(risk_engine)

    # Debug: Check the first few days of data
    logger.info("First 5 days of data:")
    print(df.head())

    # Debug: Generate signals for the first 30 days
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
                
                logger.debug(f"Signal: {signal.signal_type} - {signal.direction} - {sub_df.index[-1]}")
                
        except Exception as e:
            logger.error(f"Error generating signals for {sub_df.index[-1]}: {e}")
            continue

    # Convert signals to DataFrame
    signals_df = pd.DataFrame(signals)
    logger.info(f"Generated {len(signals_df)} signals")

    # Analyze signals
    logger.info("\nSignal Analysis:")
    logger.info("----------------")
    logger.info(f"Signal Types: {signals_df['signal_type'].value_counts()}")
    logger.info(f"Directions: {signals_df['direction'].value_counts()}")
    logger.info(f"Regimes: {signals_df['regime'].value_counts()}")

    # Filter contrarian signals
    contrarian_signals = signals_df[signals_df['signal_type'] == 'contrarian']
    logger.info(f"\nContrarian Signals: {len(contrarian_signals)}")
    if len(contrarian_signals) > 0:
        logger.info(f"Contrarian Directions: {contrarian_signals['direction'].value_counts()}")
        logger.info(f"Contrarian Regimes: {contrarian_signals['regime'].value_counts()}")
        logger.info("\nFirst 10 Contrarian Signals:")
        print(contrarian_signals.head(10))

    # Plot price and signals
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        plt.figure(figsize=(15, 8))
        plt.plot(df['close'], label='XAUUSD')
        
        # Plot signals
        if len(signals_df) > 0:
            # Buy signals
            buy_signals = signals_df[signals_df['direction'].isin(['long', 'bullish'])]
            buy_dates = [mdates.date2num(pd.to_datetime(t)) for t in buy_signals['timestamp']]
            plt.scatter(buy_dates, df.loc[buy_signals['timestamp']]['close'], 
                       marker='^', color='green', label='Buy/Signal', s=100)
            
            # Sell signals
            sell_signals = signals_df[signals_df['direction'].isin(['short', 'bearish'])]
            sell_dates = [mdates.date2num(pd.to_datetime(t)) for t in sell_signals['timestamp']]
            plt.scatter(sell_dates, df.loc[sell_signals['timestamp']]['close'], 
                       marker='v', color='red', label='Sell/Signal', s=100)
        
        plt.title('XAUUSD Price and Signals (First 6 Months 2023)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting: {e}")

    logger.info("\nDebug complete")

if __name__ == "__main__":
    asyncio.run(main())
