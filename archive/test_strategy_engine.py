import asyncio
import pandas as pd
from core.data_engine import DataEngine
from core.liquidity_engine import LiquidityEngine
from core.strategy_engine import StrategyEngine
from core.risk_engine import RiskEngine

async def main():
    # Load config from config_optimized.yaml
    import yaml
    with open('config/config_optimized.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize engines
    data_engine = DataEngine(config)
    await data_engine.initialize()
    
    # Download data
    df = await data_engine.get_historical_data(
        symbol='XAUUSD',
        timeframe='4h',
        start=pd.Timestamp('2024-12-01'),
        end=pd.Timestamp('2025-01-01')
    )
    
    # Initialize liquidity and strategy engines
    liquidity_engine = LiquidityEngine(config)
    strategy_engine = StrategyEngine(config)
    
    # Initialize risk engine
    risk_engine = RiskEngine(config)
    strategy_engine.set_risk_engine(risk_engine)
    
    print(f"Data shape: {df.shape}")
    print()
    
    # Test liquidity engine
    print("=== Liquidity Engine Analysis ===")
    analysis = liquidity_engine.analyze_market(df)
    print(f"Order Blocks: {len(analysis['order_blocks'])}")
    if analysis['order_blocks']:
        print(f"First Order Block Type: {analysis['order_blocks'][0].direction}")
        print(f"First Order Block Price Range: {analysis['order_blocks'][0].price_range}")
    print(f"FVG Zones: {len(analysis['fvg_zones'])}")
    if analysis['fvg_zones']:
        print(f"First FVG Type: {analysis['fvg_zones'][0].type}")
        print(f"First FVG Size: {analysis['fvg_zones'][0].size_pips:.2f} pips")
    print(f"Buy-side Liquidity Zones: {len(analysis['liquidity']['buy_side'])}")
    print(f"Sell-side Liquidity Zones: {len(analysis['liquidity']['sell_side'])}")
    print(f"Current Trend: {analysis['structure']['current_trend']}")
    print()
    
    # Test strategy engine
    print("=== Strategy Engine Analysis ===")
    
    # Test regime detection
    regime, confidence = await strategy_engine.detect_market_regime(df)
    print(f"Detected Regime: {regime.value} (Confidence: {confidence:.2f})")
    
    # Test regime parameters
    regime_params = strategy_engine.get_regime_parameters(regime)
    print(f"Preferred Strategies: {regime_params['preferred_strategies']}")
    print(f"Min Confidence: {regime_params['min_confidence']:.2f}")
    
    # Test signal generation step by step
    liquidity_analysis = strategy_engine.liquidity_engine.analyze_market(df)
    contrarian_signals = strategy_engine._generate_contrarian_signals(df, liquidity_analysis, regime)
    print(f"Contrarian Signals Generated: {len(contrarian_signals)}")
    if contrarian_signals:
        for i, signal in enumerate(contrarian_signals):
            print(f"Signal {i+1}: Type={signal.signal_type.value}, "
                  f"Confidence={signal.confidence:.2f}, "
                  f"Direction={signal.direction}")
    
    # Test filtering
    filtered_signals = strategy_engine._apply_regime_filters(contrarian_signals, regime, regime_params)
    print(f"Contrarian Signals After Filtering: {len(filtered_signals)}")
    if filtered_signals:
        for i, signal in enumerate(filtered_signals):
            print(f"Signal {i+1}: Type={signal.signal_type.value}, "
                  f"Confidence={signal.confidence:.2f}, "
                  f"Direction={signal.direction}")
    
    # Test full signal generation
    signals = await strategy_engine.generate_trading_signals(df, 'XAUUSD')
    print(f"Final Signals Generated: {len(signals)}")
    if signals:
        for i, signal in enumerate(signals):
            print(f"\nSignal {i+1}:")
            print(f"  Type: {signal.signal_type.value}")
            print(f"  Direction: {signal.direction}")
            print(f"  Confidence: {signal.confidence:.2f}")
            print(f"  Regime: {signal.regime.value}")
            print(f"  Entry: {signal.entry_price:.2f}")
            print(f"  Stop Loss: {signal.stop_loss:.2f}")
            print(f"  Take Profit: {signal.take_profit:.2f}")
    print()
    
    # Test specific signal generation methods
    print("=== Testing Specific Signal Generators ===")
    bos_signals = strategy_engine._generate_bos_signals(df, analysis, strategy_engine.regime_history[-1][1] if strategy_engine.regime_history else None)
    print(f"BOS Signals: {len(bos_signals)}")
    
    choch_signals = strategy_engine._generate_choch_signals(df, analysis, strategy_engine.regime_history[-1][1] if strategy_engine.regime_history else None)
    print(f"CHOCH Signals: {len(choch_signals)}")
    
    fvg_signals = strategy_engine._generate_fvg_signals(df, analysis, strategy_engine.regime_history[-1][1] if strategy_engine.regime_history else None)
    print(f"FVG Signals: {len(fvg_signals)}")
    
    liquidity_signals = strategy_engine._generate_liquidity_signals(df, analysis, strategy_engine.regime_history[-1][1] if strategy_engine.regime_history else None)
    print(f"Liquidity Signals: {len(liquidity_signals)}")
    
    order_block_signals = strategy_engine._generate_order_block_signals(df, analysis, strategy_engine.regime_history[-1][1] if strategy_engine.regime_history else None)
    print(f"Order Block Signals: {len(order_block_signals)}")
    
    contrarian_signals = strategy_engine._generate_contrarian_signals(df, analysis, strategy_engine.regime_history[-1][1] if strategy_engine.regime_history else None)
    print(f"Contrarian Signals: {len(contrarian_signals)}")
    print()
    
    await data_engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
