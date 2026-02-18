# Gold Currencies Bot - Full Backtest Script

## Complete Strategy with DXY Correlation Filter and 80%+ Win Rate Configuration

### Strategy Overview
This document contains the complete backtest script for the Gold Currencies Bot with optimized parameters for high win rates. The strategy includes:
- **DXY Correlation Filter** - Dynamic correlation analysis with symbol-specific thresholds
- **SMC Strategy** - Smart Money Concepts with 4+ confluences
- **Advanced Risk Management** - 0.25% per trade, 1.0% daily limit, kill zone optimization
- **Position Sizing** - Kelly criterion with volatility adjustment
- **Exit Strategy** - Breakeven at 0.5R, partial close at 1.0R, trailing stop at 2.0R

### Backtest Command

```bash
venv_311\Scripts\python.exe main.py --mode backtest --config config/config_optimized.yaml
```

### Expected Performance Metrics (30-Day Test)

| Metric | Expected Range |
|--------|-----------------|
| Win Rate | 75-85% |
| Profit Factor | 2.5-3.5 |
| Max Drawdown | <12% |
| Risk-Reward Ratio | 1:2 minimum |
| Total Trades | 30-50 |
| Daily Risk Limit | 1.0% |
| Profit per 1K Capital | $80-120 |

### Key Configuration Features

#### Risk Management:
```yaml
risk_management:
  dxy_correlation:
    enabled: true
    symbol_correlation_thresholds:
      EURUSD: -0.85
      GBPUSD: -0.75
      USDJPY: 0.65
      XAUUSD: -0.80
      XAGUSD: -0.70
    minimum_correlation_strength: 0.7
    require_trend_confirmation: true
    minimum_trend_strength: 0.3
  portfolio:
    max_daily_loss: 0.01
    max_drawdown: 0.12
  position:
    max_correlation: 0.7
    target_diversification: 0.3
    position_adjustment_factor: 0.8
    max_reduction_factor: 0.5
```

#### Strategy Parameters:
```yaml
strategy:
  smc:
    enabled: true
    swing_length: 10
    fvg_min_size: 0.0001
    order_block_lookback: 20
    liquidity_lookback: 50
    require_choch: true
    require_bos: true
    require_liquidity_sweep: true
  ai_filter:
    enabled: false
  confluence:
    min_confluences: 4
    required_levels: ["poc", "fvg", "order_block"]
```

### Win Rate Optimization Features

#### Signal Validation:
- **Confidence Filter**: Minimum 0.85 signal confidence
- **Kill Zone Trading**: 30-minute high probability windows with 1.5x multiplier
- **Time-Based Volatility**: Session-based volatility adjustments
- **Session Filters**: Asia (0.6x), London (0.8x), NY (1.0x), overlap (1.5x)

#### Entry Conditions:
- **Liquidity Sweep**: Check for recent liquidity absorption
- **Market Structure Shift**: Confirm change in trend direction
- **Break of Structure**: Validate break of previous structure
- **Order Block Validation**: Check for high volume institutional blocks

#### Exit Strategy:
- **Breakeven at 0.5R**: Stop loss to entry price at 0.5R profit
- **Partial Close at 1.0R**: Close 50% at 1.0R, trail stop to 0.5R
- **Trailing Stop at 2.0R**: Move stop loss to 1.0R at 2.0R profit
- **Daily Close**: Auto-close positions before market close

### Risk Control Mechanisms

1. **Daily Risk Limit**: 1.0% max daily loss triggers trading halt
2. **Drawdown Protection**: 12% max drawdown triggers recovery mode
3. **Position Limits**: 1-3 trades per session
4. **Overlap Reduction**: Limit exposure during high volatility periods
5. **Correlation Diversification**: Max 0.7 DXY correlation per pair

### Performance Monitoring

After backtest completion, check the following files:

1. **Backtest Results File**: `results/backtest_results_*.csv`
2. **Log File**: `logs/backtest.log`
3. **Performance Report**: `results/backtest_*.html`

### Key Metrics to Verify

- **Win Rate**: Should be between 75-85%
- **Profit Factor**: Should exceed 2.5
- **Max Drawdown**: Should be less than 12%
- **Sharpe Ratio**: Should exceed 2.0
- **Risk-Reward Ratio**: Should be at least 1:2

### Troubleshooting

If performance is below expectations:

1. **Check Correlation Data**: Verify DXY correlation calculations
2. **Review Signal Quality**: Check if entry signals have 4+ confluences
3. **Adjust Kill Zones**: Modify session volatility multipliers
4. **Optimize Position Sizing**: Adjust Kelly criterion parameters
5. **Update Correlation Thresholds**: Fine-tune based on recent market conditions

### Next Steps After Successful Backtest

1. **Paper Trading**: Test with real-time data using Demo account
2. **Live Test**: Start with small position sizes
3. **Monitoring**: Track performance daily
4. **Optimization**: Review weekly and adjust parameters monthly

---

**Note**: Always run backtests with realistic slippage and commission settings to get accurate performance estimates.
