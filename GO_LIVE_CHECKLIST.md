# Gold Currencies Bot - Go Live Checklist

## 🚨 Critical Pre-Conditions
- [ ] Bot version: Ensure you're running the latest stable version
- [ ] Environment: Set `environment: "production"` in `config.yaml`
- [ ] Configuration: All API keys, passwords, and secrets are properly set in `.env` file
- [ ] Backup: Complete backup of all strategy parameters, models, and configuration files

## 📊 Backtesting & Validation
- [ ] Walk-forward optimization completed on last 6-12 months of data
- [ ] Profit factor > 1.5, win rate > 40%, max drawdown < 10%
- [ ] Backtested on multiple market conditions (trending, ranging, volatile)
- [ ] Slippage and commission parameters matched actual broker conditions
- [ ] Stress testing completed with extreme market scenarios

## 🔧 Technical Setup
### Infrastructure
- [ ] VPS/dedicated server provisioned (low latency, high uptime)
- [ ] Operating system: Linux (preferred) or Windows Server
- [ ] Server security: Firewall, DDoS protection, regular updates
- [ ] Power backup and failover system configured

### Dependencies
- [ ] All Python dependencies installed: `pip install -r requirements.txt`
- [ ] MetaTrader 5 terminal installed (if trading Forex)
- [ ] Docker containerization tested (if using Docker)
- [ ] Network connectivity: Low latency to broker servers
- [ ] API rate limits configured properly

## 🔒 Security
### API Keys & Credentials
- [ ] API keys with limited permissions (read-only first, then trade permissions)
- [ ] API secret keys encrypted or stored in secure vault
- [ ] No hardcoded credentials in source code
- [ ] Two-factor authentication enabled for all broker accounts
- [ ] IP whitelisting configured where possible

### Data Security
- [ ] Encrypted communication with broker APIs (HTTPS/WebSocket SSL)
- [ ] Secure data storage for trade history and logs
- [ ] Regular backups of trading data and configuration

## 📈 Risk Management
### Initial Settings
- [ ] Max risk per trade: ≤ 1% of account equity
- [ ] Max daily risk: ≤ 3% of account equity
- [ ] Max drawdown limit: ≤ 10% of initial capital
- [ ] Position sizing method configured (ATR-based recommended)
- [ ] Scaling rules for consecutive losses enabled

### Broker-Specific Settings
- [ ] Leverage settings verified (≤ 1:10 for Forex, ≤ 1:20 for crypto)
- [ ] Margin call and stop-out levels configured
- [ ] Broker's margin requirements understood and accounted for

## 🚀 Live Trading Preparation
### Simulation & Paper Trading
- [ ] Paper trading on live market data for 2-4 weeks
- [ ] No significant deviations from backtest performance
- [ ] All order types tested (market, limit, stop, OCO)
- [ ] Slippage and execution time measured
- [ ] Error handling and recovery mechanisms tested

### Monitoring & Alerts
- [ ] TradingView or similar charting tool configured for manual monitoring
- [ ] Telegram/email notifications set up for trade executions and errors
- [ ] Server monitoring (CPU, RAM, disk space, network) configured
- [ ] Profit/loss alerts set at key levels

## 🎯 Broker Setup
### Forex Brokers (MT5)
- [ ] Account type: ECN/STP (no dealing desk)
- [ ] Spreads: ≤ 1 pip for major pairs
- [ ] Execution speed: ≤ 50ms
- [ ] Minimum deposit: $1000+ (professional account)
- [ ] Regulatory compliance: FCA, CySEC, ASIC, or similar

### Crypto Exchanges (Binance)
- [ ] Futures trading enabled (if using leverage)
- [ ] Trading fees: ≤ 0.1% per trade (maker/taker)
- [ ] Withdrawal limits: Sufficient for expected trading volume
- [ ] API rate limits: ≥ 1200 requests per minute
- [ ] Insurance fund coverage for futures trading

## 🔄 System Testing
### Order Execution
- [ ] Manual override functionality tested
- [ ] Emergency stop mechanism working
- [ ] Partial close and trailing stop logic tested
- [ ] Breakeven management verified
- [ ] Position sizing calculations correct

### Error Handling
- [ ] Network failure recovery tested
- [ ] API timeout and retry logic working
- [ ] Order rejection handling tested
- [ ] Position liquidation scenarios tested
- [ ] Database connection issues handled

## 📝 Documentation
### Trading Journal
- [ ] Trade journal template created
- [ ] Log format configured (JSON for easy analysis)
- [ ] Performance metrics tracking enabled
- [ ] Error and warning logs configured

### Strategy Documentation
- [ ] Strategy rules and parameters documented
- [ ] Risk management policy documented
- [ ] Emergency procedures manual created
- [ ] Contact information for support and brokers

## 🎮 Go Live Steps
### Phase 1: Soft Launch (1-2 Weeks)
1. [ ] Start with 50% of normal position size
2. [ ] Monitor every trade manually
3. [ ] Track performance against backtest results
4. [ ] Adjust risk parameters if needed

### Phase 2: Full Launch
1. [ ] Increase position size to 100%
2. [ ] Continue daily monitoring
3. [ ] Weekly performance reviews
4. [ ] Monthly strategy optimization

## 🚨 Emergency Procedures
- [ ] Immediate stop: `Ctrl+C` or kill switch
- [ ] Manual position closure: Through broker platform
- [ ] Server restart: Automated recovery script
- [ ] Contact support: Broker and bot developer contacts

## 📊 Post-Launch Monitoring
### Daily
- [ ] Review trade log and performance metrics
- [ ] Check for errors or anomalies
- [ ] Verify account balance and margin levels

### Weekly
- [ ] Compare live performance to backtest
- [ ] Analyze winning/losing trades
- [ ] Check for strategy decay or market regime changes

### Monthly
- [ ] Full performance report generation
- [ ] Strategy optimization if needed
- [ ] Risk parameter review

## 🔧 Strategy Engine Configuration Check
### Core Strategy Parameters
- [ ] Swing length parameter validated (2-50 periods)
- [ ] FVG minimum size parameter configured (positive value ≤ 100)
- [ ] Liquidity lookback period verified (5-200 periods)
- [ ] Preferred strategy list for each market regime validated

### Market Regime Detection
- [ ] Regime detection method configured (hmm/clustering/volatility_based/rule_based)
- [ ] Regime lookback periods validated (10-500 periods)
- [ ] Regime update frequency set (1-100 periods)
- [ ] Regime-specific minimum confidence thresholds configured
- [ ] Position size multipliers per regime verified
- [ ] Stop loss/take profit multipliers per regime configured
- [ ] Max trades per regime setting validated

### Strategy Signals & Confluence
- [ ] BOS (Break of Structure) signal parameters checked
- [ ] CHOCH (Change of Character) signal logic verified
- [ ] Liquidity sweep signal parameters validated
- [ ] FVG (Fair Value Gap) entry rules configured
- [ ] Order block bounce signal settings checked
- [ ] Contrarian signal logic verified
- [ ] Multi-timeframe signal confirmation enabled
- [ ] Kill zone detection parameters validated

### SMC (Smart Money Concepts) Configuration
- [ ] Swing point detection algorithm validated
- [ ] Order block identification parameters configured
- [ ] FVG zone detection and mitigation logic checked
- [ ] Liquidity pool detection settings verified
- [ ] OTE (Optimal Trade Entry) zone parameters configured (62%/79% levels)
- [ ] Market structure analysis rules validated

## 🛡️ Risk Management Configuration Check
### Core Risk Parameters
- [ ] Risk management model selected (Kelly/ATR/Fixed/Optimal_F/VaR)
- [ ] Position sizing method configured (fixed_risk/percentage_risk/Kelly/optimal_f/atr_based/volatility_adjusted)
- [ ] Risk per trade limit set (≤ 10% of account equity)
- [ ] Daily/weekly/monthly drawdown limits configured
- [ ] Maximum consecutive losses limit set
- [ ] Profit target and stop loss logic verified

### DXY Correlation Filter
- [ ] DXY correlation filter enabled/disabled correctly
- [ ] Correlation method selected (Pearson/Spearman)
- [ ] Correlation lookback period configured (10-200 periods)
- [ ] Minimum/maximum correlation strength thresholds validated (0-1)
- [ ] Position size adjustment based on correlation enabled
- [ ] Extreme correlation event detection configured

### Portfolio Risk Management
- [ ] Total exposure limits per symbol/sector configured
- [ ] Leverage limits per asset class verified
- [ ] Margin requirements and stop-out levels understood
- [ ] Position concentration limits per symbol/sector set
- [ ] VaR (Value at Risk) and Expected Shortfall calculations configured
- [ ] Stress testing parameters validated

### Trade Risk Controls
- [ ] Risk-reward ratio minimum requirements (≥1.5:1 recommended)
- [ ] Stop loss placement rules configured (ATR-based recommended)
- [ ] Take profit target calculations verified
- [ ] Breakeven management logic enabled
- [ ] Trailing stop parameters configured
- [ ] Partial close optimization rules validated

## 📊 Market Regime Detection Calibration
### Feature Extraction
- [ ] Volatility calculation parameters validated
- [ ] Trend strength measurement (linear regression) configured
- [ ] Volume profile analysis settings verified
- [ ] Range width calculation parameters checked
- [ ] ADX (Average Directional Index) period set (14-period standard)
- [ ] RSI (Relative Strength Index) period configured (14-period standard)
- [ ] MACD histogram calculation parameters verified
- [ ] Bollinger Bands width measurement settings checked
- [ ] ATR percentage calculation configured
- [ ] Hurst exponent and fractal dimension parameters validated

### Regime Classification
- [ ] Rule-based regime detection thresholds calibrated
- [ ] Clustering algorithm parameters configured
- [ ] HMM (Hidden Markov Model) states and transition probabilities validated
- [ ] Volatility-based regime classification percentiles set
- [ ] Regime transition detection sensitivity configured
- [ ] Confidence score calculation for each regime verified

### Regime-Specific Parameters
- [ ] Trending bullish/bearish regime parameters calibrated
- [ ] Ranging market regime parameters verified
- [ ] Volatile market regime parameters configured
- [ ] Quiet market regime parameters validated
- [ ] Breakout regime parameters checked
- [ ] Reversal regime parameters verified
- [ ] Accumulation/distribution regime parameters configured

### Performance Tracking
- [ ] Regime performance metrics tracking enabled
- [ ] Win rate per regime measured and validated
- [ ] Profit factor per regime calculated
- [ ] Average R-multiple per regime verified
- [ ] Best/worst performing regimes identified
- [ ] Regime transition impact on performance analyzed

## ⏰ Session Management Configuration
### Trading Sessions
- [ ] Asia session start/end times configured
- [ ] London session start/end times validated
- [ ] New York session start/end times verified
- [ ] London-NY overlap session parameters configured
- [ ] Pre-Asia/Post-NY session settings checked
- [ ] Session timezone settings (UTC/Asia/Tokyo/Europe/London/America/New_York) validated

### Session Parameters
- [ ] Volatility multipliers per session configured
- [ ] Minimum/maximum volatility thresholds per session set
- [ ] Strategy bias per session selected (trend/range/momentum/scalp/breakout/mean_reversion/neutral)
- [ ] Preferred currency pairs per session configured
- [ ] Excluded currency pairs per session verified
- [ ] Risk multipliers per session validated
- [ ] Maximum trades per session limit set
- [ ] Minimum confidence threshold per session configured

### Kill Zones (High-Probability Zones)
- [ ] London Open kill zone (08:00-09:00 GMT) configured
- [ ] NY Open kill zone (13:00-14:00 GMT) validated
- [ ] London-NY overlap kill zone (13:00-16:00 GMT) verified
- [ ] Tokyo Open kill zone (00:00-01:00 GMT) checked
- [ ] London Lunch kill zone (12:00-13:00 GMT) parameters validated
- [ ] Kill zone probability scores and win rates configured

### Adaptive Parameters
- [ ] Adaptive parameter tuning enabled/disabled
- [ ] Lookback period for adaptive adjustments configured
- [ ] Volatility threshold adaptation logic verified
- [ ] Confidence threshold adaptation based on win rate enabled
- [ ] Risk multiplier adjustment based on session performance configured
- [ ] Parameter bounds validation logic checked

### Market Hours & Holidays
- [ ] Weekend detection logic validated
- [ ] Major holiday list configured
- [ ] News event detection enabled (NFP/FOMC/CPI times)
- [ ] News avoidance windows (±30 minutes) configured
- [ ] Market closure detection for holidays verified

## 🎯 Consistency Verification
### Configuration Consistency
- [ ] All config.yaml settings match intended values
- [ ] .env file variables correctly mapped to configuration
- [ ] Docker environment variables match production settings
- [ ] Configuration version control tags verified
- [ ] Configuration backup completed

### Parameter Consistency
- [ ] Strategy engine parameters match backtest settings
- [ ] Risk management parameters consistent across environments
- [ ] Session management settings identical in paper/live trading
- [ ] Market regime detection parameters synchronized
- [ ] All components using same configuration source

### Backtest vs Live Consistency
- [ ] Slippage parameters match broker's live conditions
- [ ] Commission and fee calculations consistent
- [ ] Leverage settings identical in backtest and live
- [ ] Position sizing logic matches across environments
- [ ] Stop loss/take profit placement identical

### Data Consistency
- [ ] Historical data source matches live data feed
- [ ] Data granularity and timeframe consistency verified
- [ ] DXY correlation data source configured correctly
- [ ] Volume and volatility data consistency checked
- [ ] Market structure analysis on live data matches backtest

### Performance Consistency
- [ ] Paper trading performance within backtest ranges
- [ ] Win rate and profit factor consistent over 2+ weeks
- [ ] Drawdown levels within expected ranges
- [ ] Risk-reward ratios match backtest expectations
- [ ] Trade frequency and timing consistent

## ✅ Final Checklist
- [ ] All pre-conditions met
- [ ] Paper trading successful for 2+ weeks
- [ ] Security settings verified
- [ ] Risk management configured correctly
- [ ] Emergency procedures in place
- [ ] Monitoring and alerts set up
- [ ] Broker accounts funded and verified
- [ ] All strategy engine configurations validated
- [ ] Risk management system properly calibrated
- [ ] Market regime detection parameters optimized
- [ ] Session management settings configured
- [ ] Configuration consistency verified
