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

## ✅ Final Checklist
- [ ] All pre-conditions met
- [ ] Paper trading successful for 2+ weeks
- [ ] Security settings verified
- [ ] Risk management configured correctly
- [ ] Emergency procedures in place
- [ ] Monitoring and alerts set up
- [ ] Broker accounts funded and verified
