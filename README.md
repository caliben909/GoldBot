# Institutional Trading Bot

A production-grade, modular, scalable trading bot with institutional-level risk management and smart money logic.

## Features

### Core Modules

1. **Session Engine** - Detects Asia, London, New York sessions with dynamic volatility thresholds and strategy bias per session.

2. **Smart Money Strategy Engine** - Implements advanced trading concepts:
   - Market Structure Break (BOS)
   - Change of Character (CHOCH)
   - Liquidity sweeps
   - Fair Value Gaps (FVG)
   - Order Blocks
   - Kill zone logic

3. **AI Filter** - ONNX model integration with features: RSI, ATR, Volume, Structure strength. Only takes trades above 0.70 confidence.

4. **Risk Engine** - Institutional-level risk management:
   - Dynamic lot sizing based on equity and ATR volatility
   - Max 1% risk per trade
   - Daily drawdown limit: 3%
   - Weekly shutdown: 6%
   - Auto scale down after 2 losses
   - Breakeven after 1R
   - Partial close at 2R
   - Trail at 3R

5. **Execution Engine** - MT5 and Binance Futures API support with slippage tolerance, spread filter, and news event avoidance.

6. **Backtest Engine** - Historical OHLC loading, strategy simulation, equity curve plotting, Sharpe ratio, win rate, max drawdown, profit factor.

7. **Logging** - Structured logging, trade journal CSV output, and error handling.

## Installation

### Prerequisites

- Python 3.11+
- pip package manager
- Git

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd trading-bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API credentials in `config/config.yaml`:
```yaml
binance:
  api_key: 'YOUR_BINANCE_API_KEY'
  api_secret: 'YOUR_BINANCE_API_SECRET'

mt5:
  login: 12345678
  password: 'YOUR_MT5_PASSWORD'
  server: 'MetaQuotes-Demo'
```

## Usage

### Test Imports

First, verify all modules are importable:
```bash
cd multi_asset_trading_bot
python test_imports.py
```

### Backtesting

Run backtest on default configuration:
```bash
python main.py --mode backtest
```

Run backtest on specific symbol:
```bash
python main.py --mode backtest --symbol EURUSD
```

### Live Trading

Run live trading:
```bash
python main.py --mode live
```

### Forward Testing

Run forward testing (simulated live conditions):
```bash
python main.py --mode forward
```

## Configuration

All settings are defined in `config/config.yaml`. Key sections:

- **general**: Mode selection, logging settings, data source
- **backtest**: Initial equity, date range, symbols
- **risk**: Risk management parameters
- **strategy**: Strategy settings, AI filter
- **ai**: Model and scaler paths
- **binance**: Binance API configuration
- **mt5**: MT5 connection details
- **performance**: Target performance metrics

## Performance Targets

Aim for:
- Win rate > 85%
- RR average >= 1:5
- Max drawdown < 10%

## File Structure

```
trading-bot/
├── core/
│   ├── data_engine.py        # Data loading and preprocessing
│   ├── execution_engine.py   # Trade execution
│   ├── risk_engine.py        # Risk management
│   ├── strategy_engine.py    # Smart money strategy
│   ├── session_engine.py     # Session detection
│   ├── ai_engine.py          # AI integration
│   └── liquidity_engine.py   # Liquidity analysis
├── config/
│   └── config.yaml           # Configuration file
├── utils/
│   ├── backtest_engine.py    # Backtesting framework
│   └── logging_config.py     # Logging configuration
├── backtest/                 # Backtest results
├── logs/                     # Log files
├── models/                   # AI models and scalers
├── live/                     # Live trading logs
├── main.py                   # Main entry point
└── requirements.txt          # Dependencies
```

## VS Code Setup

1. Open VS Code and install the Python extension
2. Create a new Python environment using the `venv` module
3. Install dependencies using `pip install -r requirements.txt`
4. Configure your API credentials in `config/config.yaml`
5. Run backtest: Right-click on `main.py` and select "Run Python File"
6. View results in `backtest/backtest_results.png` and CSV files

## Docker Support

Build Docker image:
```bash
docker build -t trading-bot .
```

Run container:
```bash
docker run -v $(pwd)/config:/app/config -v $(pwd)/logs:/app/logs -v $(pwd)/backtest:/app/backtest trading-bot --mode backtest
```

## Development

### Adding New Features

1. Create new module in appropriate directory
2. Extend existing base classes
3. Update configuration in `config.yaml`
4. Add tests in `tests/` directory

### Testing

Run tests:
```bash
pytest tests/
```

Run coverage report:
```bash
pytest tests/ --cov=core --cov-report=html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Add tests
5. Submit pull request

## License

MIT License

## Disclaimer

This software is for educational purposes only. Trading involves significant risk. The author is not responsible for any financial losses.

Always start with a demo account and test your strategies thoroughly before trading with real money.
