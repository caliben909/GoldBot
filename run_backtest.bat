@echo off
echo ================================================
echo Gold Currencies Bot - Backtesting
echo ================================================
echo.

REM Check if Python is available
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python not found in PATH
    echo Please install Python from https://www.python.org/
    echo.
    pause
    exit /b 1
)

echo Python found:
for /f "tokens=*" %%i in ('where python') do echo   %%i
echo.

REM Run backtest with optimized configuration
echo Running backtest on optimized strategy configuration...
echo.
echo Configuration: config\config_optimized.yaml
echo Timeframe: 1 hour
echo Symbols: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD, USDCHF, XAUUSD, XAGUSD
echo.

python "main.py" --mode backtest --config "config/config_optimized.yaml"

if %ERRORLEVEL% neq 0 (
    echo.
    echo Error: Backtest failed with error code %ERRORLEVEL%
    echo.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ================================================
echo Backtest completed successfully!
echo ================================================
echo.
echo Results saved to:
echo   - backtest/results/equity_curve.png
echo   - backtest/results/trade_journal.csv
echo   - backtest/results/performance_report.md
echo.
echo Analyze the results to verify:
echo   - Win rate distribution (target: 75-85%)
echo   - Profit factor (target: 2.5-3.5)
echo   - Risk metrics
echo.
pause
