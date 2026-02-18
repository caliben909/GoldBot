@echo off
echo ================================================
echo Running Gold Currencies Bot Backtest (Python 3.11)
echo ================================================
echo.

call venv_311\Scripts\activate.bat

echo.
echo Checking dependencies...
python -c "import sys, numpy, pandas, yaml; print('Python:', sys.version)" >nul 2>&1

if %ERRORLEVEL% neq 0 (
    echo ERROR: Dependencies not installed or virtual environment not activated!
    echo Please run test_311.bat first.
    pause
    exit /b 1
)

echo Dependencies check passed!

echo.
echo Running backtest with optimized configuration...
echo.
python main.py --mode backtest --config config/config_optimized.yaml

if %ERRORLEVEL% equ 0 (
    echo.
    echo ================================================
    echo SUCCESS: Backtest completed successfully!
    echo ================================================
    echo.
    echo Results saved to:
    echo   backtest/results/equity_curve.png
    echo   backtest/results/trade_journal.csv
    echo   backtest/results/performance_report.md
    echo.
    echo Key metrics to check:
    echo   - Win rate should be between 75-85%
    echo   - Profit factor should be between 2.5-3.5
    echo   - Max drawdown should be less than 12%
    echo.
    echo To view detailed report:
    echo   Notepad backtest/results/performance_report.md
) else (
    echo.
    echo ERROR: Backtest failed!
    echo Please check the error message and try again.
)

echo.
echo To deactivate virtual environment:
echo   deactivate

pause
