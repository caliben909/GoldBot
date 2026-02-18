@echo off
echo ================================================
echo Gold Currencies Bot - Win Rate Optimization
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

REM Run the win rate optimization
echo Running win rate optimization...
echo.

python "win_rate_optimization.py"

if %ERRORLEVEL% neq 0 (
    echo.
    echo Error: Optimization failed with error code %ERRORLEVEL%
    echo.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ================================================
echo Optimization completed successfully!
echo ================================================
echo.
echo Next steps:
echo 1. Run backtest on optimized configuration:
echo    python main.py --mode backtest --config config/config_optimized.yaml
echo.
echo 2. Analyze the results and refine if needed
echo.
pause
