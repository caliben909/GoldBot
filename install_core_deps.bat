@echo off
echo ================================================
echo Installing Core Dependencies for Python 3.14
echo ================================================
echo.

REM Try to install core dependencies directly
echo Installing core dependencies...
"C:\Users\Mixxmasterz\AppData\Local\Programs\Python\Python314\python.exe" -m pip install --upgrade pip

if %ERRORLEVEL% neq 0 (
    echo.
    echo WARNING: Pip upgrade failed!
    echo Trying to install without upgrade...
    echo.
)

echo.
echo Installing numpy, pandas, pydantic, pyyaml, python-dotenv...
"C:\Users\Mixxmasterz\AppData\Local\Programs\Python\Python314\python.exe" -m pip install numpy pandas pydantic pyyaml python-dotenv

if %ERRORLEVEL% neq 0 (
    echo.
    echo WARNING: Core dependencies installation failed!
    echo.
    pause
    exit /b 1
)

echo.
echo Installing technical analysis libraries...
"C:\Users\Mixxmasterz\AppData\Local\Programs\Python\Python314\python.exe" -m pip install ta pandas-ta

if %ERRORLEVEL% neq 0 (
    echo.
    echo WARNING: Technical analysis libraries installation failed!
    echo.
)

echo.
echo Installing backtesting libraries...
"C:\Users\Mixxmasterz\AppData\Local\Programs\Python\Python314\python.exe" -m pip install vectorbt backtrader

if %ERRORLEVEL% neq 0 (
    echo.
    echo WARNING: Backtesting libraries installation failed!
    echo.
)

echo.
echo Installing visualization libraries...
"C:\Users\Mixxmasterz\AppData\Local\Programs\Python\Python314\python.exe" -m pip install matplotlib plotly seaborn

if %ERRORLEVEL% neq 0 (
    echo.
    echo WARNING: Visualization libraries installation failed!
    echo.
)

echo.
echo Installing API and networking libraries...
"C:\Users\Mixxmasterz\AppData\Local\Programs\Python\Python314\python.exe" -m pip install aiohttp requests

if %ERRORLEVEL% neq 0 (
    echo.
    echo WARNING: API libraries installation failed!
    echo.
)

echo.
echo ================================================
echo Core Dependencies Installation Complete!
echo ================================================
echo.

REM Verify installation
echo Verifying core dependencies...
"C:\Users\Mixxmasterz\AppData\Local\Programs\Python\Python314\python.exe" -c "import sys, numpy, pandas, yaml; print('Python: ' + sys.version); print('NumPy: ' + numpy.__version__); print('Pandas: ' + pandas.__version__); print('YAML: ' + yaml.__version__); print('All core dependencies loaded successfully!')"

if %ERRORLEVEL% equ 0 (
    echo.
    echo ================================================
    echo SUCCESS: Bot is ready to run!
    echo ================================================
    echo.
    echo To run the backtest:
    echo   "C:\Users\Mixxmasterz\AppData\Local\Programs\Python\Python314\python.exe" main.py --mode backtest --config config/config_optimized.yaml
    echo.
) else (
    echo.
    echo WARNING: Core dependencies verification failed!
    echo Please check the installation logs above.
)

echo.
pause
