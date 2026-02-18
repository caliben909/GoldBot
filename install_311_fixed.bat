@echo off
echo ================================================
echo Installing Gold Currencies Bot (Python 3.11.9)
echo ================================================
echo.

echo Activating virtual environment...
call venv_311\Scripts\activate.bat

echo.
echo Installing Python 3.11 compatible dependencies...
python -m pip install -r requirements_311.txt

if %ERRORLEVEL% neq 0 (
    echo.
    echo WARNING: Some dependencies may have failed to install!
    echo.
)

echo.
echo ================================================
echo Installation Complete!
echo ================================================
echo.

REM Verify installation
echo Verifying core dependencies...
python -c "import sys, numpy, pandas, yaml; print('Python: ' + sys.version); print('NumPy: ' + numpy.__version__); print('Pandas: ' + pandas.__version__); print('YAML: ' + yaml.__version__); print('All core dependencies loaded successfully!')"

if %ERRORLEVEL% equ 0 (
    echo.
    echo ================================================
    echo SUCCESS: Bot is ready to run!
    echo ================================================
    echo.
    echo To run the backtest:
    echo   venv_311\Scripts\python.exe main.py --mode backtest --config config/config_optimized.yaml
    echo.
) else (
    echo.
    echo WARNING: Core dependencies verification failed!
    echo Please check the installation logs above.
)

echo.
echo To deactivate virtual environment:
echo   deactivate
echo.

pause
