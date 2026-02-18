@echo off
echo ================================================
echo Testing Python 3.11 Environment
echo ================================================
echo.

call venv_311\Scripts\activate.bat

echo Checking Python version...
python --version

echo.
echo Checking core dependencies...
python -c "
import sys
import numpy
import pandas
import yaml
import os

print('Python:', sys.version)
print('NumPy:', numpy.__version__)
print('Pandas:', pandas.__version__)
print('YAML:', yaml.__version__)

print('\nInstallation successful!')
"

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: One or more dependencies failed to import!
    echo Please check installation logs and try again.
    pause
    exit /b 1
)

echo.
echo ================================================
echo SUCCESS: Python 3.11 Environment is Ready!
echo ================================================
echo.
echo To run the backtest:
echo   venv_311\Scripts\python.exe main.py --mode backtest --config config/config_optimized.yaml
echo.

pause
