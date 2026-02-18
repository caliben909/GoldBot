@echo off
echo ================================================
echo Gold Currencies Bot - Installation Check
echo ================================================
echo.

echo Checking Python installation status...
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Virtual environment found at venv\
    echo.
    
    REM Try to activate and test virtual environment
    echo Testing virtual environment...
    call venv\Scripts\activate.bat >nul 2>&1
    
    if %ERRORLEVEL% equ 0 (
        echo Virtual environment activated successfully!
        echo.
        
        REM Check core dependencies
        echo Checking core dependencies...
        venv\Scripts\python.exe -c "import sys, numpy, pandas, yaml; print('Python: ' + sys.version); print('NumPy: ' + numpy.__version__); print('Pandas: ' + pandas.__version__); print('YAML: ' + yaml.__version__); print('All core dependencies loaded successfully!')" >nul 2>&1
        
        if %ERRORLEVEL% equ 0 (
            echo.
            echo ================================================
            echo SUCCESS: Bot is ready to run!
            echo ================================================
            echo.
            echo Available commands:
            echo   1. Run backtest with optimized configuration:
            echo      venv\Scripts\python.exe main.py --mode backtest --config config/config_optimized.yaml
            echo.
            echo   2. Run backtest with default configuration:
            echo      venv\Scripts\python.exe main.py --mode backtest --config config/config.yaml
            echo.
            echo   3. Run optimization:
            echo      venv\Scripts\python.exe win_rate_optimization.py
            echo.
            echo To activate the virtual environment manually:
            echo   call venv\Scripts\activate.bat
            echo.
        ) else (
            echo WARNING: Virtual environment is damaged or dependencies are missing!
            echo Please run install_python_311_auto.ps1 again.
        )
    ) else (
        echo WARNING: Virtual environment activation failed!
        echo Please check the installation and try again.
    )
    
    echo.
    echo To deactivate virtual environment:
    echo   deactivate
) else (
    echo Virtual environment not found!
    echo.
    
    REM Check if Python 3.11 is installed
    if exist "C:\Users\Mixxmasterz\AppData\Local\Programs\Python\Python311\python.exe" (
        echo Python 3.11 found at C:\Users\Mixxmasterz\AppData\Local\Programs\Python\Python311\python.exe
        echo Virtual environment creation failed or was canceled.
        echo.
        echo To create virtual environment manually:
        echo   "C:\Users\Mixxmasterz\AppData\Local\Programs\Python\Python311\python.exe" -m venv venv
        echo   venv\Scripts\activate.bat
        echo   venv\Scripts\python.exe -m pip install -r requirements_core.txt
    ) else (
        echo Python 3.11 not installed!
        echo.
        echo To install Python 3.11 manually:
        echo 1. Download from https://www.python.org/downloads/release/python-31115/
        echo 2. Check "Add Python 3.11 to PATH" during installation
        echo 3. Run install_python_311_auto.ps1 to create virtual environment
    )
)

echo.
pause
