@echo off
echo ================================================
echo Retrying Core Dependencies Installation (Python 3.14)
echo ================================================
echo.

REM Kill any stuck Python processes
echo Killing stuck Python processes...
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im pythonw.exe >nul 2>&1

echo.
echo Clearing temporary files...
rmdir /s /q "%TEMP%\pip-unpack*" >nul 2>&1
del "%TEMP%\*.whl" >nul 2>&1

echo.
echo Retrying installation...
"C:\Users\Mixxmasterz\AppData\Local\Programs\Python\Python314\python.exe" -m pip install --no-cache-dir numpy pandas pydantic pyyaml python-dotenv

if %ERRORLEVEL% equ 0 (
    echo.
    echo ================================================
    echo CORE DEPENDENCIES INSTALLED SUCCESSFULLY!
    echo ================================================
    echo.
    echo Verifying installation...
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
    )
) else (
    echo.
    echo WARNING: Installation failed again!
    echo.
    echo Please check your internet connection and try again.
    echo.
    echo If the issue persists, try downloading Python 3.11 from:
    echo https://www.python.org/downloads/release/python-31115/
)

echo.
pause
