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
python test_environment.py

echo.
echo To deactivate virtual environment:
echo   deactivate

pause
