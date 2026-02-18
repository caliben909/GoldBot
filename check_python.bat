@echo off
echo ================================================
echo Python Installation Check
echo ================================================
echo.

echo Checking Python installation...
echo.

REM Try running Python directly
echo 1. Testing direct Python execution:
python --version
if %ERRORLEVEL% equ 0 (
    echo Python found in PATH!
) else (
    echo Python not found in PATH
)
echo.

echo 2. Checking Python executable location:
if exist "C:\Users\Mixxmasterz\AppData\Local\Microsoft\WindowsApps\python.exe" (
    echo Python executable exists at:
    echo   C:\Users\Mixxmasterz\AppData\Local\Microsoft\WindowsApps\python.exe
    echo.
    
    REM Check execution aliases
    echo Checking App execution aliases:
    reg query "HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\App Paths\python.exe" >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo App execution alias for Python is set
    ) else (
        echo No App execution alias found for Python
    )
    echo.
    
    echo Checking Python command:
    "C:\Users\Mixxmasterz\AppData\Local\Microsoft\WindowsApps\python.exe" --version
    if %ERRORLEVEL% equ 0 (
        echo Python executable is working directly!
    ) else (
        echo Python executable not working
    )
) else (
    echo Python executable not found in WindowsApps directory!
    echo.
    echo Please install Python from Microsoft Store or visit https://www.python.org/
)

echo.
echo ================================================
echo Solution Options:
echo ================================================
echo.
echo Option 1: Disable Python execution alias in Settings
echo   1. Go to Settings ^> Apps ^> Advanced app settings
echo   2. Click on "App execution aliases"
echo   3. Find Python and disable the checkboxes
echo   4. Restart Command Prompt
echo.
echo Option 2: Install Python from python.org
echo   - Download from https://www.python.org/downloads/
echo   - During installation, select "Add Python to PATH"
echo.
echo Option 3: Use Python from Microsoft Store
echo   - Search for "Python" in Microsoft Store
echo   - Install Python 3.11 or later
echo.

pause
