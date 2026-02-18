@echo off
echo ================================================
echo Python Installation Guide
echo ================================================
echo.

echo Python executable not found in your system!
echo This script will help you install Python correctly.
echo.

echo ================================================
echo Option 1: Install from Microsoft Store (Recommended)
echo ================================================
echo 1. Press Windows key + S and search for "Microsoft Store"
echo 2. In Microsoft Store, search for "Python"
echo 3. Select Python 3.11 or later (free version)
echo 4. Click "Install" and wait for installation to complete
echo 5. Once installed, Python will be automatically available
echo.

echo ================================================
echo Option 2: Install from python.org
echo ================================================
echo 1. Open your web browser and go to: https://www.python.org/downloads/
echo 2. Click on "Download Python" button for latest version (3.11+ recommended)
echo 3. Run the downloaded installer
echo 4. IMPORTANT: Check "Add Python to PATH" checkbox!
echo 5. Click "Customize installation"
echo 6. Keep all optional features selected
echo 7. Click "Next" and complete installation
echo.

echo ================================================
echo Verification After Installation
echo ================================================
echo After installing Python, open a new Command Prompt and run:
echo.
echo python --version
echo pip --version
echo.

echo You should see output like:
echo   Python 3.11.0
echo   pip 23.0.1 from ... 
echo.

echo If you see "Python not recognized", try restarting your computer.
echo.

echo ================================================
echo Step 3: Install Dependencies
echo ================================================
echo After Python is installed, run:
echo.
echo pip install -r requirements.txt
echo.

echo This will install all the required dependencies.
echo.

pause
