@echo off
echo Quick Installation Check
echo =========================
echo.

echo Python Version:
venv_311\Scripts\python.exe --version
echo.

echo Installed Packages:
venv_311\Scripts\python.exe -m pip list
echo.

echo Active Python Processes:
tasklist /FI "IMAGENAME eq python.exe"
echo.

echo Installation Status: Running - Downloading dependencies...
echo.
echo Please wait for installation to complete (this may take 10-15 minutes)
