<#
.SYNOPSIS
Tests Python 3.11 environment and core dependencies.

.DESCRIPTION
This script checks if the Python 3.11 virtual environment is properly configured
and all core dependencies are available.
#>

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Testing Python 3.11 Environment" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Configuration
$venvPath = "venv_311"
$activateScript = Join-Path $venvPath "Scripts\activate.ps1"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"

# Check if virtual environment exists
if (-not (Test-Path $venvPath)) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run install_311_fixed.bat first" -ForegroundColor Yellow
    Read-Host "Press Enter to exit..."
    exit 1
}

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
try {
    & $activateScript
} catch {
    Write-Host "ERROR: Failed to activate virtual environment!" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit..."
    exit 1
}

# Check Python version
Write-Host "`nChecking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = & $pythonExe --version
    Write-Host $pythonVersion -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to get Python version!" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit..."
    exit 1
}

# Check core dependencies
Write-Host "`nChecking core dependencies..." -ForegroundColor Yellow
try {
    $testResult = & $pythonExe -c "
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
    
    Write-Host $testResult -ForegroundColor Green
    
    Write-Host "`n================================================" -ForegroundColor Cyan
    Write-Host "SUCCESS: Python 3.11 Environment is Ready!" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Cyan
    
    Write-Host "`nTo run the backtest:" -ForegroundColor White
    Write-Host "  venv_311\Scripts\python.exe main.py --mode backtest --config config/config_optimized.yaml" -ForegroundColor Yellow
    
    Write-Host "`nResults will be saved to:" -ForegroundColor White
    Write-Host "  backtest/results/" -ForegroundColor Yellow
    
} catch {
    Write-Host "ERROR: One or more dependencies failed to import!" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    
    Write-Host "`nPlease check installation logs and try again." -ForegroundColor Yellow
    Read-Host "Press Enter to exit..."
    exit 1
}

Read-Host "`nPress Enter to exit..."
