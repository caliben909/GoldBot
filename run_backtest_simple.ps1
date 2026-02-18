<#
.SYNOPSIS
Simple PowerShell script to run backtest without execution policy issues
#>

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Gold Currencies Bot - Backtesting" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Check Python installation
$pythonPath = "C:\Users\Mixxmasterz\AppData\Local\Microsoft\WindowsApps\python.exe"

if (Test-Path $pythonPath) {
    Write-Host "✅ Python executable found at: $pythonPath" -ForegroundColor Green
} else {
    Write-Host "❌ Python executable not found!" -ForegroundColor Red
    Write-Host "Please install Python or check the installation path" -ForegroundColor Yellow
    Read-Host "`nPress Enter to exit..."
    exit 1
}

# Run backtest
try {
    Write-Host "`nRunning backtest..." -ForegroundColor Yellow
    Write-Host "Configuration: config\config_optimized.yaml" -ForegroundColor White
    
    $startTime = Get-Date
    
    & $pythonPath "main.py" "--mode" "backtest" "--config" "config\config_optimized.yaml"
    
    $runTime = (Get-Date) - $startTime
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Backtest completed successfully!" -ForegroundColor Green
        Write-Host "Time taken: $($runTime.TotalMinutes.ToString('0.00')) minutes" -ForegroundColor Cyan
    } else {
        Write-Host "`n❌ Backtest failed!" -ForegroundColor Red
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "`n❌ Backtest failed!" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
}

Read-Host "`nPress Enter to exit..."
