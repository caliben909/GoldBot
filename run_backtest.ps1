<#
.SYNOPSIS
Runs the Gold Currencies Bot backtest with optimized configuration

.DESCRIPTION
This PowerShell script runs the backtesting process using the optimized
strategy configuration designed for 80% win rate potential.
#>

# Configuration
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$pythonExecutable = "python.exe"
$mainScript = Join-Path $scriptPath "main.py"
$configFile = Join-Path $scriptPath "config" "config_optimized.yaml"

# Set working directory
Set-Location $scriptPath

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Gold Currencies Bot - Backtesting" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Check Python installation
try {
    Write-Host "Checking Python installation..." -ForegroundColor Yellow
    $pythonVersion = & $pythonExecutable --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please fix the execution alias issue!" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    
    Write-Host "`nRun check_python.bat for assistance with fixing the issue."
    Read-Host "`nPress Enter to exit..."
    exit 1
}

# Run backtest
try {
    Write-Host "`nRunning backtest with optimized configuration..." -ForegroundColor Yellow
    Write-Host "Configuration: $configFile" -ForegroundColor White
    Write-Host "`n"
    
    $startTime = Get-Date
    
    # Run the backtest
    & $pythonExecutable $mainScript --mode backtest --config $configFile
    
    $runTime = (Get-Date) - $startTime
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Backtest completed successfully!" -ForegroundColor Green
        Write-Host "Time taken: $($runTime.TotalMinutes.ToString('0.00')) minutes" -ForegroundColor Cyan
        
        Write-Host "`nResults saved to:" -ForegroundColor White
        $resultsDir = Join-Path $scriptPath "backtest" "results"
        $equityFile = Join-Path $resultsDir "equity_curve.png"
        $journalFile = Join-Path $resultsDir "trade_journal.csv"
        $reportFile = Join-Path $resultsDir "performance_report.md"
        
        if (Test-Path $equityFile) { Write-Host "  - $equityFile" -ForegroundColor Yellow }
        if (Test-Path $journalFile) { Write-Host "  - $journalFile" -ForegroundColor Yellow }
        if (Test-Path $reportFile) { Write-Host "  - $reportFile" -ForegroundColor Yellow }
        
        Write-Host "`nAnalyze the results to verify:" -ForegroundColor White
        Write-Host "  - Win rate distribution (target: 75-85%)"
        Write-Host "  - Profit factor (target: 2.5-3.5)"
        Write-Host "  - Risk metrics"
    } else {
        Write-Host "`n❌ Backtest failed!" -ForegroundColor Red
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "`n❌ Backtest failed!" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
}

Read-Host "`nPress Enter to exit..."
