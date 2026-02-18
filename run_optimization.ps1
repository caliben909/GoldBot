<#
.SYNOPSIS
Runs the win rate optimization for the Gold Currencies Bot

.DESCRIPTION
This PowerShell script runs the win rate optimization process to
configure the bot for 80% win rate potential.
#>

# Configuration
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$pythonExecutable = "python.exe"
$optimizationScript = Join-Path $scriptPath "win_rate_optimization.py"
$configFile = Join-Path $scriptPath "config" "config_optimized.yaml"

# Check if Python is available
try {
    $pythonVersion = & $pythonExecutable --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python from https://www.python.org/" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit..."
    exit 1
}

# Run optimization
try {
    Write-Host "`n================================================" -ForegroundColor Cyan
    Write-Host "Running Win Rate Optimization" -ForegroundColor Cyan
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "Script: $optimizationScript" -ForegroundColor White
    Write-Host "`n"
    
    $result = & $pythonExecutable $optimizationScript
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Optimization completed successfully!" -ForegroundColor Green
        Write-Host "`nOptimized configuration created at:" -ForegroundColor White
        Write-Host "  $configFile" -ForegroundColor Yellow
        Write-Host "`nNext steps:" -ForegroundColor White
        Write-Host "1. Run backtest on optimized configuration"
        Write-Host "2. Analyze results"
        Write-Host "3. Validate in paper trading"
    } else {
        Write-Host "`n❌ Optimization failed!" -ForegroundColor Red
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Red
        Write-Host "`nError output:" -ForegroundColor White
        $result
    }
} catch {
    Write-Host "`n❌ Optimization failed!" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
}

Read-Host "`nPress Enter to exit..."
