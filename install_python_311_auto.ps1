<#
.SYNOPSIS
Automatically installs Python 3.11 and sets up virtual environment

.DESCRIPTION
This script automatically installs Python 3.11 and creates a virtual environment
with compatible packages for the Gold Currencies Bot.
#>

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Gold Currencies Bot - Python 3.11 Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Check if Python 3.11 is already installed
$python311Path = "C:\Users\Mixxmasterz\AppData\Local\Programs\Python\Python311"

if (Test-Path $python311Path) {
    Write-Host "Python 3.11 already installed at: $python311Path" -ForegroundColor Green
    $pythonExe = Join-Path $python311Path "python.exe"
}
else {
    Write-Host "Python 3.11 not found" -ForegroundColor Red
    Write-Host "`nDownloading and installing Python 3.11..." -ForegroundColor Yellow
    
    # Download Python 3.11.5
    $pythonUrl = "https://www.python.org/ftp/python/3.11.5/python-3.11.5-amd64.exe"
    $pythonInstaller = Join-Path $env:TEMP "python-3.11.5-amd64.exe"
    
    try {
        Write-Host "Downloading Python 3.11.5 installer..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri $pythonUrl -OutFile $pythonInstaller -UseBasicParsing
        
        Write-Host "Installing Python 3.11.5..." -ForegroundColor Yellow
        $arguments = "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0"
        Start-Process -FilePath $pythonInstaller -ArgumentList $arguments -Wait -NoNewWindow
        
        if (Test-Path $python311Path) {
            $pythonExe = Join-Path $python311Path "python.exe"
            Write-Host "Python 3.11 installed successfully!" -ForegroundColor Green
        }
        else {
            Write-Host "Failed to find Python 3.11 installation" -ForegroundColor Red
            exit 1
        }
        
        # Clean up installer
        Remove-Item $pythonInstaller -Force -ErrorAction SilentlyContinue
    }
    catch {
        Write-Host "Python installation failed: $_" -ForegroundColor Red
        if (Test-Path $pythonInstaller) {
            Remove-Item $pythonInstaller -Force -ErrorAction SilentlyContinue
        }
        exit 1
    }
}

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
$venvPath = Join-Path (Get-Location) "venv"
if (Test-Path $venvPath) {
    Write-Host "Virtual environment already exists" -ForegroundColor Green
}
else {
    try {
        & $pythonExe -m venv $venvPath
        Write-Host "Virtual environment created successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "Failed to create virtual environment: $_" -ForegroundColor Red
        exit 1
    }
}

# Install dependencies
Write-Host "`nInstalling compatible packages..." -ForegroundColor Yellow
try {
    # Create temporary requirements file with compatible versions
    $tempRequirements = Join-Path (Get-Location) "requirements_compatible.txt"
    $compatibleRequirements = @'
numpy>=1.24.0,<2.0.0
pandas>=2.0.0,<3.0.0
pydantic>=2.0.0
pyyaml>=6.0
python-dotenv>=1.0.0
ta>=0.10.2
pandas-ta>=0.3.14b0
ta-lib>=0.4.28
scikit-learn>=1.3.0,<2.0.0
xgboost>=2.0.0,<3.0.0
lightgbm>=4.0.0,<5.0.0
optuna>=3.0.0,<4.0.0
mlflow>=2.0.0,<3.0.0
shap>=0.42.0,<1.0.0
onnxruntime>=1.15.0,<2.0.0
joblib>=1.3.0,<2.0.0
vectorbt>=0.26.0,<0.27.0
backtrader>=1.9.78.123
matplotlib>=3.7.0,<4.0.0
plotly>=5.15.0,<6.0.0
seaborn>=0.12.2,<0.13.0
aiohttp>=3.8.0,<4.0.0
requests>=2.31.0,<3.0.0
asyncio>=3.4.3
aiofiles>=23.0.0,<24.0.0
psutil>=5.9.0,<6.0.0
structlog>=23.1.0,<24.0.0
python-json-logger>=2.0.0,<3.0.0
apscheduler>=3.10.0,<4.0.0
scipy>=1.10.0,<2.0.0
statsmodels>=0.14.0,<0.15.0
pytest>=7.4.0,<8.0.0
pytest-asyncio>=0.21.0,<0.22.0
pytest-cov>=4.1.0,<5.0.0
black>=23.0.0,<24.0.0
isort>=5.12.0,<6.0.0
mypy>=1.5.0,<2.0.0
pre-commit>=3.3.0,<4.0.0
'@
    $compatibleRequirements | Out-File -FilePath $tempRequirements -Encoding ascii
    
    Write-Host "Upgrading pip..." -ForegroundColor Yellow
    & (Join-Path $venvPath "Scripts\python.exe") -m pip install --upgrade pip
    
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    & (Join-Path $venvPath "Scripts\python.exe") -m pip install -r $tempRequirements
    
    # Clean up temporary requirements file
    Remove-Item $tempRequirements -Force -ErrorAction SilentlyContinue
    
    Write-Host "Dependencies installed successfully!" -ForegroundColor Green
}
catch {
    Write-Host "Failed to install dependencies: $_" -ForegroundColor Red
    Remove-Item $tempRequirements -Force -ErrorAction SilentlyContinue
    exit 1
}

Write-Host "`n"
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "SUCCESS: Python 3.11 Environment Setup Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "`nTo activate and use the virtual environment:" -ForegroundColor White
Write-Host "  1. Open PowerShell or Command Prompt" -ForegroundColor White
Write-Host "  2. Navigate to your project directory" -ForegroundColor White
Write-Host "  3. Run: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "  4. Run your commands" -ForegroundColor White
Write-Host "`nTo deactivate the virtual environment:" -ForegroundColor White
Write-Host "  deactivate" -ForegroundColor Yellow
Write-Host "`n"

Write-Host "Testing virtual environment..." -ForegroundColor Yellow
try {
    $testResult = & (Join-Path $venvPath "Scripts\python.exe") -c "import sys, numpy, pandas, yaml; print('Python: %s' % sys.version); print('NumPy: %s' % numpy.__version__); print('Pandas: %s' % pandas.__version__); print('YAML: %s' % yaml.__version__); print('All core dependencies loaded successfully!')"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nVirtual environment test passed!" -ForegroundColor Green
        Write-Host $testResult -ForegroundColor White
    }
    else {
        Write-Host "`nVirtual environment test failed!" -ForegroundColor Red
    }
}
catch {
    Write-Host "`nVirtual environment test failed: $_" -ForegroundColor Red
}
