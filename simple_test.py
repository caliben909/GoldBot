"""
Simple test script to check if we can run basic functionality
"""
import sys

print("Python Version:", sys.version)
print()

required_packages = [
    "numpy",
    "pandas",
    "yaml",
    "scipy",
    "statsmodels",
    "ta",
    "aiohttp",
    "python-dotenv",
    "ccxt"
]

available_packages = []
missing_packages = []

for package in required_packages:
    try:
        module = __import__(package)
        if package == "python-dotenv":
            package = "python_dotenv"
        version = getattr(module, "__version__", "Unknown")
        available_packages.append(f"{package}: {version}")
    except ImportError as e:
        missing_packages.append(package)

print("Available Packages:")
for pkg in available_packages:
    print(f"  ✔️ {pkg}")

if missing_packages:
    print("\nMissing Packages:")
    for pkg in missing_packages:
        print(f"  ❌ {pkg}")

print()

# Check if we can import core modules
print("Checking Core Modules:")
try:
    from core.risk_engine import RiskEngine
    print("  ✔️ RiskEngine imported successfully")
except Exception as e:
    print(f"  ❌ RiskEngine: {e}")

try:
    from core.strategy_engine import StrategyEngine
    print("  ✔️ StrategyEngine imported successfully")
except Exception as e:
    print(f"  ❌ StrategyEngine: {e}")

try:
    from core.risk.dxy_correlation_filter import DXYCorrelationFilter
    print("  ✔️ DXYCorrelationFilter imported successfully")
except Exception as e:
    print(f"  ❌ DXYCorrelationFilter: {e}")

print()

if len(missing_packages) == 0:
    print("✅ All packages installed! Ready to run backtests.")
else:
    print(f"⚠️  Still missing {len(missing_packages)} packages. Installation in progress...")
    print("Expected time remaining: 5-10 minutes")
