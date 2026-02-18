import sys
import importlib.util

required_packages = [
    "numpy",
    "pandas",
    "yaml",
    "scipy",
    "statsmodels",
    "ta",
    "asyncio",
    "aiohttp",
    "dotenv",
    "ccxt"
]

print("=" * 50)
print("Python 3.11 Environment Test")
print("=" * 50)
print()

print(f"Python Version: {sys.version}")
print()

success = True

for package in required_packages:
    try:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "Unknown")
        print("OK:", package, ":", version)
    except ImportError:
        print("ERROR:", package, ": NOT INSTALLED")
        success = False

print()
print("=" * 50)
if success:
    print("SUCCESS: All required packages installed successfully!")
else:
    print("ERROR: Some packages are missing!")
print("=" * 50)

if success:
    print()
    print("To run the backtest:")
    print("venv_311\\Scripts\\python.exe main.py --mode backtest --config config/config_optimized.yaml")
