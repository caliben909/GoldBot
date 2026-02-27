
import sys
import subprocess
import re

def check_package(package_name):
    try:
        # Try to import the package
        import importlib
        # Handle special cases for package names that don't match module names
        package_map = {
            'python-dotenv': 'dotenv',
            'python-binance': 'binance',
            'scikit-learn': 'sklearn',
            'PyYAML': 'yaml'
        }
        module_name = package_map.get(package_name, package_name.replace('-', '_'))
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def main():
    # Read requirements file
    with open('requirements.txt', 'r') as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
    
    print('Checking installed packages against requirements.txt:')
    print('=' * 60)
    
    installed = []
    missing = []
    
    for req in requirements:
        # Extract package name (remove version specifiers)
        package = re.split(r'[>=<]', req)[0]
        
        if check_package(package):
            print(f'OK {req}')
            installed.append(req)
        else:
            print(f'MISSING {req}')
            missing.append(req)
    
    print('\n' + '=' * 60)
    print(f'Installed packages: {len(installed)}/{len(requirements)}')
    print(f'Missing packages: {len(missing)}')
    
    if missing:
        print('\nMissing packages:')
        for pkg in missing:
            print(f'  - {pkg}')
    
    return len(missing) == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
