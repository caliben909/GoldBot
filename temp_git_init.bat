cd /d "c:/Users/Mixxmasterz/Documents/Gold Currencies Bot"
del /s /q .git 2>NUL
rmdir /s /q .git 2>NUL
git init
git remote add origin https://github.com/caliben909/GoldBot.git
git config user.name "Your Name"
git config user.email "you@example.com"
git add .gitignore
git add --exclude=venv_311 --exclude=*.pyc --exclude=__pycache__ --exclude=*.tmp --exclude=*.temp --exclude=*.log --exclude=*.csv --exclude=*.db --exclude=*.sqlite --exclude=*.onnx --exclude=*.h5 --exclude=*.pkl .
echo Git repository initialized successfully!