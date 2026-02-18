cd /d "c:/Users/Mixxmasterz/Documents/Gold Currencies Bot"
takeown /r /f .git
icacls .git /grant %username%:F /t
rd /s /q .git
echo Done!
pause