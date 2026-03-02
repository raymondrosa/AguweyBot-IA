@echo off
title Iniciar AguweyBot
echo Iniciando el programa...

:: Ejecutar comandos de PowerShell desde batch
powershell -Command "Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force"
powershell -Command "& '.\venv\Scripts\Activate.ps1'; streamlit run AguweyBot.py"

pause