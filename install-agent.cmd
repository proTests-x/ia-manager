@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo Creando el entorno virtual...
python -m venv .venv
if errorlevel 1 goto :error
echo Actualizando el instalador de paquetes...
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :error
echo Instalando Ollama PC Agent...
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 goto :error
echo.
echo Instalacion completada. Ejecuta run-agent.cmd
exit /b 0

:error
echo.
echo La instalacion fallo. Revisa los mensajes anteriores.
exit /b 1
