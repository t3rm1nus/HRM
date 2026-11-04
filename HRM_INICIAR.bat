@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0hrm_control.ps1" -Accion start
if errorlevel 1 (
    echo.
    echo ERROR: no se pudo completar la accion start. Revisa los mensajes anteriores.
)
echo.
pause
endlocal
