@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0hrm_control.ps1" -Accion restart
if errorlevel 1 (
    echo.
    echo ERROR: no se pudo completar la accion restart. Revisa los mensajes anteriores.
)
echo.
pause
endlocal
