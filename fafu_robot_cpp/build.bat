@echo off
setlocal EnableExtensions
REM Usage: build.bat [Release or Debug] [path-to-python.exe]
set "BUILD_TYPE=%~1"
if not defined BUILD_TYPE set "BUILD_TYPE=Release"
set "PY_COMMAND=%~2"
if not defined PY_COMMAND set "PY_COMMAND=python"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1" -Configuration "%BUILD_TYPE%" -Python "%PY_COMMAND%"
exit /b %ERRORLEVEL%
