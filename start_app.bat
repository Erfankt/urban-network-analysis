@echo off
rem Double-click to start the web app. First time: creates .venv and installs Streamlit
rem (the page then installs the rest). Close this window to stop the app.
cd /d "%~dp0"
title Street Network app

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY ( where python >nul 2>nul && set "PY=python" )
if not defined PY (
  echo Python was not found. Install Python 3.11 from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" in the installer, then double-click this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating the environment in .venv ^(first time only^) ...
  %PY% -m venv .venv
  if errorlevel 1 ( echo Could not create .venv. & pause & exit /b 1 )
)

".venv\Scripts\python.exe" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
  echo Installing Streamlit ^(first time only^) ...
  ".venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check streamlit
  if errorlevel 1 ( echo Could not install Streamlit; check the internet connection. & pause & exit /b 1 )
)

echo.
echo Starting the app. Your browser opens by itself; close this window to stop the app.
echo.
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py
echo.
echo The app has stopped.
pause
