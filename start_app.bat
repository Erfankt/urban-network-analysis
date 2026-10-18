@echo off
rem Double-click to start the web app. First time: creates .venv and installs Streamlit
rem (the page then installs the rest). The app then runs without any window: closing its
rem browser tab stops it about half a minute later.
cd /d "%~dp0"
title Street Network app

rem Already running? Just open the browser.
powershell -NoProfile -Command "try{if((Invoke-WebRequest -UseBasicParsing http://localhost:8501/_stcore/health -TimeoutSec 2).StatusCode -ne 200){exit 1}}catch{exit 1}" >nul 2>nul
if not errorlevel 1 (
  start "" http://localhost:8501
  exit /b 0
)

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY ( where python >nul 2>nul && set "PY=python" )
if not defined PY (
  echo Python was not found. Install Python 3.11 or newer from https://www.python.org/downloads/
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
echo Starting the app; your browser opens by itself and this window closes on its own.
echo Closing the app's browser tab stops the app.
powershell -NoProfile -Command "Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-m','streamlit','run','app/streamlit_app.py' -WindowStyle Hidden -WorkingDirectory '%CD%'"
if errorlevel 1 goto :foreground

set /a tries=0
:wait
powershell -NoProfile -Command "try{if((Invoke-WebRequest -UseBasicParsing http://localhost:8501/_stcore/health -TimeoutSec 2).StatusCode -ne 200){exit 1}}catch{exit 1}" >nul 2>nul
if not errorlevel 1 exit /b 0
set /a tries+=1
if %tries% geq 20 goto :foreground
timeout /t 3 /nobreak >nul
goto :wait

:foreground
echo The app did not come up in the background; starting it in this window so any error is readable.
echo Close this window to stop the app.
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py
pause
