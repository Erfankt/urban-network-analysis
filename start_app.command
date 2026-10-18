#!/bin/bash
# Double-click (macOS) or run `./start_app.command` (Linux) to start the web app.
# First time: creates .venv and installs Streamlit (the page then installs the rest).
# The app keeps running after this window is closed; closing the app's browser tab
# stops it about half a minute later.
cd "$(dirname "$0")" || exit 1

open_url() { command -v open >/dev/null && open "$1" || { command -v xdg-open >/dev/null && xdg-open "$1"; }; }
alive() { curl -s --max-time 2 http://localhost:8501/_stcore/health >/dev/null 2>&1; }

if alive; then
  echo "The app is already running; opening the browser."
  open_url http://localhost:8501
  exit 0
fi

PY=$(command -v python3 || command -v python)
if [ -z "$PY" ]; then
  echo "Python was not found. Install Python 3.11 or newer from https://www.python.org/downloads/ and run this again."
  read -r -p "Press Enter to close."; exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "Creating the environment in .venv (first time only) ..."
  "$PY" -m venv .venv || { echo "Could not create .venv."; read -r -p "Press Enter to close."; exit 1; }
fi

if ! .venv/bin/python -c "import streamlit" 2>/dev/null; then
  echo "Installing Streamlit (first time only) ..."
  .venv/bin/python -m pip install --quiet --disable-pip-version-check streamlit \
    || { echo "Could not install Streamlit; check the internet connection."; read -r -p "Press Enter to close."; exit 1; }
fi

echo
echo "Starting the app. Your browser opens by itself; you can close this window."
echo "Closing the app's browser tab stops the app."
nohup .venv/bin/python -m streamlit run app/streamlit_app.py >/dev/null 2>&1 &
disown
for _ in $(seq 1 20); do
  sleep 3
  alive && exit 0
done
echo "The app did not come up in the background; starting it here so any error is readable (Ctrl+C stops it)."
exec .venv/bin/python -m streamlit run app/streamlit_app.py
