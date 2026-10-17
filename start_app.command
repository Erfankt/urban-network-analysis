#!/bin/bash
# Double-click (macOS) or run `./start_app.command` (Linux) to start the web app.
# First time: creates .venv and installs Streamlit (the page then installs the rest).
# Close this window (or press Ctrl+C) to stop the app.
cd "$(dirname "$0")" || exit 1

PY=$(command -v python3 || command -v python)
if [ -z "$PY" ]; then
  echo "Python was not found. Install Python 3.11 from https://www.python.org/downloads/ and run this again."
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
echo "Starting the app. Your browser opens by itself; close this window to stop the app."
echo
.venv/bin/python -m streamlit run app/streamlit_app.py
