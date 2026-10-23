@echo off
setlocal
cd /d "%~dp0"
title RouteRescue AI - Setup and Launch

echo ======================================================
echo             ROUTERESCUE AI - ONE CLICK
echo ======================================================
echo.
where py >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python was not found. Install Python 3.10+ from https://www.python.org/downloads/windows/
  echo During installation, select "Add Python to PATH".
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Creating Python environment...
  py -m venv .venv
  if errorlevel 1 goto fail
) else (
  echo [1/4] Python environment already exists.
)
echo.
echo [2/4] Installing required packages (first run may take a few minutes)...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto fail
echo.
if not exist "data\tgsrtc_gtfs\agency.txt" (
  echo [3/4] Attempting to download the public TGSRTC timetable...
  ".venv\Scripts\python.exe" install_bus_data.py
  if errorlevel 1 (
    echo.
    echo WARNING: Automatic timetable download did not succeed.
    echo The app will still open, but real bus search needs a valid GTFS timetable.
    echo Official data page: https://www.tgsrtc.telangana.gov.in/open-data
    echo Feed listing: https://mobilitydatabase.org/feeds/gtfs/mdb-3361
  )
) else (
  echo [3/4] TGSRTC timetable files already exist.
)
echo.
echo [4/4] Starting RouteRescue AI...
echo When Streamlit starts, open http://localhost:8501 in your browser.
echo Keep this window open while using the website.
echo Press Ctrl+C here to stop the app.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto fail
exit /b 0
:fail
echo.
echo Setup could not finish. Copy the error shown above and send it to ChatGPT.
pause
exit /b 1
