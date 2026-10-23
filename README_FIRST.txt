ROUTERESCUE AI — BUS-ONLY PROJECT PACKAGE
========================================

IMPORTANT: This package contains the Python application and its modules. The actual TGSRTC GTFS bus timetable data was NOT included in the uploaded files, so real timetable searches cannot be verified until you add the data files described below. Do not present estimated/live bus arrivals unless a genuine real-time source is connected.

FEATURES IN THE APP CODE
- Bus-only dashboard and scheduled bus search / alternatives
- Bus transfer journey planning (based on available GTFS records)
- Nearby bus stops
- Per-user favourites and search history
- Evidence-based passenger advisor / knowledge-source ingestion
- Clear distinction between scheduled timetable data and real-time operation

WHAT MAKES IT DIFFERENT
Unlike a simple map that only shows stops, RouteRescue aims to help a passenger decide: compare scheduled options, find a transfer itinerary, discover nearby stops, save useful journeys, and get answers grounded in the available source evidence. A static GTFS timetable cannot confirm a bus is currently running, its live location, delays, or exact arrival at a stop. Those require a real-time TGSRTC feed/API or a verified third-party live source.

WINDOWS SETUP (VS CODE TERMINAL)
1. Extract this ZIP to a normal folder, for example Desktop\RouteRescue-AI.
2. Open that folder in VS Code (File > Open Folder).
3. Open Terminal > New Terminal.
4. Run these commands one at a time:

   py -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m streamlit run app.py

5. Open the local URL shown in the terminal (usually http://localhost:8501).

BUS TIMETABLE DATA REQUIRED
FASTEST OPTION: double-click download_bus_data.bat. It attempts to download and install the published TGSRTC GTFS archive and creates source metadata automatically. If the download is blocked, use the browser fallback below.

Browser fallback: open https://mobilitydatabase.org/feeds/gtfs/mdb-3361, download the GTFS ZIP, extract it, and place the extracted GTFS text files in data\tgsrtc_gtfs\ . Required files include:
- agency.txt
- feed_info.txt
- routes.txt
- trips.txt
- stops.txt
- calendar.txt
- stop_times.txt
- calendar_dates.txt is optional

Also place data\tgsrtc_gtfs_source.json beside the tgsrtc_gtfs folder under data. The application expects metadata with source_url, retrieved_at_utc (UTC ISO-8601 timestamp), and sha256 (archive SHA-256). Use a legitimate, traceable TGSRTC/MobilityDatabase source archive and its real metadata; do not invent these values. The GTFS archive itself is not included because it was not among the files uploaded here.

PASSENGER KNOWLEDGE
Optional reviewed PDF/TXT/HTML/Markdown sources can be added through the app's knowledge-source flow. Do not upload private or sensitive documents.

SAFETY / PRIVACY
- Never put API keys, passwords, or .env files in a ZIP shared publicly.
- The application currently uses static GTFS data, not live vehicle tracking.
- Python syntax checks passing does not prove the app's data-dependent features work; test after adding the genuine feed.

DEMO / PROJECT PRESENTATION
1. Show the bus search and compare multiple scheduled journeys.
2. Show the transfer planner and explain that connections are built from timetable stop sequences.
3. Show nearby stops and saved favourites/history.
4. Ask the passenger advisor a question and show its cited evidence or its honest no-evidence response.
5. Explain the key limitation honestly: live arrivals need a real-time data source. Do not call static scheduled times “live arrivals.”


LATEST USABILITY FIXES
- Nearby Stops now lets passengers choose familiar landmarks instead of entering latitude/longitude.
- Fixed the manual Bookmark a Route form so its destination selector uses the correct column.
- Fixed transfer result HTML indentation so the journey cards render as cards instead of showing raw HTML.
- Transfer planner now explains in plain language that a transfer means changing from one bus to another.
