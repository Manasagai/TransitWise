"""Download and install the published static TGSRTC GTFS timetable.

This is scheduled data, not live vehicle tracking. Run with the project venv Python.
"""
from __future__ import annotations
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
FEED_DIR = DATA_DIR / 'tgsrtc_gtfs'
META_PATH = DATA_DIR / 'tgsrtc_gtfs_source.json'
URL = ('https://data.opencity.in/dataset/88e2d145-7ec6-4666-88dd-6cf18b18312e/'
       'resource/1b0d18bb-b2fb-4a79-8ed0-1e071da5790c/download/'
       'telangana_opendata_gtfs_tgsrtc_08_february_2026.zip')
REQUIRED = {'agency.txt', 'feed_info.txt', 'routes.txt', 'trips.txt', 'stops.txt', 'calendar.txt', 'stop_times.txt'}
ARCHIVE = DATA_DIR / 'tgsrtc_gtfs_download.zip'


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print('Downloading published TGSRTC static bus timetable...')
    print('This may take a few minutes depending on your connection.')
    request = urllib.request.Request(URL, headers={'User-Agent': 'RouteRescueAI/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=90) as response, ARCHIVE.open('wb') as out:
            shutil.copyfileobj(response, out)
    except Exception as exc:
        ARCHIVE.unlink(missing_ok=True)
        print('\nDownload failed:', exc)
        print('Open this listing in your browser and download its latest GTFS ZIP:')
        print('https://mobilitydatabase.org/feeds/gtfs/mdb-3361')
        print('Then extract it and place the GTFS .txt files in data\\tgsrtc_gtfs\\.')
        return 1
    if not zipfile.is_zipfile(ARCHIVE):
        ARCHIVE.unlink(missing_ok=True)
        print('The downloaded response was not a valid ZIP file. Open the feed listing in a browser.')
        return 1
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    temp = DATA_DIR / 'gtfs_extract_temp'
    shutil.rmtree(temp, ignore_errors=True)
    temp.mkdir()
    try:
        with zipfile.ZipFile(ARCHIVE) as zf:
            for info in zf.infolist():
                p = Path(info.filename)
                if p.is_absolute() or '..' in p.parts:
                    raise ValueError('Unsafe path found in downloaded archive')
            zf.extractall(temp)
        found = None
        for directory in [temp, *[p for p in temp.rglob('*') if p.is_dir()]]:
            if REQUIRED.issubset({p.name for p in directory.iterdir() if p.is_file()}):
                found = directory
                break
        if found is None:
            raise ValueError('ZIP did not contain the required GTFS timetable files at one directory level')
        new_feed = DATA_DIR / 'tgsrtc_gtfs_new'
        shutil.rmtree(new_feed, ignore_errors=True)
        shutil.copytree(found, new_feed)
        backup = DATA_DIR / 'tgsrtc_gtfs_previous'
        shutil.rmtree(backup, ignore_errors=True)
        if FEED_DIR.exists():
            FEED_DIR.rename(backup)
        new_feed.rename(FEED_DIR)
        metadata = {
            'source_url': URL,
            'retrieved_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
            'sha256': digest,
            'static_data_only': True,
            'source_name': 'TGSRTC Hyderabad static GTFS schedule via OpenCity/Mobility Database',
            'archive_snapshot_id': 'opencity-tgsrtc-08-february-2026',
        }
        META_PATH.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print('\nSUCCESS: GTFS files installed in data\\tgsrtc_gtfs\\')
        print('Archive SHA-256:', digest)
        print('NOTE: This is scheduled timetable data, not live bus locations or live arrival predictions.')
        print('You can now start RouteRescue AI.')
        return 0
    except Exception as exc:
        print('\nCould not install the GTFS archive:', exc)
        return 1
    finally:
        shutil.rmtree(temp, ignore_errors=True)
        ARCHIVE.unlink(missing_ok=True)

if __name__ == '__main__':
    raise SystemExit(main())
