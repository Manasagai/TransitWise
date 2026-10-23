"""Reader and scheduled-journey search for the downloaded TGSRTC static GTFS feed."""

import csv
import json
import math
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class TgsrtcGtfsError(ValueError):
    """Raised when the extracted GTFS feed is missing or inconsistent."""


_GTFS_TIME = re.compile(r"^(\d+):([0-5]\d):([0-5]\d)$")
_WEEKDAY_COLUMNS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
_HYDERABAD_STOP_IDS = (
    ("hQYYZWl5", "Mgbs"),
    ("gVRyRzSN", "Jbs"),
    ("u3bvsFOc", "Secunderabad"),
    ("qbtDyPmY", "Uppal X Roads"),
    ("5lBRTe5B", "Dilsukhnagar Bus Depot"),
    ("jNMgv2Al", "Lb Nagar"),
    ("URqOZwMM", "Kothapet X Road"),
    ("9ahqTeb2", "Koti Women College"),
    ("RgqMpbyP", "Charminar"),
    ("RvkZZP0d", "Afzalgunj"),
    ("8A7oqeuK", "Ameerpet"),
    ("YY5SqKbk", "Punjagutta"),
    ("0JxeENcx", "Kukatpally"),
    ("DFla4Krg", "Kphb"),
    ("QaDZj4Nl", "Miyapur Metro Station"),
    ("NUmq3A4e", "Mehdipatnam"),
    ("USn7tbeB", "Lakdikapool"),
    ("0Yr3hLAU", "Nampally Railway Station"),
    ("AZ9WhQ64", "Kachiguda Railway Station"),
    ("are2VdTm", "Tarnaka"),
    ("PRepdcnz", "Ecil X Road"),
    ("UqKAXjCm", "Hitech City (Cyber Towers)"),
    ("NvZJ6Bq3", "Lingampally"),
    ("WcY4kntl", "Gachibowli"),
)


class TgsrtcGtfsFeed:
    """Validate and search scheduled TGSRTC services without implying live status."""

    def __init__(
        self,
        feed_dir: Optional[Path] = None,
        source_metadata_path: Optional[Path] = None,
    ) -> None:
        project_root = Path(__file__).resolve().parent
        self.feed_dir = Path(feed_dir or project_root / "data" / "tgsrtc_gtfs")
        self.source_metadata_path = Path(
            source_metadata_path or project_root / "data" / "tgsrtc_gtfs_source.json"
        )
        self._agency = self._read_table(
            "agency.txt",
            {"agency_id", "agency_name", "agency_url", "agency_timezone"},
        )
        self._feed_info = self._read_table(
            "feed_info.txt",
            {"feed_publisher_name", "feed_publisher_url", "feed_start_date", "feed_end_date", "feed_version"},
        )
        self._routes = self._read_table(
            "routes.txt", {"route_id", "route_long_name", "route_type"}
        )
        self._trips = self._read_table(
            "trips.txt", {"route_id", "trip_id", "service_id", "trip_short_name"}
        )
        self._stops = self._read_table(
            "stops.txt", {"stop_id", "stop_name", "stop_lat", "stop_lon"}
        )
        self._calendar = self._read_table(
            "calendar.txt",
            {"service_id", "start_date", "end_date", *_WEEKDAY_COLUMNS},
        )
        self._calendar_dates = self._read_optional_table(
            "calendar_dates.txt", {"service_id", "date", "exception_type"}
        )
        self._stop_times_path = self.feed_dir / "stop_times.txt"
        if not self._stop_times_path.is_file():
            raise TgsrtcGtfsError(f"Required GTFS file is missing: {self._stop_times_path}")
        self.stop_time_columns = self._require_columns(
            self._stop_times_path,
            {"trip_id", "stop_sequence", "stop_id", "departure_time", "arrival_time"},
        )

        if len(self._agency) != 1 or len(self._feed_info) != 1:
            raise TgsrtcGtfsError("The feed must contain exactly one agency and one feed_info row.")
        self.agency = self._agency[0]
        self.feed_info = self._feed_info[0]
        try:
            self.timezone = ZoneInfo(self.agency["agency_timezone"])
        except (KeyError, ZoneInfoNotFoundError) as error:
            raise TgsrtcGtfsError("The agency timezone is missing or unsupported.") from error

        self._routes_by_id = self._unique_index(self._routes, "route_id", "routes.txt")
        self._trips_by_id = self._unique_index(self._trips, "trip_id", "trips.txt")
        self._stops_by_id = self._unique_index(self._stops, "stop_id", "stops.txt")
        self._calendar_by_service = self._unique_index(
            self._calendar, "service_id", "calendar.txt"
        )
        self._exceptions = self._load_calendar_exceptions()
        self._check_relationships()
        self._metadata = self._load_source_metadata()
        self._routes_by_stop: Dict[str, Set[str]] = {}
        self.stop_times_count = self._validate_stop_times()
        self.validated_at_utc = datetime.now(timezone.utc).isoformat()

        try:
            self.feed_start_date = self._parse_gtfs_date(self.feed_info["feed_start_date"])
            self.feed_end_date = self._parse_gtfs_date(self.feed_info["feed_end_date"])
        except (KeyError, ValueError) as error:
            raise TgsrtcGtfsError("feed_info.txt has invalid feed coverage dates.") from error
        if self.feed_start_date > self.feed_end_date:
            raise TgsrtcGtfsError("feed_info.txt has a reversed feed coverage range.")

    @property
    def source_metadata(self) -> Dict[str, Any]:
        """Return the validated archive provenance metadata."""
        return dict(self._metadata)

    @property
    def route_count(self) -> int:
        return len(self._routes_by_id)

    @property
    def trip_count(self) -> int:
        return len(self._trips_by_id)

    @property
    def stop_count(self) -> int:
        return len(self._stops_by_id)

    @staticmethod
    def _require_columns(path: Path, required: set) -> Set[str]:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as source:
                header = next(csv.reader(source), None)
        except OSError as error:
            raise TgsrtcGtfsError(f"Cannot read GTFS file {path.name}: {error}") from error
        if header is None or not required.issubset(set(header)):
            missing = sorted(required - set(header or []))
            raise TgsrtcGtfsError(
                f"{path.name} is missing required columns: {', '.join(missing)}"
            )
        return set(header)

    def _read_table(self, filename: str, required: set) -> List[Dict[str, str]]:
        path = self.feed_dir / filename
        if not path.is_file():
            raise TgsrtcGtfsError(f"Required GTFS file is missing: {path}")
        self._require_columns(path, required)
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as source:
                return list(csv.DictReader(source))
        except (OSError, csv.Error) as error:
            raise TgsrtcGtfsError(f"Cannot parse GTFS file {filename}: {error}") from error

    def _read_optional_table(
        self, filename: str, required: set
    ) -> List[Dict[str, str]]:
        path = self.feed_dir / filename
        if not path.exists():
            return []
        return self._read_table(filename, required)

    @staticmethod
    def _unique_index(
        rows: List[Dict[str, str]], key: str, filename: str
    ) -> Dict[str, Dict[str, str]]:
        index: Dict[str, Dict[str, str]] = {}
        for row_number, row in enumerate(rows, start=2):
            identifier = row.get(key, "").strip()
            if not identifier or identifier in index:
                raise TgsrtcGtfsError(
                    f"{filename} has a missing or duplicate {key} at row {row_number}."
                )
            index[identifier] = row
        return index

    @staticmethod
    def _parse_gtfs_date(value: str) -> date:
        if not re.fullmatch(r"\d{8}", value or ""):
            raise ValueError("Expected YYYYMMDD.")
        return datetime.strptime(value, "%Y%m%d").date()

    @staticmethod
    def _parse_gtfs_time(value: str, filename: str, row_number: int) -> int:
        match = _GTFS_TIME.fullmatch(value or "")
        if match is None:
            raise TgsrtcGtfsError(
                f"{filename} has an invalid GTFS time at row {row_number}."
            )
        hours, minutes, seconds = (int(part) for part in match.groups())
        return hours * 3600 + minutes * 60 + seconds

    def _load_calendar_exceptions(self) -> Dict[Tuple[str, date], int]:
        exceptions: Dict[Tuple[str, date], int] = {}
        for row_number, row in enumerate(self._calendar_dates, start=2):
            try:
                service_date = self._parse_gtfs_date(row["date"])
                exception_type = int(row["exception_type"])
            except (KeyError, ValueError) as error:
                raise TgsrtcGtfsError(
                    f"calendar_dates.txt has an invalid record at row {row_number}."
                ) from error
            if exception_type not in (1, 2):
                raise TgsrtcGtfsError(
                    f"calendar_dates.txt has an unsupported exception type at row {row_number}."
                )
            key = (row["service_id"], service_date)
            if key in exceptions:
                raise TgsrtcGtfsError(
                    f"calendar_dates.txt has a duplicate service/date at row {row_number}."
                )
            exceptions[key] = exception_type
        return exceptions

    def _check_relationships(self) -> None:
        for row_number, calendar in enumerate(self._calendar, start=2):
            try:
                start = self._parse_gtfs_date(calendar["start_date"])
                end = self._parse_gtfs_date(calendar["end_date"])
            except (KeyError, ValueError) as error:
                raise TgsrtcGtfsError(
                    f"calendar.txt has invalid dates at row {row_number}."
                ) from error
            if start > end or any(
                calendar.get(day) not in ("0", "1") for day in _WEEKDAY_COLUMNS
            ):
                raise TgsrtcGtfsError(
                    f"calendar.txt has an invalid date range or weekday flag at row {row_number}."
                )

        for route in self._routes:
            if route.get("route_type") != "3":
                raise TgsrtcGtfsError(
                    f"routes.txt contains a non-bus route type for route {route.get('route_id')}."
                )
            agency_id = route.get("agency_id")
            if agency_id and agency_id != self.agency["agency_id"]:
                raise TgsrtcGtfsError(
                    f"Route {route['route_id']} references an unknown agency."
                )

        for trip in self._trips:
            if trip["route_id"] not in self._routes_by_id:
                raise TgsrtcGtfsError(f"Trip {trip['trip_id']} references an unknown route.")
            if (
                trip["service_id"] not in self._calendar_by_service
                and not any(key[0] == trip["service_id"] for key in self._exceptions)
            ):
                raise TgsrtcGtfsError(f"Trip {trip['trip_id']} references an unknown service.")

        for row_number, stop in enumerate(self._stops, start=2):
            try:
                latitude = float(stop["stop_lat"])
                longitude = float(stop["stop_lon"])
            except (KeyError, ValueError) as error:
                raise TgsrtcGtfsError(
                    f"stops.txt has invalid coordinates at row {row_number}."
                ) from error
            if (
                not math.isfinite(latitude)
                or not math.isfinite(longitude)
                or not -90 <= latitude <= 90
                or not -180 <= longitude <= 180
            ):
                raise TgsrtcGtfsError(
                    f"stops.txt has out-of-range coordinates at row {row_number}."
                )

    def _load_source_metadata(self) -> Dict[str, Any]:
        if not self.source_metadata_path.is_file():
            return {}
        try:
            with self.source_metadata_path.open("r", encoding="utf-8") as source:
                metadata = json.load(source)
        except (OSError, json.JSONDecodeError) as error:
            raise TgsrtcGtfsError("GTFS source provenance metadata is invalid.") from error
        required = ("source_url", "retrieved_at_utc", "sha256")
        if not isinstance(metadata, dict) or any(
            not isinstance(metadata.get(key), str) or not metadata[key].strip()
            for key in required
        ):
            raise TgsrtcGtfsError(
                "GTFS source metadata requires source_url, retrieved_at_utc, and sha256."
            )
        try:
            retrieved_at = datetime.fromisoformat(
                metadata["retrieved_at_utc"].replace("Z", "+00:00")
            )
        except ValueError as error:
            raise TgsrtcGtfsError("GTFS retrieval timestamp is invalid.") from error
        if retrieved_at.tzinfo is None or retrieved_at.utcoffset() != timedelta(0):
            raise TgsrtcGtfsError("GTFS retrieval timestamp must be UTC.")
        return metadata

    def _validate_stop_times(self) -> int:
        row_count = 0
        seen_trip_ids = set()
        current_trip_id: Optional[str] = None
        previous_sequence = 0
        previous_departure = 0
        current_trip_count = 0
        try:
            with self._stop_times_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as source:
                for row_number, row in enumerate(csv.DictReader(source), start=2):
                    trip_id = row.get("trip_id", "")
                    if trip_id != current_trip_id:
                        if trip_id in seen_trip_ids:
                            raise TgsrtcGtfsError(
                                "stop_times.txt has non-contiguous rows for a trip."
                            )
                        if current_trip_id is not None and current_trip_count < 2:
                            raise TgsrtcGtfsError(
                                f"Trip {current_trip_id} has fewer than two stop times."
                            )
                        seen_trip_ids.add(trip_id)
                        current_trip_id = trip_id
                        previous_sequence = 0
                        previous_departure = 0
                        current_trip_count = 0
                    if trip_id not in self._trips_by_id:
                        raise TgsrtcGtfsError(
                            f"stop_times.txt row {row_number} references an unknown trip."
                        )
                    if row.get("stop_id") not in self._stops_by_id:
                        raise TgsrtcGtfsError(
                            f"stop_times.txt row {row_number} references an unknown stop."
                        )
                    try:
                        sequence = int(row["stop_sequence"])
                    except (KeyError, ValueError) as error:
                        raise TgsrtcGtfsError(
                            f"stop_times.txt has an invalid stop_sequence at row {row_number}."
                        ) from error
                    if sequence <= previous_sequence:
                        raise TgsrtcGtfsError(
                            f"stop_times.txt stop_sequence is not increasing at row {row_number}."
                        )
                    arrival = self._parse_gtfs_time(
                        row.get("arrival_time", ""), "stop_times.txt", row_number
                    )
                    departure = self._parse_gtfs_time(
                        row.get("departure_time", ""), "stop_times.txt", row_number
                    )
                    if arrival > departure or arrival < previous_departure:
                        raise TgsrtcGtfsError(
                            f"stop_times.txt has inconsistent times at row {row_number}."
                        )
                    previous_sequence = sequence
                    previous_departure = departure
                    current_trip_count += 1
                    row_count += 1
                    sid = row.get("stop_id")
                    if sid:
                        self._routes_by_stop.setdefault(sid, set()).add(
                            self._trips_by_id[trip_id]["route_id"]
                        )
        except (OSError, csv.Error) as error:
            raise TgsrtcGtfsError(
                f"Cannot validate stop_times.txt: {error}"
            ) from error
        if current_trip_id is not None and current_trip_count < 2:
            raise TgsrtcGtfsError(
                f"Trip {current_trip_id} has fewer than two stop times."
            )
        if not row_count:
            raise TgsrtcGtfsError("stop_times.txt contains no scheduled stop records.")
        missing_trip_ids = set(self._trips_by_id) - seen_trip_ids
        if missing_trip_ids:
            raise TgsrtcGtfsError(
                f"stop_times.txt is missing records for {len(missing_trip_ids)} trips."
            )
        return row_count

    def _service_is_active(self, service_id: str, service_date: date) -> bool:
        exception = self._exceptions.get((service_id, service_date))
        if exception is not None:
            return exception == 1
        calendar = self._calendar_by_service.get(service_id)
        if calendar is None:
            return False
        try:
            start = self._parse_gtfs_date(calendar["start_date"])
            end = self._parse_gtfs_date(calendar["end_date"])
            weekday_active = calendar[_WEEKDAY_COLUMNS[service_date.weekday()]] == "1"
        except (KeyError, ValueError) as error:
            raise TgsrtcGtfsError(
                f"calendar.txt has an invalid record for service {service_id}."
            ) from error
        return start <= service_date <= end and weekday_active

    def get_stops(self) -> List[Dict[str, str]]:
        """Return every unique stop in the validated GTFS feed."""
        return [
            {
                "stop_id": stop["stop_id"],
                "stop_name": stop["stop_name"],
                "label": f"{stop['stop_name']} ({stop['stop_id']})",
            }
            for stop in sorted(
                self._stops,
                key=lambda item: (item["stop_name"].casefold(), item["stop_id"]),
            )
        ]

    def get_hyderabad_stops(self) -> List[Dict[str, str]]:
        """Return feed-verified Hyderabad interchange stops for passenger selectors."""
        stops = []
        for stop_id, expected_name in _HYDERABAD_STOP_IDS:
            stop = self._stops_by_id.get(stop_id)
            if stop is None or stop["stop_name"] != expected_name:
                continue
            stops.append({
                "stop_id": stop_id,
                "stop_name": expected_name,
                "label": f"{expected_name} ({stop_id})",
            })
        return stops

    def _stop_time_rows(self) -> Iterator[Dict[str, str]]:
        try:
            with self._stop_times_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as source:
                yield from csv.DictReader(source)
        except (OSError, csv.Error) as error:
            raise TgsrtcGtfsError(f"Cannot read stop_times.txt: {error}") from error

    def _make_leg(
        self,
        trip: Dict[str, str],
        route: Dict[str, str],
        trip_rows: List[Dict[str, str]],
        start_index: int,
        end_index: int,
        service_date: date,
        leg_index: int = 1,
    ) -> Dict[str, Any]:
        origin_row = trip_rows[start_index]
        destination_row = trip_rows[end_index]
        departure_seconds = self._parse_gtfs_time(
            origin_row["departure_time"], "stop_times.txt", 0
        )
        arrival_seconds = self._parse_gtfs_time(
            destination_row["arrival_time"], "stop_times.txt", 0
        )
        service_midnight = datetime.combine(
            service_date, time.min, tzinfo=self.timezone
        )
        departure = service_midnight + timedelta(seconds=departure_seconds)
        arrival = service_midnight + timedelta(seconds=arrival_seconds)
        elapsed_minutes = (arrival - departure).total_seconds() / 60
        service_number = trip.get("trip_short_name", "").strip() or trip["trip_id"]
        route_name = route.get("route_long_name", "").strip() or route["route_id"]
        itinerary_stops = []
        for row in trip_rows[start_index : end_index + 1]:
            stop = self._stops_by_id[row["stop_id"]]
            stop_arrival_seconds = self._parse_gtfs_time(
                row["arrival_time"], "stop_times.txt", 0
            )
            stop_departure_seconds = self._parse_gtfs_time(
                row["departure_time"], "stop_times.txt", 0
            )
            itinerary_stops.append({
                "stop_id": row["stop_id"],
                "stop_name": stop["stop_name"],
                "stop_sequence": int(row["stop_sequence"]),
                "arrival_at": (
                    service_midnight + timedelta(seconds=stop_arrival_seconds)
                ).isoformat(),
                "departure_at": (
                    service_midnight + timedelta(seconds=stop_departure_seconds)
                ).isoformat(),
                "latitude": float(stop["stop_lat"]),
                "longitude": float(stop["stop_lon"]),
                "pickup_type": row.get("pickup_type", "").strip() or "0",
                "drop_off_type": row.get("drop_off_type", "").strip() or "0",
            })

        return {
            "leg_index": leg_index,
            "route_id": route["route_id"],
            "trip_id": trip["trip_id"],
            "service_number": service_number,
            "route_name": route_name,
            "mode": "Bus",
            "origin_name": self._stops_by_id[origin_row["stop_id"]]["stop_name"],
            "destination_name": self._stops_by_id[destination_row["stop_id"]]["stop_name"],
            "departure_time": departure.strftime("%H:%M"),
            "arrival_time": arrival.strftime("%H:%M"),
            "scheduled_departure_at": departure.isoformat(),
            "scheduled_arrival_at": arrival.isoformat(),
            "effective_duration_min": elapsed_minutes,
            "fare_inr": None,
            "fare_verified": False,
            "timing_verified": True,
            "data_source": "TGSRTC static GTFS schedule",
            "stops": itinerary_stops,
            "route_geometry": None,
        }

    def _make_journey(
        self,
        trip: Dict[str, str],
        route: Dict[str, str],
        trip_rows: List[Dict[str, str]],
        start_index: int,
        end_index: int,
        service_date: date,
    ) -> Dict[str, Any]:
        leg = self._make_leg(
            trip, route, trip_rows, start_index, end_index, service_date, 1
        )
        origin_row = trip_rows[start_index]
        destination_row = trip_rows[end_index]
        elapsed_minutes = leg["effective_duration_min"]
        service_number = leg["service_number"]
        route_name = leg["route_name"]
        departure = datetime.fromisoformat(leg["scheduled_departure_at"])
        arrival = datetime.fromisoformat(leg["scheduled_arrival_at"])
        return {
            "journey_id": (
                f"TGSRTC_{trip['trip_id']}_"
                f"{origin_row['stop_sequence']}_{destination_row['stop_sequence']}"
            ),
            "journey_type": "Direct scheduled bus",
            "service_number": service_number,
            "title": f"{service_number} — {route_name} (Bus)",
            "service_name": route_name,
            "mode": "Bus",
            "modes_list": ["Bus"],
            "transfers_count": 0,
            "origin_name": leg["origin_name"],
            "destination_name": leg["destination_name"],
            "departure_time": leg["departure_time"],
            "arrival_time": leg["arrival_time"],
            "scheduled_departure_at": departure.isoformat(),
            "scheduled_arrival_at": arrival.isoformat(),
            "elapsed_duration_min": elapsed_minutes,
            "base_duration_min": elapsed_minutes,
            "effective_duration_min": elapsed_minutes,
            "delay_minutes": 0,
            "operational_status": (
                "Static scheduled service; operating status is not provided"
            ),
            "status_color": "#94a3b8",
            "last_update_time": self._metadata.get("retrieved_at_utc", "not recorded"),
            "data_source": "TGSRTC static GTFS schedule",
            "data_source_id": "tgsrtc_static_gtfs",
            "source_attribution": "Contains data provided by TGSRTC",
            "source_url": self._metadata.get("source_url"),
            "retrieved_at": self._metadata.get("retrieved_at_utc"),
            "feed_version": self.feed_info["feed_version"],
            "feed_valid_from": self.feed_start_date.isoformat(),
            "feed_valid_through": self.feed_end_date.isoformat(),
            "has_disruption": False,
            "primary_disruption": None,
            "total_fare_inr": None,
            "fare_verified": False,
            "fare_comparison_group": None,
            "timing_verified": True,
            "segments_verified": True,
            "legs": [leg],
            "stops": leg["stops"],
            "transfer_station": None,
            "layover_minutes": 0,
            "minimum_transfer_buffer_min": 0,
            "is_transfer_feasible": True,
            "transfer_risk_reason": "One scheduled trip; no transfer is required.",
            "route_geometry": None,
        }

    def search_journeys(
        self,
        origin_stop_id: str,
        destination_stop_id: str,
        travel_date: date,
        earliest_departure: time,
        service_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return only date-active trips whose ordered stop_times connect both stops."""
        if origin_stop_id == destination_stop_id:
            raise ValueError("Origin and destination must be different GTFS stops.")
        if origin_stop_id not in self._stops_by_id:
            raise ValueError(f"Unknown GTFS origin stop: {origin_stop_id}")
        if destination_stop_id not in self._stops_by_id:
            raise ValueError(f"Unknown GTFS destination stop: {destination_stop_id}")
        if not self.feed_start_date <= travel_date <= self.feed_end_date:
            return []
        earliest_local = datetime.combine(
            travel_date, earliest_departure, tzinfo=self.timezone
        )
        filter_text = (service_filter or "").strip().casefold()
        journeys: List[Dict[str, Any]] = []
        current_trip_id: Optional[str] = None
        trip_rows: List[Dict[str, str]] = []

        def consume_trip(trip_id: Optional[str], rows: List[Dict[str, str]]) -> None:
            if not trip_id or not rows:
                return
            trip = self._trips_by_id[trip_id]
            if not self._service_is_active(trip["service_id"], travel_date):
                return
            route = self._routes_by_id[trip["route_id"]]
            if filter_text and filter_text not in " ".join(
                (
                    trip_id,
                    trip.get("trip_short_name", ""),
                    route.get("route_id", ""),
                    route.get("route_long_name", ""),
                )
            ).casefold():
                return

            for start_index, row in enumerate(rows):
                if row["stop_id"] != origin_stop_id:
                    continue
                if row.get("pickup_type", "").strip() == "1":
                    continue
                departure_seconds = self._parse_gtfs_time(
                    row["departure_time"], "stop_times.txt", 0
                )
                departure_at = datetime.combine(
                    travel_date, time.min, tzinfo=self.timezone
                ) + timedelta(seconds=departure_seconds)
                if departure_at < earliest_local:
                    continue
                for end_index in range(start_index + 1, len(rows)):
                    destination_row = rows[end_index]
                    if (
                        destination_row["stop_id"] == destination_stop_id
                        and destination_row.get("drop_off_type", "").strip() != "1"
                    ):
                        journeys.append(self._make_journey(
                            trip,
                            route,
                            rows,
                            start_index,
                            end_index,
                            travel_date,
                        ))

        for row in self._stop_time_rows():
            trip_id = row.get("trip_id", "")
            if trip_id != current_trip_id:
                consume_trip(current_trip_id, trip_rows)
                current_trip_id = trip_id
                trip_rows = []
            trip_rows.append(row)
        consume_trip(current_trip_id, trip_rows)
        return journeys

    def diagnose_empty_search(
        self,
        origin_stop_id: str,
        destination_stop_id: str,
        travel_date: date,
        earliest_departure: time,
    ) -> Dict[str, Any]:
        """Diagnose why no direct scheduled trips matched the query."""
        if not self.feed_start_date <= travel_date <= self.feed_end_date:
            return {
                "reason": "outside_date_range",
                "message": (
                    f"Selected date {travel_date.isoformat()} is outside the feed coverage "
                    f"({self.feed_start_date.isoformat()} to {self.feed_end_date.isoformat()})."
                ),
                "reverse_trips_count": 0,
            }

        # Check reverse direction
        try:
            reverse_trips = self.search_journeys(
                origin_stop_id=destination_stop_id,
                destination_stop_id=origin_stop_id,
                travel_date=travel_date,
                earliest_departure=time(0, 0),
            )
        except Exception:
            reverse_trips = []

        def _journey_departure_key(trip: Dict[str, Any]) -> str:
            """Stable sort key: search_journeys returns file order, not departure order."""
            return str(trip.get("scheduled_departure_at") or trip.get("departure_time") or "")

        if reverse_trips:
            first_dep = min(
                reverse_trips, key=_journey_departure_key
            ).get("departure_time", "")
            orig_name = self._stops_by_id.get(origin_stop_id, {}).get("stop_name", origin_stop_id)
            dest_name = self._stops_by_id.get(destination_stop_id, {}).get("stop_name", destination_stop_id)
            return {
                "reason": "reverse_service_exists",
                "message": (
                    f"Direct scheduled service exists in the opposite direction "
                    f"({dest_name} → {orig_name}: {len(reverse_trips)} scheduled direct trip(s) "
                    f"found on {travel_date.isoformat()}, earliest departure at {first_dep}). "
                    "In municipal GTFS feeds, stops on opposite sides of a divided corridor have distinct stop IDs, "
                    "or services operate unidirectionally along that arterial."
                ),
                "reverse_trips_count": len(reverse_trips),
                "first_reverse_departure": first_dep,
            }

        # Check if trips exist earlier in the day
        try:
            earlier_trips = self.search_journeys(
                origin_stop_id=origin_stop_id,
                destination_stop_id=destination_stop_id,
                travel_date=travel_date,
                earliest_departure=time(0, 0),
            )
        except Exception:
            earlier_trips = []

        if earlier_trips:
            last_dep = max(
                earlier_trips, key=_journey_departure_key
            ).get("departure_time", "")
            return {
                "reason": "earlier_service_only",
                "message": (
                    f"{len(earlier_trips)} scheduled trip(s) operate on {travel_date.isoformat()}, "
                    f"but all departed before your requested departure time of {earliest_departure.strftime('%H:%M')} "
                    f"(the latest scheduled departure was {last_dep})."
                ),
                "reverse_trips_count": 0,
                "earlier_trips_count": len(earlier_trips),
            }

        orig_name = self._stops_by_id.get(origin_stop_id, {}).get("stop_name", origin_stop_id)
        dest_name = self._stops_by_id.get(destination_stop_id, {}).get("stop_name", destination_stop_id)
        return {
            "reason": "no_direct_service",
            "message": (
                f"No direct trip connects {orig_name} and {dest_name} in this direction in the "
                "TGSRTC static timetable. RouteRescue AI does not fabricate unverified transfers."
            ),
            "reverse_trips_count": 0,
        }

    def get_routes_for_stop(self, stop_id: str) -> List[Dict[str, str]]:
        """Return verified routes that serve the given stop_id."""
        route_ids = self._routes_by_stop.get(stop_id, set())
        results = []
        for rid in sorted(route_ids):
            route = self._routes_by_id.get(rid)
            if route:
                results.append({
                    "route_id": rid,
                    "route_name": route.get("route_long_name", "").strip() or rid,
                    "agency_id": route.get("agency_id", ""),
                })
        return results

    def find_nearby_stops(
        self,
        latitude: float,
        longitude: float,
        max_distance_km: float = 5.0,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find stops near the given coordinates using the spherical Haversine formula."""
        if not math.isfinite(latitude) or not math.isfinite(longitude):
            raise ValueError("Latitude and longitude must be valid finite numbers.")
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("Latitude or longitude out of geographic bounds.")

        r = 6371.0  # Earth radius in kilometers
        nearby = []
        for stop in self._stops:
            try:
                s_lat = float(stop["stop_lat"])
                s_lon = float(stop["stop_lon"])
            except (KeyError, ValueError):
                continue
            dlat = math.radians(s_lat - latitude)
            dlon = math.radians(s_lon - longitude)
            a = (
                math.sin(dlat / 2) ** 2
                + math.cos(math.radians(latitude))
                * math.cos(math.radians(s_lat))
                * math.sin(dlon / 2) ** 2
            )
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))
            dist_km = r * c
            if dist_km <= max_distance_km:
                nearby.append((dist_km, stop))

        nearby.sort(key=lambda item: item[0])
        results = []
        for dist_km, stop in nearby[:limit]:
            sid = stop["stop_id"]
            routes = self.get_routes_for_stop(sid)
            results.append({
                "stop_id": sid,
                "stop_name": stop["stop_name"],
                "latitude": float(stop["stop_lat"]),
                "longitude": float(stop["stop_lon"]),
                "distance_km": round(dist_km, 2),
                "distance_m": int(round(dist_km * 1000)),
                "routes_count": len(routes),
                "routes": routes,
                "label": f"{stop['stop_name']} ({sid})",
            })
        return results

    def search_stops_by_name(
        self,
        query: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search GTFS stops by case-insensitive name or ID substring."""
        query_folded = query.strip().casefold()
        if not query_folded:
            return []
        matches = []
        for stop in self._stops:
            if (
                query_folded in stop["stop_name"].casefold()
                or query_folded in stop["stop_id"].casefold()
            ):
                sid = stop["stop_id"]
                routes = self.get_routes_for_stop(sid)
                matches.append({
                    "stop_id": sid,
                    "stop_name": stop["stop_name"],
                    "latitude": float(stop["stop_lat"]),
                    "longitude": float(stop["stop_lon"]),
                    "routes_count": len(routes),
                    "routes": routes,
                    "label": f"{stop['stop_name']} ({sid})",
                })
                if len(matches) >= limit:
                    break
        return matches

    def _make_transfer_journey(
        self,
        leg1: Dict[str, Any],
        leg2: Dict[str, Any],
        transfer_stop_id: str,
        transfer_stop_name: str,
        layover_minutes: float,
        service_date: date,
    ) -> Dict[str, Any]:
        dep_dt = datetime.fromisoformat(leg1["scheduled_departure_at"])
        arr_dt = datetime.fromisoformat(leg2["scheduled_arrival_at"])
        total_duration = (arr_dt - dep_dt).total_seconds() / 60
        s1 = leg1["service_number"]
        s2 = leg2["service_number"]
        r1 = leg1["route_name"]
        r2 = leg2["route_name"]
        combined_stops = leg1["stops"] + leg2["stops"][1:]
        return {
            "journey_id": (
                f"TGSRTC_XFER_{leg1['trip_id']}_{leg2['trip_id']}_{transfer_stop_id}"
            ),
            "journey_type": "1-Transfer scheduled bus",
            "service_number": f"{s1} -> {s2}",
            "title": f"{s1} ({r1}) -> {s2} ({r2})",
            "service_name": f"{r1} + {r2}",
            "mode": "Bus",
            "modes_list": ["Bus"],
            "transfers_count": 1,
            "origin_name": leg1["origin_name"],
            "destination_name": leg2["destination_name"],
            "transfer_station": transfer_stop_name,
            "transfer_stop_id": transfer_stop_id,
            "departure_time": leg1["departure_time"],
            "arrival_time": leg2["arrival_time"],
            "scheduled_departure_at": leg1["scheduled_departure_at"],
            "scheduled_arrival_at": leg2["scheduled_arrival_at"],
            "elapsed_duration_min": total_duration,
            "base_duration_min": total_duration,
            "effective_duration_min": total_duration,
            "delay_minutes": 0,
            "layover_minutes": layover_minutes,
            "minimum_transfer_buffer_min": 5,
            "is_transfer_feasible": True,
            "transfer_risk_reason": (
                f"Scheduled transfer at {transfer_stop_name} with {int(round(layover_minutes))} min layover."
            ),
            "operational_status": (
                "Static scheduled services; operating status is not provided"
            ),
            "status_color": "#38bdf8",
            "last_update_time": self._metadata.get("retrieved_at_utc", "not recorded"),
            "data_source": "TGSRTC static GTFS schedule",
            "data_source_id": "tgsrtc_static_gtfs",
            "source_attribution": "Contains data provided by TGSRTC",
            "source_url": self._metadata.get("source_url"),
            "retrieved_at": self._metadata.get("retrieved_at_utc"),
            "feed_version": self.feed_info["feed_version"],
            "feed_valid_from": self.feed_start_date.isoformat(),
            "feed_valid_through": self.feed_end_date.isoformat(),
            "has_disruption": False,
            "primary_disruption": None,
            "total_fare_inr": None,
            "fare_verified": False,
            "fare_comparison_group": None,
            "timing_verified": True,
            "segments_verified": True,
            "legs": [leg1, leg2],
            "stops": combined_stops,
            "route_geometry": None,
        }

    def search_transfer_journeys(
        self,
        origin_stop_id: str,
        destination_stop_id: str,
        travel_date: date,
        earliest_departure: time,
        min_transfer_buffer_min: int = 5,
        max_layover_min: int = 120,
        max_options: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find valid 1-transfer journeys between two stops on the given date and time."""
        if origin_stop_id == destination_stop_id:
            raise ValueError("Origin and destination must be different GTFS stops.")
        if origin_stop_id not in self._stops_by_id:
            raise ValueError(f"Unknown GTFS origin stop: {origin_stop_id}")
        if destination_stop_id not in self._stops_by_id:
            raise ValueError(f"Unknown GTFS destination stop: {destination_stop_id}")
        if not self.feed_start_date <= travel_date <= self.feed_end_date:
            return []

        earliest_local = datetime.combine(
            travel_date, earliest_departure, tzinfo=self.timezone
        )
        service_midnight = datetime.combine(
            travel_date, time.min, tzinfo=self.timezone
        )

        leg1_candidates: Dict[str, List[Tuple[Dict[str, str], Dict[str, str], List[Dict[str, str]], int, int, datetime, datetime]]] = {}
        leg2_candidates: Dict[str, List[Tuple[Dict[str, str], Dict[str, str], List[Dict[str, str]], int, int, datetime, datetime]]] = {}

        current_trip_id: Optional[str] = None
        trip_rows: List[Dict[str, str]] = []

        def process_trip_for_transfer(tid: Optional[str], rows: List[Dict[str, str]]) -> None:
            if not tid or not rows:
                return
            trip = self._trips_by_id[tid]
            if not self._service_is_active(trip["service_id"], travel_date):
                return
            route = self._routes_by_id[trip["route_id"]]

            orig_idx = None
            for idx, r in enumerate(rows):
                if r["stop_id"] == origin_stop_id and r.get("pickup_type", "").strip() != "1":
                    dep_sec = self._parse_gtfs_time(r["departure_time"], "stop_times.txt", 0)
                    dep_dt = service_midnight + timedelta(seconds=dep_sec)
                    if dep_dt >= earliest_local:
                        orig_idx = idx
                        break
            if orig_idx is not None:
                for end_idx in range(orig_idx + 1, len(rows)):
                    transfer_stop_id = rows[end_idx]["stop_id"]
                    if transfer_stop_id != destination_stop_id and rows[end_idx].get("drop_off_type", "").strip() != "1":
                        arr_sec = self._parse_gtfs_time(rows[end_idx]["arrival_time"], "stop_times.txt", 0)
                        arr_dt = service_midnight + timedelta(seconds=arr_sec)
                        leg1_candidates.setdefault(transfer_stop_id, []).append(
                            (trip, route, rows, orig_idx, end_idx, dep_dt, arr_dt)
                        )

            dest_idx = None
            for idx, r in enumerate(rows):
                if r["stop_id"] == destination_stop_id and r.get("drop_off_type", "").strip() != "1":
                    dest_idx = idx
                    break
            if dest_idx is not None and dest_idx > 0:
                arr_sec = self._parse_gtfs_time(rows[dest_idx]["arrival_time"], "stop_times.txt", 0)
                dest_arr_dt = service_midnight + timedelta(seconds=arr_sec)
                for start_idx in range(0, dest_idx):
                    transfer_stop_id = rows[start_idx]["stop_id"]
                    if transfer_stop_id != origin_stop_id and rows[start_idx].get("pickup_type", "").strip() != "1":
                        dep_sec = self._parse_gtfs_time(rows[start_idx]["departure_time"], "stop_times.txt", 0)
                        board_dt = service_midnight + timedelta(seconds=dep_sec)
                        leg2_candidates.setdefault(transfer_stop_id, []).append(
                            (trip, route, rows, start_idx, dest_idx, board_dt, dest_arr_dt)
                        )

        for row in self._stop_time_rows():
            tid = row.get("trip_id")
            if tid != current_trip_id:
                process_trip_for_transfer(current_trip_id, trip_rows)
                current_trip_id = tid
                trip_rows = []
            trip_rows.append(row)
        process_trip_for_transfer(current_trip_id, trip_rows)

        common_stops = set(leg1_candidates.keys()) & set(leg2_candidates.keys())
        all_transfers: List[Tuple[float, float, str, Any, Any]] = []

        for x_stop in common_stops:
            x_stop_name = self._stops_by_id[x_stop]["stop_name"]
            for l1 in leg1_candidates[x_stop]:
                arr_x = l1[6]
                for l2 in leg2_candidates[x_stop]:
                    if l1[0]["trip_id"] == l2[0]["trip_id"]:
                        continue
                    dep_x = l2[5]
                    layover = (dep_x - arr_x).total_seconds() / 60
                    if min_transfer_buffer_min <= layover <= max_layover_min:
                        elapsed = (l2[6] - l1[5]).total_seconds() / 60
                        all_transfers.append((elapsed, layover, x_stop, l1, l2))

        all_transfers.sort(key=lambda item: item[0])

        journeys: List[Dict[str, Any]] = []
        seen_pairs: Set[Tuple[str, str, str]] = set()

        for elapsed, layover, x_stop, l1, l2 in all_transfers:
            pair_key = (l1[0]["trip_id"], l2[0]["trip_id"], x_stop)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            leg1_dict = self._make_leg(l1[0], l1[1], l1[2], l1[3], l1[4], travel_date, 1)
            leg2_dict = self._make_leg(l2[0], l2[1], l2[2], l2[3], l2[4], travel_date, 2)
            x_stop_name = self._stops_by_id[x_stop]["stop_name"]

            journey = self._make_transfer_journey(
                leg1_dict,
                leg2_dict,
                x_stop,
                x_stop_name,
                layover,
                travel_date,
            )
            journeys.append(journey)
            if len(journeys) >= max_options:
                break

        return journeys


