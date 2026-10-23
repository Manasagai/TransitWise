"""Configuration and verification status for external transport-data sources."""

from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


class SourceStatusCode(str, Enum):
    """Observable source states; configured does not imply operational success."""

    CONFIGURED_AND_TESTED = "configured_and_tested"
    MISSING_CREDENTIALS = "missing_credentials"
    SOURCE_NOT_CONFIGURED = "source_not_configured"
    TEMPORARILY_UNAVAILABLE = "temporarily_unavailable"
    UNSUPPORTED_ROUTE_OR_DATE = "unsupported_route_or_date"
    CONFIGURED_NOT_TESTED = "configured_not_tested"
    INVALID_RESPONSE = "invalid_response"


class SourceCheckOutcome(str, Enum):
    """Outcomes a real source adapter may report after a request."""

    SUCCESS = "success"
    TEMPORARILY_UNAVAILABLE = "temporarily_unavailable"
    UNSUPPORTED_ROUTE_OR_DATE = "unsupported_route_or_date"
    INVALID_RESPONSE = "invalid_response"


@dataclass(frozen=True)
class TrainSourceConfiguration:
    """Provider-neutral configuration facts; contains no credential values."""

    endpoint_configured: bool = False
    credentials_required: bool = False
    credentials_available: bool = False


@dataclass(frozen=True)
class SourceCheckResult:
    """Result of an adapter check after a genuine request and schema validation."""

    outcome: SourceCheckOutcome
    response_validated: bool = False
    detail: str = ""


@dataclass(frozen=True)
class DataSourceStatus:
    """Describe source availability without implying a request was made."""

    source_id: str
    mode: str
    status: SourceStatusCode
    source_name: str
    message: str
    missing_requirements: Tuple[str, ...]
    scheduled_data_available: bool = False
    live_status_available: bool = False


@dataclass(frozen=True)
class StaticFeedStatus:
    """Filesystem-only status for a user-supplied static feed archive."""

    source_id: str
    status: str
    feed_path: Path
    message: str
    attribution: str


def get_tgsrtc_gtfs_status(project_root: Optional[Path] = None) -> StaticFeedStatus:
    """Check for the user-provided TGSRTC archive without opening or trusting it."""
    root = project_root or Path(__file__).resolve().parent
    feed_path = root / "data" / "tgsrtc_gtfs.zip"
    is_present = feed_path.is_file()
    return StaticFeedStatus(
        source_id="tgsrtc_static_gtfs",
        status="present_pending_validation" if is_present else "missing",
        feed_path=feed_path,
        message=(
            "Feed archive found, but its contents, schema, coverage, and attribution "
            "metadata have not yet been validated."
            if is_present
            else "TGSRTC static GTFS archive is not present in the project."
        ),
        attribution="Contains data provided by TGSRTC",
    )


class InvalidTrainScheduleResponse(ValueError):
    """Raised when a provider response cannot safely represent train schedules."""


class UnsupportedTrainScheduleQuery(ValueError):
    """Raised when a provider explicitly reports no service for the request."""


def validate_train_schedule_response(
    payload: Mapping[str, Any],
    origin_station_code: str,
    destination_station_code: str,
    travel_date: date,
    station_id_map: Mapping[str, str],
) -> Tuple[Dict[str, Any], ...]:
    """Validate a provider-normalized response without performing network I/O.

    Providers need a separate, authorized adapter to map their schema into this
    contract. Returned items are scheduled-service records, not RouteEngine
    candidates or evidence of live operating status.
    """
    if not isinstance(payload, Mapping):
        raise InvalidTrainScheduleResponse("Response must be a mapping.")

    outcome = payload.get("outcome")
    if outcome == SourceCheckOutcome.UNSUPPORTED_ROUTE_OR_DATE.value:
        raise UnsupportedTrainScheduleQuery(
            "Source reports no service for the requested route or date."
        )
    if outcome != SourceCheckOutcome.SUCCESS.value:
        raise InvalidTrainScheduleResponse("Response outcome is not a supported success.")

    source = payload.get("source")
    if not isinstance(source, Mapping) or any(
        not isinstance(source.get(field), str) or not source[field].strip()
        for field in ("source_id", "source_name", "attribution")
    ):
        raise InvalidTrainScheduleResponse(
            "Source metadata requires source_id, source_name, and attribution."
        )

    if payload.get("travel_date") != travel_date.isoformat():
        raise InvalidTrainScheduleResponse("Response travel_date does not match the request.")
    try:
        valid_from = date.fromisoformat(payload["timetable_valid_from"])
        valid_through = date.fromisoformat(payload["timetable_valid_through"])
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidTrainScheduleResponse(
            "Timetable validity requires ISO dates timetable_valid_from and timetable_valid_through."
        ) from error
    if valid_from > valid_through:
        raise InvalidTrainScheduleResponse(
            "Timetable validity range is reversed."
        )
    if not valid_from <= travel_date <= valid_through:
        raise UnsupportedTrainScheduleQuery(
            "Requested travel_date falls outside the timetable validity range."
        )

    retrieved_at_raw = payload.get("retrieved_at")
    if not isinstance(retrieved_at_raw, str):
        raise InvalidTrainScheduleResponse("Response requires an ISO-8601 retrieved_at timestamp.")
    try:
        retrieved_at = datetime.fromisoformat(retrieved_at_raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise InvalidTrainScheduleResponse("retrieved_at is not a valid ISO-8601 timestamp.") from error
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() != timezone.utc.utcoffset(retrieved_at):
        raise InvalidTrainScheduleResponse("retrieved_at must be expressed in UTC.")
    if retrieved_at > datetime.now(timezone.utc):
        raise InvalidTrainScheduleResponse("retrieved_at cannot be in the future.")

    services = payload.get("services")
    if not isinstance(services, list):
        raise InvalidTrainScheduleResponse("Response services must be a list.")
    if not services:
        raise UnsupportedTrainScheduleQuery(
            "Source returned no scheduled services for the requested route or date."
        )

    normalized = []
    for index, service in enumerate(services):
        if not isinstance(service, Mapping):
            raise InvalidTrainScheduleResponse(f"Service at index {index} must be an object.")
        required_text = (
            "service_id",
            "service_number",
            "origin_station_code",
            "destination_station_code",
            "origin_name",
            "destination_name",
            "departure_at",
            "arrival_at",
        )
        if any(
            not isinstance(service.get(field), str) or not service[field].strip()
            for field in required_text
        ):
            raise InvalidTrainScheduleResponse(
                f"Service at index {index} is missing a required identifier or schedule field."
            )
        if service.get("travel_date") != travel_date.isoformat():
            raise InvalidTrainScheduleResponse(
                f"Service at index {index} travel_date does not match the request."
            )

        origin_code = service["origin_station_code"]
        destination_code = service["destination_station_code"]
        if origin_code != origin_station_code or destination_code != destination_station_code:
            raise InvalidTrainScheduleResponse(
                f"Service at index {index} does not match the requested station codes."
            )
        if origin_code not in station_id_map or destination_code not in station_id_map:
            raise InvalidTrainScheduleResponse(
                f"Service at index {index} has an unmapped station code."
            )

        try:
            departure = datetime.fromisoformat(service["departure_at"].replace("Z", "+00:00"))
            arrival = datetime.fromisoformat(service["arrival_at"].replace("Z", "+00:00"))
        except ValueError as error:
            raise InvalidTrainScheduleResponse(
                f"Service at index {index} has an invalid scheduled timestamp."
            ) from error
        if departure.tzinfo is None or arrival.tzinfo is None:
            raise InvalidTrainScheduleResponse(
                f"Service at index {index} timestamps must include timezone information."
            )
        if departure.date() != travel_date or arrival <= departure:
            raise InvalidTrainScheduleResponse(
                f"Service at index {index} has inconsistent scheduled dates or times."
            )

        normalized.append({
            "mode": "Train",
            "service_id": service["service_id"],
            "service_number": service["service_number"],
            "origin_station_code": origin_code,
            "destination_station_code": destination_code,
            "origin_id": station_id_map[origin_code],
            "destination_id": station_id_map[destination_code],
            "origin_name": service["origin_name"],
            "destination_name": service["destination_name"],
            "travel_date": travel_date.isoformat(),
            "scheduled_departure_at": departure.isoformat(),
            "scheduled_arrival_at": arrival.isoformat(),
            "data_source": source["source_name"],
            "source_id": source["source_id"],
            "source_attribution": source["attribution"],
            "retrieved_at": retrieved_at.astimezone(timezone.utc).isoformat(),
            "timetable_valid_from": valid_from.isoformat(),
            "timetable_valid_through": valid_through.isoformat(),
            "live_status": None,
            "fare": None,
            "route_geometry": None,
        })
    return tuple(normalized)


def get_train_schedule_source_status(
    configuration: Optional[TrainSourceConfiguration] = None,
    check_result: Optional[SourceCheckResult] = None,
) -> DataSourceStatus:
    """Resolve truthful status from explicit configuration and a validated check.

    This function performs no network requests. A success status requires an
    adapter-provided success result whose response was validated.
    """
    config = configuration or TrainSourceConfiguration()
    missing_requirements = (
        "Approved machine-readable train timetable source and documented endpoint",
        "Terms or authorization permitting automated access and reuse",
        "Provider schema, station/service identifier mapping, and schedule conventions",
    )

    if not config.endpoint_configured:
        status = SourceStatusCode.SOURCE_NOT_CONFIGURED
        message = (
            "No approved train timetable endpoint or dataset is configured. "
            "The passenger-facing enquiry page is not treated as an API."
        )
    elif config.credentials_required and not config.credentials_available:
        status = SourceStatusCode.MISSING_CREDENTIALS
        message = "The configured provider requires credentials, but none are available."
        missing_requirements += ("Provider credentials supplied through an approved secret store",)
    elif check_result is None:
        status = SourceStatusCode.CONFIGURED_NOT_TESTED
        message = (
            "An endpoint is configured, but no authorized adapter has made and validated "
            "a genuine request."
        )
    elif check_result.outcome == SourceCheckOutcome.TEMPORARILY_UNAVAILABLE:
        status = SourceStatusCode.TEMPORARILY_UNAVAILABLE
        message = check_result.detail or "The configured source is temporarily unavailable."
    elif check_result.outcome == SourceCheckOutcome.UNSUPPORTED_ROUTE_OR_DATE:
        status = SourceStatusCode.UNSUPPORTED_ROUTE_OR_DATE
        message = check_result.detail or "The source does not support the requested route or date."
    elif (
        check_result.outcome == SourceCheckOutcome.SUCCESS
        and check_result.response_validated
    ):
        status = SourceStatusCode.CONFIGURED_AND_TESTED
        message = "A genuine source request succeeded and its response passed schema validation."
    else:
        status = SourceStatusCode.INVALID_RESPONSE
        message = check_result.detail or (
            "The source response did not pass validation; it must not be used for journeys."
        )

    return DataSourceStatus(
        source_id="train_schedule",
        mode="Train",
        status=status,
        source_name="External train schedule source",
        message=message,
        missing_requirements=missing_requirements,
        scheduled_data_available=False,
        live_status_available=False,
    )
