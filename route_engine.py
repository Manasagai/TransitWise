"""
RouteRescue AI — Route & Alternative Comparison Engine
Module: route_engine.py

Constructs the multi-modal South Indian transit network using NetworkX, assesses
simulated disruptions across Bus, Train, and Flight, computes adjusted travel times,
evaluates feasible multi-leg transfers, performs what-if delay simulations,
and delivers explainable rescue recommendations and transparent risk indicators.
"""

import os
import datetime
import math
from numbers import Real
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go


class RouteEngine:
    """Manages transit network topology, transfer feasibility, and disruption evaluation."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        allow_unverified_data: bool = False
    ):
        """
        Initialize the route engine and load datasets.
        
        Args:
            data_dir: Path to data directory. Defaults to 'data/' relative to script.
        """
        if data_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(base_dir, "data")
        else:
            self.data_dir = data_dir

        self.allow_unverified_data = allow_unverified_data
        self.df_stops = pd.DataFrame()
        self.df_routes = pd.DataFrame()
        self.df_schedules = pd.DataFrame()
        self.df_disruptions = pd.DataFrame()
        self.graph = nx.DiGraph()

        self._load_datasets()
        self._build_network_graph()

    def _load_datasets(self) -> None:
        """Load all CSV files with error handling and fallback defaults."""
        files = {
            "stops": "stops.csv",
            "routes": "routes.csv",
            "schedules": "schedules.csv",
            "disruptions": "disruptions.csv",
        }

        for attr, filename in files.items():
            filepath = os.path.join(self.data_dir, filename)
            try:
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    setattr(self, f"df_{attr}", df)
                else:
                    print(f"[RouteEngine Warning] Missing file: {filepath}")
                    setattr(self, f"df_{attr}", pd.DataFrame())
            except Exception as e:
                print(f"[RouteEngine Error] Failed loading {filename}: {e}")
                setattr(self, f"df_{attr}", pd.DataFrame())

    def _build_network_graph(self) -> None:
        """Construct NetworkX directed graph from stops and routes."""
        self.graph.clear()

        # Add stop nodes
        if not self.df_stops.empty:
            for _, row in self.df_stops.iterrows():
                self.graph.add_node(
                    row["stop_id"],
                    name=row["stop_name"],
                    city=row["city"],
                    state=row.get("state", "South India"),
                    lat=float(row["latitude"]),
                    lon=float(row["longitude"]),
                    hub_type=row.get("hub_type", "Station"),
                    description=row.get("description", "")
                )

        # Add route edges
        if not self.df_routes.empty:
            for _, row in self.df_routes.iterrows():
                u = row["origin_id"]
                v = row["destination_id"]
                self.graph.add_edge(
                    u,
                    v,
                    route_id=row["route_id"],
                    route_name=row["route_name"],
                    mode=row["mode"],
                    distance_km=float(row["distance_km"]),
                    base_duration_min=int(row["base_duration_min"]),
                    base_fare_inr=float(row["base_fare_inr"]),
                    frequency=row.get("frequency_desc", "Regular"),
                    intermediate_stops=row.get("intermediate_stops", "")
                )

    def get_stops(self) -> List[Dict[str, str]]:
        """Return list of all stops for general dropdown selection."""
        if self.df_stops.empty:
            return []
        stops_list = []
        for _, row in self.df_stops.iterrows():
            stops_list.append({
                "stop_id": row["stop_id"],
                "stop_name": row["stop_name"],
                "city": row["city"],
                "state": row.get("state", ""),
                "mode_category": row.get("mode_category", "Bus"),
                "label": f"{row['stop_name']} ({row['city']})"
            })
        return sorted(stops_list, key=lambda x: (x["city"], x["stop_name"]))

    def get_stops_by_mode(self, mode: str) -> List[Dict[str, str]]:
        """
        Return stops filtered strictly by mode category ('Bus', 'Train', 'Flight').
        Guarantees airports never appear in train/bus lists and vice versa.
        """
        if self.df_stops.empty:
            return []
        
        mode_clean = mode.strip().lower()
        if "mode_category" in self.df_stops.columns:
            filtered = self.df_stops[self.df_stops["mode_category"].str.lower() == mode_clean]
        else:
            filtered = self.df_stops

        stops_list = []
        for _, row in filtered.iterrows():
            stops_list.append({
                "stop_id": row["stop_id"],
                "stop_name": row["stop_name"],
                "city": row["city"],
                "state": row.get("state", ""),
                "hub_type": row.get("hub_type", ""),
                "mode_category": row.get("mode_category", mode),
                "label": f"{row['stop_name']} ({row['city']})"
            })
        return sorted(stops_list, key=lambda x: x["stop_name"])

    def get_cities(self) -> List[str]:
        """Return distinct list of cities covered in network."""
        if self.df_stops.empty:
            return []
        return sorted(self.df_stops["city"].unique().tolist())

    def get_stops_for_city(self, city: str) -> List[Dict[str, str]]:
        """Return all stops situated in a given city."""
        all_stops = self.get_stops()
        return [s for s in all_stops if s["city"].lower() == city.lower()]

    def get_active_disruptions(self, include_unverified: bool = False) -> pd.DataFrame:
        """Return disruption records only when explicitly requested for demo use."""
        if not include_unverified:
            return pd.DataFrame()
        if self.df_disruptions.empty:
            return pd.DataFrame()
        return self.df_disruptions.copy()

    def get_network_summary(self) -> Dict[str, Any]:
        """Return high-level summary metrics of the simulated network."""
        active_disruptions_count = 0
        if not self.df_disruptions.empty and "status" in self.df_disruptions.columns:
            active_disruptions_count = len(
                self.df_disruptions[self.df_disruptions["status"].str.lower() == "active"]
            )

        states_covered = (
            self.df_stops["state"].nunique()
            if not self.df_stops.empty and "state" in self.df_stops.columns
            else 5
        )

        modes_covered = (
            self.df_routes["mode"].unique().tolist()
            if not self.df_routes.empty and "mode" in self.df_routes.columns
            else ["Bus", "Train", "Flight"]
        )

        return {
            "total_stops": len(self.df_stops),
            "total_routes": len(self.df_routes),
            "total_schedules": len(self.df_schedules),
            "active_disruptions": active_disruptions_count,
            "states_count": states_covered,
            "modes": modes_covered,
            "network_scope": "South Indian Regional Corridor (TS, AP, KA, TN, KL)",
            "simulation_mode": "Simulated Benchmark Data"
        }

    def _get_disruption_for_route(self, route_id: str) -> Optional[Dict[str, Any]]:
        """Check if an active simulated disruption affects the given route."""
        if self.df_disruptions.empty:
            return None

        matches = self.df_disruptions[
            (self.df_disruptions["route_id"] == route_id) &
            (self.df_disruptions["status"].str.lower() == "active")
        ]

        if not matches.empty:
            dis = matches.iloc[0]
            return {
                "disruption_id": dis.get("disruption_id", ""),
                "severity": dis.get("severity", "Medium"),
                "delay_minutes": int(dis.get("delay_minutes", 0)),
                "incident_type": dis.get("incident_type", "Operational Advisory"),
                "description": dis.get("description", ""),
                "alternative_recommendation": dis.get("alternative_recommendation", ""),
                "location_segment": dis.get("location_segment", "En-route")
            }
        return None

    @staticmethod
    def _time_str_to_minutes(time_str: str) -> int:
        """Convert 'HH:MM' string to minutes from midnight."""
        try:
            parts = time_str.strip().split(":")
            return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            return 8 * 60  # Default 08:00 AM

    @staticmethod
    def _minutes_to_time_str(minutes: int) -> str:
        """Convert minutes from midnight to 'HH:MM' format."""
        norm_min = minutes % (24 * 60)
        hh = norm_min // 60
        mm = norm_min % 60
        return f"{hh:02d}:{mm:02d}"

    def _get_min_transfer_buffer(self, mode1: str, mode2: str, same_station: bool) -> int:
        """Determine minimum realistic transfer buffer in minutes."""
        if mode1 == "Flight" or mode2 == "Flight":
            return 75  # Flight requires check-in / baggage collection
        if same_station and mode1 == mode2:
            return 30  # Same station rail-rail or bus-bus
        return 45  # Inter-modal or terminal transfer

    def search_journeys(
        self,
        origin_id: str,
        destination_id: str,
        departure_time_str: str = "08:00",
        mode_filter: str = "All Modes",
        preference: str = "fastest",
        service_filter: Optional[str] = None,
        widen_window: bool = False,
        allow_unverified_data: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for direct and 1-transfer journeys between two stops.

        Args:
            origin_id: canonical stop_id of the origin stop
            destination_id: canonical stop_id of the destination stop
            departure_time_str: earliest acceptable departure (HH:MM); treated as lower bound
            mode_filter: 'Bus', 'Train', 'Flight', or 'All Modes'
            preference: 'fastest', 'cheapest', or 'fewest_transfers'
            service_filter: optional service number / route name substring filter
            widen_window: if True, search all available slots (ignores departure_time lower bound)
                          Used for "Find Next Available Service" feature.

        Returns:
            List of feasible candidate dictionaries only when unverified data is
            explicitly enabled for demo use. Normal mode rejects the local CSV data.
            Errors are raised rather than silently swallowed.
        """
        if self.df_routes.empty:
            raise RuntimeError(
                "ROUTES_MISSING: routes.csv could not be loaded. "
                "The search cannot proceed without route data."
            )
        if not (
            self.allow_unverified_data
            if allow_unverified_data is None
            else allow_unverified_data
        ):
            raise RuntimeError(
                "UNVERIFIED_LOCAL_DATA: journey search is disabled because the available "
                "CSV route, schedule, and disruption records are illustrative and have no "
                "verified operational source."
            )
        if self.df_stops.empty:
            raise RuntimeError(
                "STOPS_MISSING: stops.csv could not be loaded. "
                "The search cannot proceed without stop data."
            )

        dep_min = self._time_str_to_minutes(departure_time_str)
        journeys: List[Dict[str, Any]] = []

        # Validate stop IDs exist in the network
        known_stop_ids = set(self.df_stops["stop_id"].tolist())
        if origin_id not in known_stop_ids:
            raise ValueError(f"UNKNOWN_ORIGIN: stop_id '{origin_id}' not found in stops dataset.")
        if destination_id not in known_stop_ids:
            raise ValueError(f"UNKNOWN_DEST: stop_id '{destination_id}' not found in stops dataset.")

        # ──────────────────────────────────────────────────────────────────
        # 1. Direct Journeys
        # ──────────────────────────────────────────────────────────────────
        direct_df = self.df_routes[
            (self.df_routes["origin_id"] == origin_id) &
            (self.df_routes["destination_id"] == destination_id)
        ]

        if mode_filter != "All Modes":
            direct_df = direct_df[direct_df["mode"].str.lower() == mode_filter.lower()]

        if service_filter and service_filter.strip():
            s_q = service_filter.strip().lower()
            mask = (
                direct_df["service_number"].astype(str).str.lower().str.contains(s_q, na=False) |
                direct_df["route_name"].astype(str).str.lower().str.contains(s_q, na=False)
            )
            direct_df = direct_df[mask]

        for _, r in direct_df.iterrows():
            route_id = r["route_id"]
            svc_num = str(r.get("service_number", route_id))
            base_dur = int(r["base_duration_min"])
            fare = float(r["base_fare_inr"])
            mode = r["mode"]

            disruption = self._get_disruption_for_route(route_id)
            delay = disruption["delay_minutes"] if disruption else 0
            effective_dur = base_dur + delay

            # ── Schedule matching ──────────────────────────────────────────
            # Treat departure_time_str as a STRICT lower bound.
            # We only accept departures >= requested time.
            # If widen_window=True, accept any departure (for "Find Next Available").
            next_dep, sched_source = self._find_next_departure(
                route_id=route_id,
                requested_time_str=departure_time_str,
                widen_window=widen_window
            )

            if next_dep is None:
                # Confirmed: schedule data exists for this route but
                # all slots are before the requested time. Skip silently.
                continue

            act_dep_min = self._time_str_to_minutes(next_dep)
            arr_min = act_dep_min + effective_dur
            arrival_time_str = self._minutes_to_time_str(arr_min)

            op_status = (
                f"Illustrative demo delay scenario (+{delay}m)"
                if delay > 0
                else "No delay scenario recorded (not a live status)"
            )
            status_color = "#ef4444" if delay >= 30 else ("#f59e0b" if delay > 0 else "#10b981")

            leg_data = {
                "leg_index": 1,
                "route_id": route_id,
                "service_number": svc_num,
                "route_name": r["route_name"],
                "mode": mode,
                "origin_name": r["origin_name"],
                "destination_name": r["destination_name"],
                "departure_time": next_dep,
                "arrival_time": arrival_time_str,
                "base_duration_min": base_dur,
                "delay_minutes": delay,
                "effective_duration_min": effective_dur,
                "fare_inr": fare,
                "data_source": sched_source,
                "timing_verified": False,
                "fare_verified": False,
                "disruption_info": disruption,
                "has_disruption": bool(disruption)
            }

            journeys.append({
                "journey_id": f"DIR_{route_id}_{next_dep.replace(':','')}",
                "journey_type": "Direct Route",
                "service_number": svc_num,
                "title": f"{svc_num} — {r['route_name']} ({mode})",
                "service_name": r["route_name"],
                "mode": mode,
                "modes_list": [mode],
                "transfers_count": 0,
                "origin_name": r["origin_name"],
                "destination_name": r["destination_name"],
                "departure_time": next_dep,
                "arrival_time": arrival_time_str,
                "base_duration_min": base_dur,
                "effective_duration_min": effective_dur,
                "delay_minutes": delay,
                "operational_status": op_status,
                "status_color": status_color,
                "last_update_time": "Project Dataset (Static Timetable)",
                "data_source": sched_source,
                "has_disruption": bool(disruption),
                "primary_disruption": disruption,
                "total_fare_inr": fare,
                "fare_verified": False,
                "timing_verified": False,
                "segments_verified": False,
                "legs": [leg_data],
                "transfer_station": None,
                "layover_minutes": 0,
                "is_transfer_feasible": True,
                "transfer_risk_reason": "Direct non-stop service without connection risk."
            })

        # ──────────────────────────────────────────────────────────────────
        # 2. Feasible 1-Transfer Connecting Journeys (via NetworkX)
        # ──────────────────────────────────────────────────────────────────
        if self.graph.has_node(origin_id) and self.graph.has_node(destination_id):
            intermediate_candidates = (
                set(self.graph.successors(origin_id)) &
                set(self.graph.predecessors(destination_id))
            )

            for mid_id in intermediate_candidates:
                if mid_id in (origin_id, destination_id):
                    continue

                mid_node = self.graph.nodes[mid_id]
                mid_stop_name = mid_node.get("name", "Transfer Interchange")

                leg1_routes = self.df_routes[
                    (self.df_routes["origin_id"] == origin_id) &
                    (self.df_routes["destination_id"] == mid_id)
                ]
                leg2_routes = self.df_routes[
                    (self.df_routes["origin_id"] == mid_id) &
                    (self.df_routes["destination_id"] == destination_id)
                ]

                for _, r1 in leg1_routes.iterrows():
                    for _, r2 in leg2_routes.iterrows():
                        # Mode filter: at least one leg must match requested mode
                        if mode_filter != "All Modes":
                            if (r1["mode"].lower() != mode_filter.lower() and
                                    r2["mode"].lower() != mode_filter.lower()):
                                continue

                        # Leg 1 Disruption & Duration
                        d1 = self._get_disruption_for_route(r1["route_id"])
                        delay1 = d1["delay_minutes"] if d1 else 0
                        dur1 = int(r1["base_duration_min"]) + delay1
                        fare1 = float(r1["base_fare_inr"])

                        # Leg 1 Departure
                        dep1_str, src1 = self._find_next_departure(
                            r1["route_id"], departure_time_str, widen_window
                        )
                        if dep1_str is None:
                            continue

                        dep1_min = self._time_str_to_minutes(dep1_str)
                        arr1_min = dep1_min + dur1

                        # Leg 2 Disruption & Duration
                        d2 = self._get_disruption_for_route(r2["route_id"])
                        delay2 = d2["delay_minutes"] if d2 else 0
                        dur2 = int(r2["base_duration_min"]) + delay2
                        fare2 = float(r2["base_fare_inr"])

                        # Evaluate realistic connection window
                        same_station = (r1["destination_id"] == r2["origin_id"])
                        min_buffer = self._get_min_transfer_buffer(r1["mode"], r2["mode"], same_station)
                        earliest_dep2_min = arr1_min + min_buffer

                        # Find Leg 2 departure after arrival + buffer
                        dep2_str, src2 = self._find_next_departure(
                            r2["route_id"],
                            self._minutes_to_time_str(earliest_dep2_min),
                            widen_window=True  # always search forward for connecting leg
                        )

                        if dep2_str is None:
                            continue

                        matched_dep2_min = self._time_str_to_minutes(dep2_str)
                        layover_min = matched_dep2_min - arr1_min
                        is_feasible = (layover_min >= min_buffer) and (layover_min <= 360)

                        if not is_feasible:
                            continue

                        arr2_min = matched_dep2_min + dur2
                        arr2_str = self._minutes_to_time_str(arr2_min)

                        total_duration_min = dur1 + layover_min + dur2
                        total_fare = fare1 + fare2
                        total_delay = delay1 + delay2
                        primary_dis = d1 or d2

                        leg1_item = {
                            "leg_index": 1,
                            "route_id": r1["route_id"],
                            "service_number": str(r1.get("service_number", r1["route_id"])),
                            "route_name": r1["route_name"],
                            "mode": r1["mode"],
                            "origin_name": r1["origin_name"],
                            "destination_name": r1["destination_name"],
                            "departure_time": dep1_str,
                            "arrival_time": self._minutes_to_time_str(arr1_min),
                            "base_duration_min": int(r1["base_duration_min"]),
                            "delay_minutes": delay1,
                            "effective_duration_min": dur1,
                            "fare_inr": fare1,
                            "data_source": src1,
                            "timing_verified": False,
                            "fare_verified": False,
                            "disruption_info": d1,
                            "has_disruption": bool(d1)
                        }

                        leg2_item = {
                            "leg_index": 2,
                            "route_id": r2["route_id"],
                            "service_number": str(r2.get("service_number", r2["route_id"])),
                            "route_name": r2["route_name"],
                            "mode": r2["mode"],
                            "origin_name": r2["origin_name"],
                            "destination_name": r2["destination_name"],
                            "departure_time": dep2_str,
                            "arrival_time": arr2_str,
                            "base_duration_min": int(r2["base_duration_min"]),
                            "delay_minutes": delay2,
                            "effective_duration_min": dur2,
                            "fare_inr": fare2,
                            "data_source": src2,
                            "timing_verified": False,
                            "fare_verified": False,
                            "disruption_info": d2,
                            "has_disruption": bool(d2)
                        }

                        if layover_min < (min_buffer + 15):
                            risk_reason = (
                                f"Tight transfer layover ({layover_min}m available vs "
                                f"{min_buffer}m min buffer at {mid_stop_name})."
                            )
                        else:
                            risk_reason = (
                                f"Comfortable transfer window ({layover_min}m layover at {mid_stop_name})."
                            )

                        svc1 = str(r1.get("service_number", r1["route_id"]))
                        svc2 = str(r2.get("service_number", r2["route_id"]))
                        conn_svc = f"{svc1} + {svc2}"
                        op_status = (
                            f"Illustrative demo delay scenario (+{total_delay}m)"
                            if total_delay > 0
                            else "No delay scenario recorded (not a live status)"
                        )
                        status_color = "#ef4444" if total_delay >= 30 else ("#f59e0b" if total_delay > 0 else "#10b981")

                        journeys.append({
                            "journey_id": f"CONN_{r1['route_id']}_{r2['route_id']}_{dep1_str.replace(':','')}",
                            "journey_type": f"1-Transfer via {mid_node.get('city', 'Hub')}",
                            "service_number": conn_svc,
                            "title": f"{conn_svc} — {r1['route_name']} + {r2['route_name']}",
                            "service_name": f"{r1['route_name']} + {r2['route_name']}",
                            "mode": f"{r1['mode']} + {r2['mode']}",
                            "modes_list": [r1["mode"], r2["mode"]],
                            "transfers_count": 1,
                            "origin_name": r1["origin_name"],
                            "destination_name": r2["destination_name"],
                            "departure_time": dep1_str,
                            "arrival_time": arr2_str,
                            "base_duration_min": int(r1["base_duration_min"]) + int(r2["base_duration_min"]),
                            "effective_duration_min": total_duration_min,
                            "delay_minutes": total_delay,
                            "operational_status": op_status,
                            "status_color": status_color,
                            "last_update_time": "Project Dataset (Static Timetable)",
                            "data_source": f"{src1} / {src2}",
                            "has_disruption": bool(primary_dis),
                            "primary_disruption": primary_dis,
                            "total_fare_inr": total_fare,
                            "fare_verified": False,
                            "timing_verified": False,
                            "segments_verified": False,
                            "legs": [leg1_item, leg2_item],
                            "transfer_station": mid_stop_name,
                            "layover_minutes": layover_min,
                            "min_transfer_buffer": min_buffer,
                            "minimum_transfer_buffer_min": min_buffer,
                            "is_transfer_feasible": is_feasible,
                            "transfer_risk_reason": risk_reason
                        })

        # ──────────────────────────────────────────────────────────────────
        # 3. Deduplicate: keep only the earliest slot per unique route
        # ──────────────────────────────────────────────────────────────────
        seen_route_ids: dict = {}
        deduped = []
        for j in journeys:
            key = j["journey_id"].rsplit("_", 1)[0]  # strip time suffix
            dep = j["departure_time"]
            if key not in seen_route_ids or dep < seen_route_ids[key]:
                seen_route_ids[key] = dep
                deduped.append(j)
        journeys = deduped

        return self.rank_journeys(journeys, preference)

    def rank_journeys(
        self,
        journeys: List[Dict[str, Any]],
        preference: str = "fastest"
    ) -> List[Dict[str, Any]]:
        """Rank existing feasible candidates without inventing or altering journey data."""
        preference_key = preference.strip().lower()
        if preference_key not in {"fastest", "cheapest", "fewest_transfers", "earliest_departure", "earliest"}:
            preference_key = "fastest"

        def numeric_value(value: Any) -> Optional[float]:
            if isinstance(value, Real) and not isinstance(value, bool):
                number = float(value)
                if math.isfinite(number):
                    return number
            return None

        def duration(journey: Dict[str, Any]) -> Optional[float]:
            if journey.get("timing_verified") is not True:
                return None
            departure = journey.get("departure_time")
            arrival = journey.get("arrival_time")
            if not isinstance(departure, str) or not isinstance(arrival, str):
                return None
            try:
                departure_time = datetime.datetime.strptime(departure, "%H:%M")
                arrival_time = datetime.datetime.strptime(arrival, "%H:%M")
            except ValueError:
                return None
            declared_elapsed = numeric_value(journey.get("elapsed_duration_min"))
            if declared_elapsed is not None and declared_elapsed >= 0:
                return declared_elapsed
            elapsed = (
                arrival_time.hour * 60 + arrival_time.minute
                - departure_time.hour * 60 - departure_time.minute
            )
            if elapsed < 0:
                return None
            return float(elapsed)

        def transfer_count(journey: Dict[str, Any]) -> Optional[int]:
            legs = journey.get("legs")
            if journey.get("segments_verified") is True and isinstance(legs, list) and legs:
                return len(legs) - 1
            return None

        def fare(journey: Dict[str, Any]) -> Optional[float]:
            value = numeric_value(journey.get("total_fare_inr"))
            return value if value is not None and value >= 0 else None

        fare_groups = {
            journey.get("fare_comparison_group")
            for journey in journeys
            if journey.get("fare_verified") is True
            and isinstance(journey.get("fare_comparison_group"), str)
            and journey["fare_comparison_group"].strip()
        }
        fares_comparable = (
            bool(journeys)
            and len(fare_groups) == 1
            and all(
                journey.get("fare_verified") is True and fare(journey) is not None
                for journey in journeys
            )
        )
        durations_complete = all(duration(journey) is not None for journey in journeys)
        transfers_complete = all(transfer_count(journey) is not None for journey in journeys)

        indexed_journeys = list(enumerate(journeys))
        if preference_key == "cheapest" and not fares_comparable:
            indexed_journeys.sort(
                key=lambda item: (
                    str(item[1].get("journey_id", "")),
                    item[0]
                )
            )
            ranking_status = "unavailable"
        elif preference_key == "fastest":
            indexed_journeys.sort(
                key=lambda item: (
                    duration(item[1]) is None,
                    duration(item[1]) if duration(item[1]) is not None else math.inf,
                    transfer_count(item[1]) if transfer_count(item[1]) is not None else math.inf,
                    str(item[1].get("journey_id", "")),
                    item[0]
                )
            )
            ranking_status = "complete" if durations_complete else "incomplete"
        elif preference_key in {"earliest_departure", "earliest"}:
            indexed_journeys.sort(
                key=lambda item: (
                    str(item[1].get("departure_time") or item[1].get("scheduled_departure_at") or ""),
                    duration(item[1]) if duration(item[1]) is not None else math.inf,
                    str(item[1].get("journey_id", "")),
                    item[0]
                )
            )
            ranking_status = "complete"
        elif preference_key == "cheapest":
            indexed_journeys.sort(
                key=lambda item: (
                    fare(item[1]),
                    str(item[1].get("journey_id", "")),
                    item[0]
                )
            )
            ranking_status = "complete"
        else:
            indexed_journeys.sort(
                key=lambda item: (
                    transfer_count(item[1]) is None,
                    transfer_count(item[1]) if transfer_count(item[1]) is not None else math.inf,
                    str(item[1].get("journey_id", "")),
                    item[0]
                )
            )
            ranking_status = "complete" if transfers_complete else "incomplete"

        ranked = []
        total = len(indexed_journeys)
        for position, (_, candidate) in enumerate(indexed_journeys, start=1):
            journey = dict(candidate)
            journey["ranking_status"] = ranking_status
            journey["ranking_preference"] = preference_key
            journey["ranking_position"] = position
            journey["ranking_duration_min"] = duration(journey)
            journey["ranking_fare_inr"] = fare(journey) if fares_comparable else None
            journey["ranking_transfer_count"] = transfer_count(journey)
            if preference_key == "cheapest" and not fares_comparable:
                journey["ranking_reason"] = (
                    "Cheapest ranking unavailable: fares are missing, unverified, or not "
                    "comparable across candidates. Candidates are shown in stable ID order."
                )
            elif preference_key == "fastest":
                candidate_duration = duration(journey)
                journey["ranking_reason"] = (
                    f"Ranked #{position} of {total} by total journey duration "
                    f"({candidate_duration:g} minutes, including transfer time); "
                    "stable journey ID breaks ties."
                    if candidate_duration is not None
                    else f"Ranked #{position} of {total}; duration is missing, so this ranking is incomplete."
                )
            elif preference_key in {"earliest_departure", "earliest"}:
                dep_str = journey.get("departure_time") or journey.get("scheduled_departure_at") or "unspecified"
                journey["ranking_reason"] = (
                    f"Ranked #{position} of {total} by earliest departure time ({dep_str}); "
                    "duration and stable journey ID break ties."
                )
            elif preference_key == "cheapest":
                journey["ranking_reason"] = (
                    f"Ranked #{position} of {total} by verified comparable fare "
                    f"(INR {fare(journey):g})."
                )
            else:
                count = transfer_count(journey)
                journey["ranking_reason"] = (
                    f"Ranked #{position} of {total} by {count} transfer(s); "
                    "stable journey ID breaks ties."
                    if count is not None
                    else f"Ranked #{position} of {total}; transfer count is missing, so this ranking is incomplete."
                )

            self._annotate_recommendation(journey, position - 1, preference_key, [
                entry[1] for entry in indexed_journeys
            ])
            ranked.append(journey)
        return ranked

    def _find_next_departure(
        self,
        route_id: str,
        requested_time_str: str,
        widen_window: bool = False
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Find the next scheduled departure for route_id at or after requested_time_str.

        Returns:
            (departure_time_str, source_label) if a departure was found
            (None, None) if no matching scheduled departure is available
        """
        if self.df_schedules.empty:
            return None, None

        sch_matches = self.df_schedules[self.df_schedules["route_id"] == route_id]

        if sch_matches.empty:
            return None, None

        sch_times_sorted = sorted(sch_matches["departure_time"].dropna().tolist())

        if widen_window:
            # Return the very first available slot (used for "Find Next Available")
            if sch_times_sorted:
                return sch_times_sorted[0], "Project Dataset (Static Timetable)"
            return None, None

        later = [t for t in sch_times_sorted if t >= requested_time_str]
        if later:
            return later[0], "Project Dataset (Static Timetable)"

        # All scheduled slots are before the requested time
        return None, None

    def _annotate_recommendation(
        self,
        journey: Dict[str, Any],
        rank: int,
        preference: str,
        all_journeys: List[Dict[str, Any]]
    ) -> None:
        """Add explainable rationale and recommendation tags to journeys."""
        is_first = (rank == 0)
        has_dis = journey["has_disruption"]
        transfers = journey["transfers_count"]

        # Tag
        if journey.get("ranking_status") != "complete":
            journey["recommendation_tag"] = "Ranking unavailable or incomplete"
        elif is_first:
            if preference.lower() == "cheapest":
                journey["recommendation_tag"] = "🏆 Best Budget Option"
            elif preference.lower() == "fewest_transfers":
                journey["recommendation_tag"] = "🎯 Best Direct / Fewest Transfers"
            else:
                journey["recommendation_tag"] = "⚡ Fastest Journey"
        elif not has_dis and any(j["has_disruption"] for j in all_journeys[:rank]):
            journey["recommendation_tag"] = "🛡️ Recommended Disruption-Free Rescue"
        else:
            journey["recommendation_tag"] = f"Alternative #{rank}"

        # Explainable Rationale
        reasons = []
        if journey.get("ranking_status") != "complete":
            reasons.append(journey.get("ranking_reason", "Ranking unavailable."))
        elif is_first:
            reasons.append(f"Ranked #1 based on your preference for **{preference.upper()}** travel.")
        
        if has_dis:
            dis = journey["primary_disruption"]
            reasons.append(
                f"Illustrative demo scenario adds **+{dis['delay_minutes']} mins** for *{dis['incident_type']}* on {dis.get('location_segment', 'corridor')}."
            )
        else:
            reasons.append("No disruption scenario is attached to this local demo candidate.")

        if transfers == 0:
            reasons.append("Direct non-stop service eliminates any station transfer delay risk.")
        else:
            reasons.append(
                f"Requires 1 transfer at **{journey['transfer_station']}** with a feasible layover of **{journey['layover_minutes']} mins**."
            )

        journey["selection_rationale"] = " ".join(reasons)

        # Risk indicator
        journey["risk_indicator"] = self.assess_journey_risk(journey)

    def assess_journey_risk(self, journey: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate transparent heuristic risk indicators for a journey.
        Does not invent fake probabilities; evaluates explicit indicators.
        """
        delay = journey.get("delay_minutes", 0)
        transfers = journey.get("transfers_count", 0)
        layover = journey.get("layover_minutes", 0)
        min_buf = journey.get("min_transfer_buffer", 30)

        if delay >= 45:
            return {
                "level": "High Risk",
                "color": "#ef4444",
                "badge": "🔴 High Disruption Risk",
                "reason": f"Heavy simulated service delay (+{delay}m). High likelihood of schedule deviation or missed downstream plans."
            }
        elif transfers > 0 and (layover - delay) < min_buf:
            return {
                "level": "High Risk",
                "color": "#ef4444",
                "badge": "🔴 Tight Transfer Risk",
                "reason": f"Remaining layover after delay is under the required safety buffer ({min_buf}m). High risk of missed connection."
            }
        elif delay > 0 or (transfers > 0 and layover < 60):
            return {
                "level": "Moderate Risk",
                "color": "#f59e0b",
                "badge": "🟡 Moderate Risk",
                "reason": f"Moderate delay (+{delay}m) or standard connection buffer. Monitor platform announcements."
            }
        else:
            return {
                "level": "Low Risk",
                "color": "#10b981",
                "badge": "🟢 Low Risk",
                "reason": "Direct or well-buffered route with no active simulated disruption alerts."
            }

    def simulate_what_if_delay(self, journey: Dict[str, Any], extra_delay_min: int) -> Dict[str, Any]:
        """
        Simulate an extra what-if delay (15, 30, 45, 60 mins) and analyze impact on connections and arrival.
        """
        base_effective = journey["effective_duration_min"]
        new_total_delay = journey["delay_minutes"] + extra_delay_min
        new_effective = base_effective + extra_delay_min

        # Recalculate arrival
        dep_min = self._time_str_to_minutes(journey["departure_time"])
        new_arr_min = dep_min + new_effective
        new_arr_str = self._minutes_to_time_str(new_arr_min)

        connection_broken = False
        transfer_comment = "Direct route: no transfer connection to break."

        if journey["transfers_count"] > 0:
            layover = journey.get("layover_minutes", 0)
            min_buf = journey.get("min_transfer_buffer", 30)
            remaining_layover = layover - extra_delay_min
            
            if remaining_layover < min_buf:
                connection_broken = True
                transfer_comment = (
                    f"⚠️ **BROKEN CONNECTION DETECTED:** An additional {extra_delay_min}-minute delay shrinks "
                    f"your layover at {journey['transfer_station']} to only {max(0, remaining_layover)} minutes, "
                    f"which is less than the safe buffer ({min_buf} mins). You will likely miss the onward connection!"
                )
            else:
                transfer_comment = (
                    f"Connection remains feasible with {remaining_layover} minutes remaining buffer at {journey['transfer_station']}."
                )

        return {
            "extra_delay_min": extra_delay_min,
            "new_total_delay": new_total_delay,
            "new_effective_duration": new_effective,
            "new_arrival_time": new_arr_str,
            "connection_broken": connection_broken,
            "transfer_impact": transfer_comment
        }

    def get_tradeoff_comparison(self, journeys: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize available rankings without claiming unsupported fare or timing optimization."""
        if not journeys:
            return {}

        fastest_ranked = self.rank_journeys(journeys, "fastest")
        cheapest_ranked = self.rank_journeys(journeys, "cheapest")
        fewest_ranked = self.rank_journeys(journeys, "fewest_transfers")
        fastest = fastest_ranked[0] if fastest_ranked[0]["ranking_status"] == "complete" else None
        cheapest = cheapest_ranked[0] if cheapest_ranked[0]["ranking_status"] == "complete" else None
        fewest_transfers = (
            fewest_ranked[0]
            if fewest_ranked[0]["ranking_status"] == "complete"
            else None
        )
        time_saved = (
            max(0, cheapest["ranking_duration_min"] - fastest["ranking_duration_min"])
            if fastest is not None and cheapest is not None
            else None
        )
        fare_diff = (
            max(0, fastest["ranking_fare_inr"] - cheapest["ranking_fare_inr"])
            if fastest is not None and cheapest is not None
            else None
        )

        return {
            "fastest": fastest,
            "cheapest": cheapest,
            "fewest_transfers": fewest_transfers,
            "ranking_status": {
                "fastest": fastest_ranked[0]["ranking_status"],
                "cheapest": cheapest_ranked[0]["ranking_status"],
                "fewest_transfers": fewest_ranked[0]["ranking_status"]
            },
            "time_saved_mins": time_saved,
            "fare_diff_inr": fare_diff,
            "summary_text": (
                f"Choosing the **Fastest option** ({fastest['title']}) saves **{time_saved} minutes** "
                f"over the Cheapest option ({cheapest['title']}) for an additional **INR {fare_diff:g}**."
                if fastest is not None and cheapest is not None
                else "A complete trade-off comparison is unavailable because verified timing and comparable fare data are missing."
            )
        }

    def generate_disruption_timeline(self, journey: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate chronological timeline milestones for the journey.
        """
        timeline = []
        dep = journey["departure_time"]
        arr = journey["arrival_time"]

        timeline.append({
            "stage": "Journey Origin Departure",
            "time": dep,
            "status": "Illustrative demo schedule",
            "detail": f"Depart from {journey['origin_name']} on {journey['legs'][0]['route_name']} ({journey['legs'][0]['mode']})"
        })

        # Leg 1 Disruption
        leg1 = journey["legs"][0]
        if leg1.get("has_disruption"):
            d1 = leg1["disruption_info"]
            timeline.append({
                "stage": "Illustrative en-route disruption scenario",
                "time": f"+{d1['delay_minutes']}m delay",
                "status": f"Demo scenario — {d1['severity']} severity",
                "detail": f"{d1['incident_type']} on {d1['location_segment']}: {d1['description']}"
            })

        if journey["transfers_count"] > 0:
            leg2 = journey["legs"][1]
            timeline.append({
                "stage": "Transfer Interchange Arrival",
                "time": leg1["arrival_time"],
                "status": "Transit Station",
                "detail": f"Arrive at {journey['transfer_station']}. Scheduled layover window: {journey['layover_minutes']} mins."
            })

            if leg2.get("has_disruption"):
                d2 = leg2["disruption_info"]
                timeline.append({
                    "stage": "Illustrative connecting-service disruption scenario",
                    "time": f"+{d2['delay_minutes']}m delay",
                    "status": "Demo scenario — not a live alert",
                    "detail": f"{d2['incident_type']} on {leg2['route_name']}."
                })

            timeline.append({
                "stage": "Connecting Leg Departure",
                "time": leg2["departure_time"],
                "status": "Illustrative demo schedule",
                "detail": f"Board {leg2['route_name']} ({leg2['mode']}) toward destination."
            })

        timeline.append({
            "stage": "Destination Arrival",
            "time": arr,
            "status": "Final Destination",
            "detail": f"Illustrative arrival at {journey['destination_name']}. Demo duration: {journey['effective_duration_min']} mins."
        })

        return timeline

    def create_corridor_map_figure(self) -> go.Figure:
        """Return an explanatory placeholder when verified route geometry is unavailable."""
        fig = go.Figure()
        fig.add_annotation(
            text=(
                "Route map unavailable: local data contains stop coordinates but no "
                "verified route geometry. Use the itinerary timeline instead."
            ),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
        fig.update_layout(height=180, margin=dict(l=12, r=12, t=12, b=12))
        return fig

    def create_comparison_chart(self, journeys: List[Dict[str, Any]]) -> go.Figure:
        """Plot only candidates with verified timing and comparable verified fares."""
        fig = go.Figure()

        if not journeys:
            return fig
        for journey in journeys:
            elapsed = journey.get("elapsed_duration_min")
            if (
                journey.get("timing_verified") is not True
                or journey.get("fare_verified") is not True
                or not isinstance(elapsed, Real)
                or isinstance(elapsed, bool)
                or not math.isfinite(float(elapsed))
            ):
                return fig
        if self.rank_journeys(journeys, "cheapest")[0]["ranking_status"] != "complete":
            return fig

        top_candidates = journeys[:6]
        labels = [f"{j['title'][:22]}..<br>({j['journey_type'][:12]})" for j in top_candidates]
        base_dur = [j["elapsed_duration_min"] for j in top_candidates]
        fares = [j["total_fare_inr"] for j in top_candidates]

        fig.add_trace(go.Bar(
            name="Total Journey Time (min)",
            x=labels,
            y=base_dur,
            marker_color="#0284c7",
            text=[f"{b}m" for b in base_dur],
            textposition="auto"
        ))

        fig.add_trace(go.Scatter(
            name="Verified Fare (INR)",
            x=labels,
            y=fares,
            yaxis="y2",
            mode="lines+markers+text",
            marker=dict(size=10, color="#10b981"),
            line=dict(width=3, color="#10b981"),
            text=[f"₹{int(f)}" for f in fares],
            textposition="top center"
        ))

        fig.update_layout(
            title=dict(text="Total Journey Time vs Verified Fare", font=dict(size=14, color="#1e293b")),
            xaxis=dict(title="Transport Option"),
            yaxis=dict(title="Duration (Minutes)", showgrid=True),
            yaxis2=dict(title="Fare (INR)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=50, b=40),
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        return fig
