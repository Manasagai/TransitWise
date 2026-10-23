"""
RouteRescue AI — Authentication Manager
Module: auth_manager.py

Provides prototype user registration, authentication, and secure password hashing
using SHA-256 with per-user salt stored in 'data/users.json'.
Clearly marked as a prototype credential store with instant Demo Access support.
"""

import datetime
import hashlib
import json
import os
import re
import secrets
from typing import Any, Dict, List, Optional, Tuple


USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "users.json")


def _ensure_users_file() -> None:
    """Ensure data/users.json exists with pre-seeded demo user."""
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    if not os.path.exists(USERS_FILE):
        # Generate demo user credentials
        salt = secrets.token_hex(16)
        demo_pwd_hash = _hash_password("demo123", salt)
        initial_users = {
            "demo@routerescue.ai": {
                "name": "Demo Commuter",
                "email": "demo@routerescue.ai",
                "salt": salt,
                "password_hash": demo_pwd_hash,
                "created_at": "2026-10-13T12:00:00Z",
                "is_demo": True
            }
        }
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(initial_users, f, indent=2)


def _load_users() -> Dict[str, Any]:
    """Load users dictionary from JSON file."""
    _ensure_users_file()
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_users(users: Dict[str, Any]) -> None:
    """Save users dictionary to JSON file."""
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str, salt: str) -> str:
    """Hash password securely using SHA-256 with unique salt."""
    salted = f"{salt}:{password}".encode("utf-8")
    return hashlib.sha256(salted).hexdigest()


def register_user(name: str, email: str, password: str, confirm_password: str) -> Tuple[bool, str]:
    """
    Register a new user with input validation and salted password hash.
    
    Returns:
        (success: bool, message: str)
    """
    name = name.strip()
    email = email.strip().lower()

    if not name:
        return False, "Please enter your full name."

    # Validate email
    email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    if not re.match(email_regex, email):
        return False, "Please enter a valid email address (e.g., traveler@example.com)."

    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    if password != confirm_password:
        return False, "Passwords do not match. Please re-enter."

    users = _load_users()
    if email in users:
        return False, "An account with this email already exists. Please log in."

    salt = secrets.token_hex(16)
    pwd_hash = _hash_password(password, salt)

    users[email] = {
        "name": name,
        "email": email,
        "salt": salt,
        "password_hash": pwd_hash,
        "created_at": "2026-10-13T12:00:00Z",
        "is_demo": False
    }

    _save_users(users)
    return True, f"Account created successfully for {name}! You can now log in."


def authenticate_user(email: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Authenticate an existing user.
    
    Returns:
        (success: bool, user_info: dict or None, message: str)
    """
    email = email.strip().lower()
    users = _load_users()

    if email not in users:
        return False, None, "No account found with this email. Please sign up or use Explore Demo."

    user_data = users[email]
    salt = user_data["salt"]
    expected_hash = user_data["password_hash"]

    if _hash_password(password, salt) == expected_hash:
        return True, {
            "name": user_data["name"],
            "email": user_data["email"],
            "is_demo": user_data.get("is_demo", False)
        }, "Login successful!"

    return False, None, "Incorrect password. Please try again."


def get_demo_user() -> Dict[str, Any]:
    """Return pre-configured demo user profile for instant 1-click exploration."""
    return {
        "name": "Demo Commuter",
        "email": "demo@routerescue.ai",
        "is_demo": True
    }


def get_user_favourites(email: str) -> List[Dict[str, Any]]:
    """Return list of saved favourite bus routes for the specified user."""
    email = email.strip().lower()
    users = _load_users()
    user = users.get(email)
    if not user:
        return []
    return list(user.get("favourite_routes", []))


def add_user_favourite(
    email: str,
    origin_name: str,
    destination_name: str,
    origin_id: Optional[str] = None,
    destination_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> Tuple[bool, str]:
    """Save a bus route to the user's private favourites."""
    email = email.strip().lower()
    users = _load_users()
    if email not in users:
        return False, "User not found."

    favourites = users[email].setdefault("favourite_routes", [])
    for fav in favourites:
        if fav.get("origin_name") == origin_name and fav.get("destination_name") == destination_name:
            return False, f"Route from {origin_name} to {destination_name} is already in your favourites."

    new_fav = {
        "id": secrets.token_hex(6),
        "origin_name": origin_name,
        "destination_name": destination_name,
        "origin_id": origin_id or "",
        "destination_id": destination_id or "",
        "notes": (notes or "").strip(),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    favourites.append(new_fav)
    _save_users(users)
    return True, f"Saved route from {origin_name} to {destination_name} to your favourites."


def remove_user_favourite(email: str, favourite_id: str) -> Tuple[bool, str]:
    """Remove a favourite route by its ID for the specified user."""
    email = email.strip().lower()
    users = _load_users()
    if email not in users:
        return False, "User not found."

    favourites = users[email].get("favourite_routes", [])
    updated = [f for f in favourites if f.get("id") != favourite_id]
    if len(updated) == len(favourites):
        return False, "Favourite route not found."

    users[email]["favourite_routes"] = updated
    _save_users(users)
    return True, "Removed route from favourites."


def get_user_search_history(email: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Return recent search history for the specified user."""
    email = email.strip().lower()
    users = _load_users()
    user = users.get(email)
    if not user:
        return []
    history = user.get("search_history", [])
    return list(reversed(history[-limit:]))


def add_user_search_history(
    email: str,
    origin_name: str,
    destination_name: str,
    travel_date: str,
    departure_time: str,
    results_count: int = 0,
) -> None:
    """Record a search in the user's private search history."""
    email = email.strip().lower()
    users = _load_users()
    if email not in users:
        return
    history = users[email].setdefault("search_history", [])
    entry = {
        "id": secrets.token_hex(6),
        "origin_name": origin_name,
        "destination_name": destination_name,
        "travel_date": travel_date,
        "departure_time": departure_time,
        "results_count": results_count,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    history.append(entry)
    if len(history) > 50:
        users[email]["search_history"] = history[-50:]
    _save_users(users)


def clear_user_search_history(email: str) -> Tuple[bool, str]:
    """Clear the search history for the specified user."""
    email = email.strip().lower()
    users = _load_users()
    if email not in users:
        return False, "User not found."
    users[email]["search_history"] = []
    _save_users(users)
    return True, "Search history cleared."

