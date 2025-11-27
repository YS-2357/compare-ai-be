"""FastAPI 공용 dependencies."""

from __future__ import annotations

from app.auth import AuthenticatedUser, get_current_user
from app.config import get_settings
from app.rate_limit import enforce_daily_limit

__all__ = ["get_current_user", "enforce_daily_limit", "get_settings", "AuthenticatedUser"]

