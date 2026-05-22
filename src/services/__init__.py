"""Service layer — business logic; owns transactions.

Each service module composes one or more repos and exposes the
operations the API routers call.
"""

from src.services import auth, onboarding

__all__ = ["auth", "onboarding"]
