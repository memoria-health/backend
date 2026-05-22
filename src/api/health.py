"""Liveness + readiness probes.

`/health` is the cheap liveness check: did the process boot.
`/ready`  is the readiness check: can it reach Postgres.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_session

router = APIRouter(tags=["health"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: str


class ReadyResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: str
    checks: dict[str, str]


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness — is the process running."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def ready(session: SessionDep) -> ReadyResponse:
    """Readiness — is the process able to serve requests (DB reachable)."""
    try:
        result = await session.execute(text("SELECT 1"))
        result.one()
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "fail", "checks": {"db": "fail"}, "error": str(exc)},
        ) from exc
    return ReadyResponse(status="ok", checks={"db": "ok"})
