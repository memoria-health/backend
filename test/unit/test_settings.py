"""Tiny smoke test that doesn't need Supabase running — just verifies
that Settings loads from `.env`.
"""

from src.settings import get_settings


def test_settings_loads_with_required_fields() -> None:
    s = get_settings()
    assert s.db_url.startswith("postgresql+asyncpg://")
    assert s.supabase_url.startswith("http")
    assert s.env in {"local", "test", "staging", "prod"}
