"""Every table in `public` must have Row-Level Security enabled.

WHY THIS TEST EXISTS
--------------------
On 2026-08-03 a live audit found `sleep_nights`, `activity_days` and
`review_devices` shipped with RLS off. Supabase grants `anon` and
`authenticated` full SELECT/INSERT/UPDATE/DELETE/TRUNCATE on everything in
`public`, and PostgREST exposes it — so with RLS off those grants are the only
control, and `anon` is the PUBLISHABLE key that ships inside the iOS binary and
the web bundle.

Anyone could have read, forged or TRUNCATEd patient sleep and activity records.
Production was spared only because the two deltas that create those tables had
not been applied yet.

(`review_devices` no longer exists — 2026-08-05, operator paging moved to
PagerDuty, so the reviewer push tokens it held are not stored at all. The
history stays here because the lesson is about deltas, not that table.)

Nothing caught it. Twenty-four static audit reports missed it because they
never query a database, and the schema review missed it because the tables are
created in delta scripts rather than schema.sql.

This test closes that gap at AUTHORING time: a new delta that creates a table
without enabling RLS fails the unit suite. It is deliberately static (parses
SQL text, no database connection) so it runs in the suite that actually
executes here — the integration suite currently cannot reach its fixtures.

Defense in depth, not a replacement: `scripts/deltas/apply_auto_enable_rls_delta.sql`
installs an event trigger that enables RLS automatically at CREATE TABLE time.
This test catches the mistake in review; the trigger catches it in the database.
"""

from __future__ import annotations

import re
from pathlib import Path

_SQL_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = _SQL_ROOT / "supabase" / "schema.sql"
_SCRIPTS = _SQL_ROOT / "scripts"

# Tables that intentionally live without RLS. EMPTY, and it should stay that
# way — an entry here is a decision to expose a table to the publishable key,
# so it needs a name, a date and a reason.
_ALLOWLIST: dict[str, str] = {}

_CREATE = re.compile(
    r"create\s+table\s+(?:if\s+not\s+exists\s+)?"
    r"(?:public\.)?\"?(\w+)\"?",
    re.IGNORECASE,
)
_ENABLE = re.compile(
    r"alter\s+table\s+(?:public\.)?\"?(\w+)\"?\s+enable\s+row\s+level\s+security",
    re.IGNORECASE,
)


_LINE_COMMENT = re.compile(r"--[^\n]*")
_STRING_LIT = re.compile(r"'(?:[^']|'')*'", re.DOTALL)


def _strip_noise(sql: str) -> str:
    """Remove comments and single-quoted literals before pattern matching.

    Without this, SQL that TALKS about DDL is read as DDL. The event trigger in
    apply_auto_enable_rls_delta.sql contains the literal `'CREATE TABLE AS'` in
    its `when tag in (...)` clause and a `format('alter table %s ...')` call —
    the naive regex matched inside both and invented a table named `as`.
    """
    return _STRING_LIT.sub("''", _LINE_COMMENT.sub("", sql))


def _all_sql() -> str:
    parts = []
    if _SCHEMA.exists():
        parts.append(_strip_noise(_SCHEMA.read_text(errors="replace")))
    for f in sorted(_SCRIPTS.glob("*.sql")):
        parts.append(_strip_noise(f.read_text(errors="replace")))
    return "\n".join(parts)


def _sql_by_file() -> dict[str, str]:
    out = {}
    if _SCHEMA.exists():
        out[str(_SCHEMA.relative_to(_SQL_ROOT))] = _strip_noise(
            _SCHEMA.read_text(errors="replace")
        )
    for f in sorted(_SCRIPTS.glob("*.sql")):
        out[str(f.relative_to(_SQL_ROOT))] = _strip_noise(
            f.read_text(errors="replace")
        )
    return out


def test_every_created_public_table_enables_rls() -> None:
    """A `create table` in public without a matching `enable row level
    security` anywhere in the SQL corpus is a shipped vulnerability."""
    corpus = _all_sql()
    created = {m.group(1).lower() for m in _CREATE.finditer(corpus)}
    protected = {m.group(1).lower() for m in _ENABLE.finditer(corpus)}

    missing = sorted(created - protected - set(_ALLOWLIST))
    assert not missing, (
        "These public tables are created without RLS. Supabase grants `anon` "
        "(the PUBLISHABLE key, shipped in the app binary) full DML on every "
        "table in public, so RLS is the only control:\n  "
        + "\n  ".join(missing)
        + "\n\nAdd to the delta that creates them:\n"
        + "\n".join(
            f"  alter table public.{t} enable row level security;" for t in missing
        )
        + "\n\nEnabling RLS with no policy denies all non-owner access, which "
        "is the correct fail-closed default. Add policies only where a client "
        "genuinely needs to read the table directly."
    )


def test_each_delta_that_creates_a_table_also_enables_rls() -> None:
    """Per-file check. The corpus test above passes if RLS is enabled in ANY
    file, which lets a delta ship incomplete and be rescued later by a
    follow-up. Deltas are applied one at a time against production, so the
    window between them is real exposure — each file must stand alone.
    """
    offenders: list[str] = []
    for name, sql in _sql_by_file().items():
        created = {m.group(1).lower() for m in _CREATE.finditer(sql)}
        protected = {m.group(1).lower() for m in _ENABLE.finditer(sql)}
        gap = created - protected - set(_ALLOWLIST)
        # The dedicated backfill delta is the one legitimate exception: it
        # exists precisely to enable RLS on tables created elsewhere.
        if gap and "apply_rls_" not in name:
            offenders.append(f"{name}: {', '.join(sorted(gap))}")
    assert not offenders, (
        "These files create a public table without enabling RLS in the SAME "
        "file. Deltas are applied individually against production, so the gap "
        "between this delta and any later fix is a live exposure window:\n  "
        + "\n  ".join(offenders)
    )


def test_allowlist_entries_are_justified() -> None:
    """If someone adds an allowlist entry, make them explain it. An
    unexplained exemption is how the original bug would come back."""
    for table, reason in _ALLOWLIST.items():
        assert len(reason) > 30, (
            f"_ALLOWLIST['{table}'] needs a real justification — who decided, "
            f"when, and why it is safe for the publishable key to reach it."
        )
