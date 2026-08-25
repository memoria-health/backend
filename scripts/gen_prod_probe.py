#!/usr/bin/env python3
"""Generate the read-only prod pre-flight probe from scripts/deltas/*.sql.

Hand-maintaining it did not work: it covered 3 of 62 deltas, so "which
deltas does prod still need" had no read-only answer and the question was
settled by guessing. Regenerate instead of editing:

    python3 scripts/gen_prod_probe.py > scripts/probe_prod_schema.sql

A delta that adds a column, table or index can be detected by looking for
that object. One that only changes data, policies or types cannot, so it
is listed at the bottom for a human rather than silently omitted.
"""
import pathlib
import re

DELTAS = pathlib.Path(__file__).parent / "deltas"


def objects(sql: str):
    out = []
    for m in re.finditer(
            r"alter\s+table\s+(?:if\s+exists\s+)?(?:public\.)?(\w+)\s+add\s+"
            r"column\s+(?:if\s+not\s+exists\s+)?(\w+)", sql, re.I):
        out.append(("column", m.group(1), m.group(2)))
    for m in re.finditer(
            r"create\s+table\s+(?:if\s+not\s+exists\s+)?(?:public\.)?(\w+)",
            sql, re.I):
        out.append(("table", m.group(1), None))
    if not out:
        for m in re.finditer(
                r"create\s+(?:unique\s+)?index\s+(?:concurrently\s+)?"
                r"(?:if\s+not\s+exists\s+)?(\w+)", sql, re.I):
            out.append(("index", m.group(1), None))
            break
    return out


def condition(kind: str, a: str, b: "str | None") -> str:
    if kind == "column":
        return ("exists(select 1 from information_schema.columns "
                f"where table_schema='public' and table_name='{a}' "
                f"and column_name='{b}')")
    if kind == "table":
        return ("exists(select 1 from information_schema.tables "
                f"where table_schema='public' and table_name='{a}')")
    return ("exists(select 1 from pg_indexes "
            f"where schemaname='public' and indexname='{a}')")


def main() -> None:
    rows, manual = [], []
    for f in sorted(DELTAS.glob("*.sql")):
        found = objects(f.read_text())
        if not found:
            manual.append(f.name)
            continue
        seen = set()
        for kind, a, b in found:
            if (kind, a, b) in seen:
                continue
            seen.add((kind, a, b))
            label = f"{a}.{b}" if b else a
            tag = repr(f"{kind} {label}  ({f.name})")
            rows.append(f"select {tag:<58} as object, "
                        f"{condition(kind, a, b)} as present")

    print("-- READ-ONLY pre-flight probe. Run against PROD before any delta.")
    print("-- present = true  -> already applied (safe to skip)")
    print("-- present = false -> run the matching delta, BUT SEE BELOW FIRST")
    print("--")
    print("-- SUPERSEDED DELTAS READ AS ABSENT. Some columns were added by one")
    print("-- delta and dropped by a later one (episode_gap, convulsion,")
    print("-- where_scope, is_builtin, pi_source ...). Those show present=false")
    print("-- because the column really is gone — re-applying them would undo")
    print("-- the drop. Before running anything, check whether a")
    print("-- scripts/deltas/apply_drop_*.sql exists for the same object.")
    print("--")
    print("-- GENERATED. Do not hand-edit:")
    print("--   python3 scripts/gen_prod_probe.py > scripts/probe_prod_schema.sql")
    print("--")
    print("-- Paste into the PROD Supabase SQL editor, or:")
    print('--   psql "<prod connection string>" -f scripts/probe_prod_schema.sql')
    print()
    print("\nunion all\n".join(rows))
    print("order by 1;")
    if manual:
        print()
        print("-- NOT PROBED — these change data, policies or types rather than")
        print("-- adding an object, so presence cannot be inferred. Check by hand:")
        for n in manual:
            print(f"--   {n}")


if __name__ == "__main__":
    main()
