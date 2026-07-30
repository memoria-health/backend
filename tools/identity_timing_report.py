"""Identity pipeline timing report for a time window.

Answers "where did the time actually go" for any stretch of the soak —
built 2026-07-29 after an afternoon of guessing wrong three times
because the single `pod=` number conflated GPU compute with tunnel
transfer. Every batch now logs both, and this reads them back.

    # a window today, local time
    uv run python tools/identity_timing_report.py --from 15:30 --to 16:10

    # explicit date
    uv run python tools/identity_timing_report.py \
        --from "2026-07-29 15:30" --to "2026-07-29 16:10"

    # last N minutes
    uv run python tools/identity_timing_report.py --last 30

Reports per camera and overall: batch size, and the full per-tick
split (encode / queue / pod-compute / wire / publish). Also reports
CONCURRENCY — the share of minutes in which BOTH cameras were
tracking — because that is the variable that decides whether the
pipeline sits at a comfortable duty cycle or runs away.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

LOG = Path("/tmp/identity_service.log")

# Instrumented form (2026-07-29 onward) carries the compute/wire split;
# the older form only has a single pod number. Both are parsed so a
# window spanning the change still reports.
_NEW = re.compile(
    r"^(?P<ts>[\d-]+ [\d:]+),\d+ .*identity\.batch cam=(?P<cam>\w+) "
    r"frames=(?P<frames>\d+) span=(?P<span>[\d.]+)s stride=(?P<stride>[\d.]+)ms \| "
    r"encode=(?P<encode>[\d.]+) queue=(?P<queue>[\d.]+) pod=(?P<pod>[\d.]+) "
    r"\(compute=(?P<compute>[\d.]+) sam=(?P<sam>[\d.]+) decode=(?P<decode>[\d.]+) "
    r"wire=(?P<wire>-?[\d.]+)\) publish=(?P<publish>[\d.]+)"
)
_OLD = re.compile(
    r"^(?P<ts>[\d-]+ [\d:]+),\d+ .*identity\.batch cam=(?P<cam>\w+) "
    r"frames=(?P<frames>\d+) span=(?P<span>[\d.]+)s stride=(?P<stride>[\d.]+)ms \| "
    r"encode=(?P<encode>[\d.]+) queue=(?P<queue>[\d.]+) pod=(?P<pod>[\d.]+) "
    r"publish=(?P<publish>[\d.]+)"
)


def _parse_when(s: str, *, end: bool = False) -> datetime:
    s = s.strip()
    today = datetime.now().strftime("%Y-%m-%d")
    for fmt, prefix in (
        ("%Y-%m-%d %H:%M:%S", ""), ("%Y-%m-%d %H:%M", ""),
        ("%Y-%m-%d %H:%M:%S", None), ("%Y-%m-%d %H:%M", None),
    ):
        cand = s if prefix == "" else f"{today} {s}"
        try:
            return datetime.strptime(cand, fmt)
        except ValueError:
            continue
    raise SystemExit(f"could not parse time {s!r} (try 'HH:MM' or 'YYYY-MM-DD HH:MM')")


def _rows(t0: datetime, t1: datetime) -> list[dict]:
    out: list[dict] = []
    if not LOG.exists():
        raise SystemExit(f"{LOG} not found")
    with LOG.open(errors="replace") as fh:
        for line in fh:
            if "identity.batch" not in line:
                continue
            m = _NEW.match(line) or _OLD.match(line)
            if not m:
                continue
            d = m.groupdict()
            ts = datetime.strptime(d["ts"], "%Y-%m-%d %H:%M:%S")
            if not (t0 <= ts <= t1):
                continue
            row = {"ts": ts, "cam": d["cam"], "frames": int(d["frames"])}
            for k in ("span", "stride", "encode", "queue", "pod", "publish",
                      "compute", "sam", "decode", "wire"):
                row[k] = float(d[k]) if d.get(k) is not None else None
            out.append(row)
    return out


def _fmt(label: str, vals: list[float | None], total: int) -> str:
    v = [x for x in vals if x is not None]
    if not v:
        return f"  {label:<16} — (not instrumented in this window)"
    p95 = sorted(v)[min(len(v) - 1, int(len(v) * 0.95))]
    # n is printed on EVERY line because the compute/wire split only
    # exists on batches logged after the 2026-07-29 instrumentation.
    # Without it, a window straddling that change shows a pod mean over
    # all batches next to a compute mean over a handful — which reads
    # as "3 seconds unaccounted for" and is pure artifact.
    tag = "" if len(v) == total else f"  [n={len(v)}/{total}]"
    return (f"  {label:<16} mean {statistics.mean(v):7.0f}ms   "
            f"median {statistics.median(v):7.0f}ms   p95 {p95:7.0f}ms{tag}")


def _report(rows: list[dict], title: str) -> None:
    print(f"\n{title}  (n={len(rows)} batches)")
    fr = [r["frames"] for r in rows]
    print(f"  {'frames/batch':<14} mean {statistics.mean(fr):7.1f}    "
          f"median {statistics.median(fr):7.1f}    max {max(fr):7d}")
    for key, label in (("encode", "encode"), ("queue", "queue (lock)"),
                       ("pod", "pod round-trip"), ("compute", "  pod compute"),
                       ("sam", "    of which sam"), ("wire", "  wire (tunnel)"),
                       ("publish", "publish")):
        print(_fmt(label, [r.get(key) for r in rows], len(rows)))
    # The split is computed ONLY over batches that carry both numbers,
    # and says so, so it can never be read against a pod mean drawn
    # from a different (larger) sample.
    inst = [r for r in rows
            if r.get("compute") is not None and r.get("wire") is not None]
    if inst:
        tc = sum(r["compute"] for r in inst)
        tw = sum(r["wire"] for r in inst)
        pod = statistics.mean(r["pod"] for r in inst)
        print(f"  {'SPLIT':<16} compute {100 * tc / (tc + tw):.0f}%   "
              f"wire {100 * tw / (tc + tw):.0f}%   "
              f"(over {len(inst)}/{len(rows)} instrumented batches, "
              f"whose pod mean is {pod:.0f}ms)")
    else:
        print(f"  {'SPLIT':<16} unavailable — no instrumented batches "
              f"in this window (instrumentation live 2026-07-29 15:00)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="t_from", help="start (HH:MM or 'YYYY-MM-DD HH:MM')")
    ap.add_argument("--to", dest="t_to", help="end")
    ap.add_argument("--last", type=int, help="last N minutes")
    a = ap.parse_args()

    if a.last:
        t1 = datetime.now()
        t0 = t1 - timedelta(minutes=a.last)
    elif a.t_from and a.t_to:
        t0, t1 = _parse_when(a.t_from), _parse_when(a.t_to, end=True)
    else:
        raise SystemExit("need --last N, or --from and --to")

    if t1 <= t0:
        # Easy mistake with bare HH:MM; without this the tool reports
        # "no batches found", which reads like a data problem.
        raise SystemExit(
            f"end ({t1:%Y-%m-%d %H:%M}) is not after start ({t0:%Y-%m-%d %H:%M})"
        )
    rows = _rows(t0, t1)
    if not rows:
        raise SystemExit(f"no identity.batch lines between {t0} and {t1}")

    print(f"window {t0:%Y-%m-%d %H:%M} -> {t1:%H:%M}   ({len(rows)} batches)")

    # Concurrency: the variable that decides whether the pipeline is
    # comfortable or runaway. Measured per minute, since a batch takes
    # seconds and "both occupied" is a minute-scale fact.
    by_min: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_min[r["ts"].strftime("%H:%M")].add(r["cam"])
    both = sum(1 for v in by_min.values() if len(v) > 1)
    print(f"concurrency: {both}/{len(by_min)} minutes with BOTH cameras "
          f"tracking ({100 * both / max(1, len(by_min)):.0f}%)")

    cams = sorted({r["cam"] for r in rows})
    for cam in cams:
        _report([r for r in rows if r["cam"] == cam], f"CAMERA {cam}")
    if len(cams) > 1:
        _report(rows, "ALL CAMERAS")
        conc = [r for r in rows
                if len(by_min[r["ts"].strftime("%H:%M")]) > 1]
        solo = [r for r in rows
                if len(by_min[r["ts"].strftime("%H:%M")]) == 1]
        if conc and solo:
            _report(solo, "— minutes with ONE camera tracking")
            _report(conc, "— minutes with BOTH cameras tracking")


if __name__ == "__main__":
    main()
