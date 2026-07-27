"""Region-aware storage routing — the data-residency "India key" applied to
where bytes physically land.

A patient's region ('US' | 'IN', see `patients.region` in schema.sql) selects
a Backblaze bucket and prefixes Supabase Storage object keys, so a region's
media is physically isolated and liftable as a unit.

Design choice: **only non-default ('IN') regions are re-routed.** 'US' keeps
its exact existing bucket and key scheme, so nothing about the current US path
changes (behaviour and stored keys are untouched). India objects land under an
`IN/` key prefix (Supabase) or a dedicated India bucket (Backblaze). This keeps
the change additive and backward-compatible: no US data migration, no collision
(US `foo.jpg` vs IN `IN/foo.jpg` are distinct namespaces).

Reads/deletes always use the STORED key, so prefixing at write time is
self-consistent — the read path needs no region knowledge.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

SUPPORTED_REGIONS = ("US", "IN")
DEFAULT_REGION = "US"


def region_for_country(country: str | None) -> str:
    """Map an ISO 3166-1 alpha-2 country to a supported data-residency region.

    Clients pick a full country; the residency contract only has two homes
    today — India routes to 'IN', everything else (and unknown/empty) to the
    default 'US'. This is the single place that decision lives, so adding a
    third region later (e.g. an EU home) is a one-line change here and nowhere
    else. The country is recorded verbatim on the row; only the *routing* region
    collapses to what we physically support."""
    if country and country.strip().upper() == "IN":
        return "IN"
    return DEFAULT_REGION


def normalize(region: str | None) -> str:
    """Coerce to a supported region, defaulting to US. Never raises: an
    unexpected value must not break an upload — it degrades to US and logs."""
    if region in SUPPORTED_REGIONS:
        return region  # type: ignore[return-value]
    if region is not None:
        log.warning("region.unknown value=%r; defaulting to %s", region, DEFAULT_REGION)
    return DEFAULT_REGION


def storage_key(region: str | None, key: str) -> str:
    """Region-prefix a Supabase Storage object key. US is unchanged; IN gets
    an `IN/` prefix (e.g. `IN/patient-photos/{patient_id}.jpg`)."""
    r = normalize(region)
    if r == DEFAULT_REGION:
        return key
    return f"{r}/{key.lstrip('/')}"


def b2_bucket_for(region: str | None, settings: Any) -> str:
    """Pick the Backblaze bucket for a region. US uses the default bucket; IN
    routes to its own bucket for isolation + liftability. If the India bucket
    isn't configured yet (India infra not stood up), fall back to the default
    bucket with a loud warning rather than failing data collection."""
    r = normalize(region)
    if r == DEFAULT_REGION:
        return settings.b2_bucket
    india = getattr(settings, "b2_bucket_in", "") or ""
    if india:
        return india
    log.warning(
        "region.b2_bucket_in unset; IN clip routing to default bucket %r "
        "(commingled until an India bucket is configured)",
        settings.b2_bucket,
    )
    return settings.b2_bucket
