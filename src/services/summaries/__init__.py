"""Nightly and daily records for the `record_only` detectors.

Sleep and activity never fire an event, so nothing in the alerting path
carries them to a caregiver. Their route is this: fold a window of
`analyses` ticks into one row (`sleep_nights`, `activity_days`) that the
Home surface and the journal read.

`nightly.py` owns both the fold and the schedule. It used to live only
in `tools/run_daily_summaries.py`, which meant a night became a row only
when somebody remembered to type the command — and between 2026-08-18
and 2026-08-25 nobody did, so seven nights of collected data were never
summarised. The tool is still there and still works; it is now a thin
front end onto the same code.
"""
