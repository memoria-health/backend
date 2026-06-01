"""One-shot: send a test APNs push to every registered caregiver device.

Exercises the LIVE APNs config (auth key, key id, team id, topic, host)
end-to-end against your real device token(s) — independent of the
detector pipeline and the 30s event dedup. Use it to confirm push
delivery instantly, without performing a fall.

It reuses the backend's own `mint_provider_jwt` + `build_event_payload`
and the same request shape as `services.push._send_to_device`, so a
success here means a real fall WILL deliver too.

Run against local: `uv run python -m scripts.test_push`
Custom message:    `uv run python -m scripts.test_push --body "hello there"`
"""

from __future__ import annotations

import argparse
import asyncio
import json

import httpx
from sqlalchemy import select

from src.db.caregiver_device import CaregiverDevice
from src.db.session import get_sessionmaker
from src.services.push import (
    _APNS_PROD_HOST,
    _APNS_SANDBOX_HOST,
    build_event_payload,
    mint_provider_jwt,
)
from src.settings import get_settings

# APNs `reason` codes worth translating into plain English. These are the
# ones that actually show up when local push config is subtly wrong.
_HINTS = {
    "BadDeviceToken": (
        "token doesn't match the host's environment (sandbox token sent to "
        "the prod host or vice-versa), or it's from a different app."
    ),
    "TopicDisallowed": "apns-topic isn't permitted for this auth key.",
    "BadTopic": "apns-topic is malformed / not this app's bundle id.",
    "DeviceTokenNotForTopic": (
        "this token wasn't issued for apns-topic — bundle-id mismatch "
        "between the backend and the installed app."
    ),
    "InvalidProviderToken": (
        "the .p8 auth key's TEAM doesn't own this app. You need an APNs key "
        "generated under the same Apple team that signs the app."
    ),
    "ExpiredProviderToken": "the provider JWT expired — clock skew or stale mint.",
    "Unregistered": "Apple says this token is dead (app uninstalled).",
}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--title", default="Memoria — test push")
    ap.add_argument(
        "--body",
        default="If you can read this, APNs is wired up correctly.",
    )
    args = ap.parse_args()

    settings = get_settings()

    print("APNs config the backend will use:")
    print(f"  apns-topic (must == app bundle id) : {settings.apns_bundle_id}")
    print(f"  team (iss)                         : {settings.apns_team_id}")
    print(f"  key id (kid)                       : {settings.apns_key_id}")
    print(f"  settings env                       : {settings.apns_environment}\n")

    jwt = mint_provider_jwt(settings)
    if not jwt:
        print(
            "✗ APNs not configured — team_id / key_id / auth_key is empty. "
            "Nothing to send."
        )
        return

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        rows = (
            await session.execute(
                select(
                    CaregiverDevice.apns_token,
                    CaregiverDevice.environment,
                    CaregiverDevice.bundle_id,
                )
            )
        ).all()

    if not rows:
        print(
            "✗ No registered devices in `caregiver_devices`. Launch Memoria "
            "Care, accept the notification prompt, then re-run."
        )
        return

    payload = build_event_payload(
        title=args.title,
        body=args.body,
        # Dummy ids — a tap just deep-links to a (missing) event, harmless.
        event_id="00000000-0000-0000-0000-000000000000",
        patient_id="00000000-0000-0000-0000-000000000000",
    )
    body_bytes = json.dumps(payload).encode()

    print(f"Sending to {len(rows)} device(s)…\n")
    async with httpx.AsyncClient(http2=True, timeout=15.0) as client:
        for token, env, bundle in rows:
            host = _APNS_PROD_HOST if env == "production" else _APNS_SANDBOX_HOST
            headers = {
                "authorization": f"bearer {jwt}",
                "apns-topic": settings.apns_bundle_id,
                "apns-push-type": "alert",
                "apns-priority": "10",
            }
            url = f"{host}/3/device/{token}"
            short = token[:10]
            try:
                resp = await client.post(url, content=body_bytes, headers=headers)
            except Exception as exc:  # noqa: BLE001
                print(f"  ✗ {short}… [{env}] transport error: {exc}")
                continue

            if resp.status_code == 200:
                apns_id = resp.headers.get("apns-id", "?")
                print(f"  ✓ {short}… [{env}] delivered  (apns-id={apns_id})")
                continue

            try:
                reason = resp.json().get("reason", resp.text)
            except Exception:  # noqa: BLE001
                reason = resp.text
            print(f"  ✗ {short}… [{env}] HTTP {resp.status_code}  reason={reason}")
            hint = _HINTS.get(reason)
            if resp.status_code == 403 and reason == "InvalidProviderToken":
                hint = _HINTS["InvalidProviderToken"]
            if hint:
                print(f"      → {hint}")
            if bundle and bundle != settings.apns_bundle_id:
                print(
                    f"      → note: this device registered under bundle "
                    f"'{bundle}', but apns-topic is '{settings.apns_bundle_id}'."
                )


if __name__ == "__main__":
    asyncio.run(main())
