# Nemara end-to-end test plan

First full-stack validation across all three Nemara apps post-Phase-4.

**Topology**

- **Wall phone** — iPhone running `nemara-camera-app` (mounted, plugged in)
- **Caregiver phone** — iPhone running `nemara-caregiver-app` (in your hand)
- **Laptop** — `nemara-backend` (`make dev` on `:8002`), local Supabase
  (`supabase start`), and an `ngrok` / `cloudflared` tunnel pointing at
  the backend so LiveKit Cloud webhooks can reach you.

Both phones reach the laptop over Wi-Fi at the Mac's LAN address.

---

## A. One-time setup

### A1. Apple Developer artifacts

Both iOS apps need a real Developer Team. In Xcode for each
`.xcodeproj`: **Target → Signing & Capabilities → Team**. Pick the
same Team for both so the bundle IDs aren't ambiguous.

- The caregiver app's bundle ID is `co.elephantlabs.nemara`. In the
  [Apple Developer portal](https://developer.apple.com/account/resources/identifiers/list)
  → Identifiers, confirm that ID has **Push Notifications** capability
  ticked. If you don't see it, create the identifier and tick it.
- Apple Developer → **Keys** → **+** → Apple Push Notification service.
  Download the `.p8` file (you only get one chance to download it).
  Note the **Key ID** (10 chars) and your **Team ID**
  (Membership → 10 chars).

### A2. Supabase Storage buckets

Local Supabase Studio (`supabase status` shows the URL — Studio
lives on a different port than the API; both are listed in the
output):

- Storage → New bucket → name `patient-photos`, **private**.
- Storage → New bucket → name `incident-clips`, **private**.

Both should be private so the backend's signed-URL flow is the only
read path.

### A3. LiveKit Cloud project

At [livekit.io](https://livekit.io) → create a project → Settings:

- WebSocket URL: `wss://<your-project>.livekit.cloud`
- API key + API secret

Webhooks: same dashboard → Webhooks → add one pointing at
`https://<your-ngrok-or-cloudflared-URL>/v1/livekit/webhooks`.
LiveKit Cloud needs to reach your laptop to drive
`participant_joined` / `participant_left` / `egress_ended` events.

### A4. Backend `.env`

In `nemara-backend/.env`:

```
# Database
NEMARA_DB_URL=postgresql+asyncpg://postgres:postgres@127.0.0.1:54342/postgres

# Supabase (from `supabase status -o env`)
NEMARA_SUPABASE_URL=http://127.0.0.1:54341
NEMARA_SUPABASE_PUBLISHABLE_KEY=eyJhbGciOi...
NEMARA_SUPABASE_JWT_SECRET=super-secret-jwt-token-with-at-least-32-characters-long
NEMARA_SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOi...

# LiveKit (real Cloud project)
NEMARA_LIVEKIT_URL=wss://your-project.livekit.cloud
NEMARA_LIVEKIT_API_KEY=APIxxxx
NEMARA_LIVEKIT_API_SECRET=secretxxxx

# LLM providers
NEMARA_GEMINI_API_KEY=AIza...
NEMARA_OPENAI_API_KEY=sk-...

# APNs (paste the .p8 file contents preserving newlines)
NEMARA_APNS_TEAM_ID=ABCDE12345
NEMARA_APNS_KEY_ID=XYZ9876543
NEMARA_APNS_AUTH_KEY="-----BEGIN PRIVATE KEY-----
MIGTAg...
-----END PRIVATE KEY-----
"
NEMARA_APNS_BUNDLE_ID=co.elephantlabs.nemara
NEMARA_APNS_ENVIRONMENT=sandbox
```

### A5. Both iOS apps point at the laptop

In each `.xcodeproj`: **Edit Scheme → Run → Arguments → Environment
Variables**, override:

- `NEMARA_BACKEND_URL` → `http://<your-Mac's-LAN-IP>:8002` (e.g.
  `http://10.0.0.245:8002`)
- `SUPABASE_URL` → `http://<your-Mac's-LAN-IP>:54341`

The caregiver app's `SUPABASE_ANON_KEY` stays as the well-known local
key (already in the scheme).

### A6. Launch order

```bash
# Terminal 1
cd nemara-backend
supabase start
make db-reset
make dev                           # backend on :8002

# Terminal 2
ngrok http 8002                    # paste the public URL into LiveKit webhooks
```

Then build + run both apps onto their phones from Xcode (⌘R, target
= real device).

---

## B. Golden paths

Each step unlocks the next. If any step fails, fix before continuing.

### B1. Pair the wall phone

1. Caregiver app — Home — open the seeded patient "Maggie" (or
   onboard a new patient) → Settings → "Pair a new phone".
2. A 6-digit code appears with a 10-min countdown.
3. Wall phone — `PairingView` is the launch screen on a fresh
   install. Enter the code → Pair.

**Expected:** Wall phone swaps to the live screen. LiveKit status
flips to "Connecting" then "Connected" within ~3s. Motion heartbeat
ticks every 5s.

### B2. Live tab + Talk

1. Caregiver app — Maggie — Live tab.
2. Tap the phone-call button on the camera tile.
3. Speak into your caregiver phone — sound comes out of the wall
   phone's speaker.
4. Tap call button again to end.

**Expected:** Wall phone shows "ON AIR" badge during the call; goes
silent on hang-up.

### B3. Detector fires → APNs → tap → clip playback

1. Background the caregiver app (swipe up, don't quit).
2. **Trigger a fall:** drop a stuffed animal in front of the wall
   phone, briefly squat to the floor in view, or fall onto a couch.
3. **Backend logs:** `event_promotion.fired event=…`, then
   `push.delivered event=ok device=…`, then ~15s later
   `clip_archival.done event=… key=events/<uuid>.mp4`.
4. **Caregiver phone:** APNs banner on the lock screen within ~5s of
   the event log. Title = "Maggie", body = the narrative.
5. Tap the notification.
6. **Manual nav** (deep-link routing isn't wired yet): Home → Maggie
   → Events → tap the new event.

**Expected:** EventDetail shows "Clip processing…" placeholder if
you opened within 15s of the trigger, then flips to inline
`VideoPlayer` once `has_clip=true` arrives via Realtime. Tap play →
~30s of footage with audio.

### B4. Audio detector

1. Wall phone in a quiet room, app running.
2. Backend logs for a tick — `detector_fields` should have
   `speech_heard: false`.
3. Speak clearly near the wall phone: "Help, I've fallen."
4. **Expected:** Next tick's `speech_heard: true`. If
   agitation/fall confidence is above the promotion threshold, B3
   fires again.

### B5. Patient photo

1. Caregiver — Maggie — Settings → tap the camera-chip badge on the
   avatar → PhotosPicker → select an image.
2. **Expected:** Within ~2s the avatar updates in the settings card,
   the header, AND the Home roster (verify all three).
3. Sign out → sign in → photo persists.
4. Tap "Remove photo" in settings.
5. **Expected:** Avatar reverts to initials in all three places.

---

## C. Resilience scenarios

Run after the goldens pass.

### C1. JWT expiry (needs >1h wall-clock)

- Start a session per B1+B2. Leave the wall phone running.
- After ~55 min, watch the camera-app + backend logs for the
  refresh — the participant_left + participant_joined cycle is the
  visible artifact.
- After 1h 5min, walk in front of the camera.

**Expected:** Detection still works. No "Connection lost" / black
screen on the caregiver Live tab.

### C2. LiveKit reconnect — wall phone

- During an active session: wall phone → Airplane Mode ON for 30s →
  OFF.

**Expected:** Wall phone reconnects. Backend logs
`participant_left` then `participant_joined`. Ticks resume.

### C3. LiveKit reconnect — subscriber bot

- Mid-session: laptop → toggle Wi-Fi off for 30s → back on.

**Expected backend logs:** `room_worker.disconnected … reconnecting`
then `room_worker.connected` within ~30s, then ticks resume. The
exponential-backoff loop from the FT commit owns this path.

### C4. Backend restart

- Mid-session: backend on laptop → `Ctrl-C` then `make dev` again.

**Expected:** Wall phone keeps publishing. New backend boots,
webhook fires when bot rejoins, ticks resume. No duplicate
`analyses` rows (verify in Supabase Studio).

### C5. APNs while killed

- Caregiver app — swipe-kill from app switcher.
- Trigger a fall.

**Expected:** Push still lands on lock screen. (APNs delivers to
the OS, not the app process.)

### C6. Caregiver Realtime drop

- During an active session: caregiver phone → Airplane Mode ON for
  30s.
- Trigger a fall during that window.
- Turn airplane mode OFF.

**Expected:** Within seconds the event appears on the Events tab.
Supabase Realtime re-subscribes and backfills.

### C7. Photo upload backpressure

- Edit the caregiver app's downsample logic temporarily (or use a
  debug build that disables it) to send a 5 MB JPEG.

**Expected:** 413 response. Backend logs `image_bytes too large`.
Backend RAM stays flat (verify with `top` / Activity Monitor) —
the streaming size cap from the FT commit prevents buffering the
full body.

### C8. Cross-caregiver isolation

- Create a second caregiver account.
- Sign in as caregiver B.

**Expected:** Maggie does not appear in the roster. No events. No
photo URLs accessible. RLS catches every leak vector.

---

## D. Diagnostics during testing

```bash
# Tail backend logs
make dev | grep -E "(tick|push|clip_archival|room_worker)"

# Spot-check DB
psql postgresql://postgres:postgres@127.0.0.1:54342/postgres
# select id, title, has_clip, clip_storage_key from events order by occurred_at desc limit 10;
# select id, preferred_name, photo_storage_key from patients;
# select apns_token, environment from caregiver_devices;

# Inspect a stored clip
# Supabase Studio → Storage → incident-clips → events/{event_id}.mp4 → Download

# Force-fire an event (no APNs / no clip — direct INSERT only)
uv run e2e fire-event --patient maggie --kind fall
```

---

## E. Known gaps you'll hit

Flagged here so they aren't misread as bugs:

- **Notification tap doesn't deep-link** to the event row (B3 step
  6 → 7). The AppDelegate posts `.nemaraEventPushTapped` but
  RootView doesn't subscribe yet. ~30 min follow-up.
- **`e2e fire-event` skips Phase 4 hooks.** It inserts the row
  directly; APNs + clip-archival run only from the real detector
  pipeline. The CLI plumbing fix is a small follow-up.
- **Live-view sensitivity gate not built yet.** Tapping Talk while
  the patient is changing clothes / using the bathroom has no
  current safeguard.
- **Custom-prompt input sanitizer not built yet.** A caregiver can
  add a detector with any prompt text.

---

## F. Pass criteria

Done when **all of B1–B5 + C1–C8 pass on real hardware**. Anything
in C that fails is a real bug to chase before shipping; anything in
B that fails is a blocker.

Expect 2–3 issues on the first end-to-end pass. The gap between
unit/integration tests and real Wi-Fi + cellular + APNs servers is
where bugs hide.
