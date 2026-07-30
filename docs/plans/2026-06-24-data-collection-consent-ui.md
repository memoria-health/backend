# Data-Collection Consent UI — Implementation Plan (web + iOS)

> Status (2026-07-27): implemented — web (`DataCollectionConsentModal.tsx`, `DataCollectionConsentGate.tsx`, `DataCollectionSettings.tsx` in nemara-web-app) and iOS (`DataCollectionConsentSheet.swift`, `DataCollectionCard.swift` in nemara-caregiver-app) both shipped; backend flag + `decided_at` stamping live in `src/schemas/patient.py` / `src/db/patient.py`.

> **For Claude:** REQUIRED SUB-SKILL: superpowers:executing-plans. Companion to `2026-06-24-video-data-collection-recorder.md` (the backend recorder). Ship this **with** the recorder — recording must not run without the consent flow live.

**Goal:** Per-patient consent for data collection on **both** the web app and the iOS caregiver app: a **first-open banner** (shown until a decision is recorded) + a **settings toggle** (changeable anytime), both driving the backend `data_collection_enabled` flag.

**Backend contract (from the recorder plan, Phase 1–2 — must land first):**
- `patients.data_collection_enabled` (bool, default false) + `patients.data_collection_decided_at` (timestamptz, null).
- `PATCH /v1/patients/{id}` accepts `data_collection_enabled`; backend **stamps `decided_at` whenever `data_collection_enabled` is present in the payload** (so both "Share" and "Not now" record a decision).
- Patient read endpoints return both fields.
- **`decided_at IS NULL` ⇒ never asked ⇒ show the banner.** Non-null ⇒ decided ⇒ never auto-show again; `enabled` is the actual on/off.

**Consent copy (shared, plain language — get legal review):**
> **Help improve care for everyone.** With your permission, Memoria will securely store short video **and audio** clips of [patient] — only when someone is present — to help improve our AI's understanding of elderly health and disease. This is optional, never required, and you can turn it off anytime in Settings.
> **[ Share recordings ]   [ Not now ]**

---

## Conventions
- Branch per repo (`feat/data-collection-consent`), PR, never main.
- Both surfaces: **optimistic update → PATCH → rollback on error**, control disabled during the request (mirror the existing email-alerts toggle).
- "Share" → PATCH `{data_collection_enabled: true}`; "Not now" → PATCH `{data_collection_enabled: false}`. Both stamp `decided_at`, so the banner won't re-nag; the Settings toggle is how they change later.

---

## Phase A — Web (`nemara-web-app`)

### A1. Extend types
`lib/api/types.ts`: add to `Patient` (`data_collection_enabled: boolean`, `data_collection_decided_at: string | null`) and to `UpdatePatientRequest` (`data_collection_enabled?: boolean`). Commit.

### A2. Consent modal component
Create `app/_components/DataCollectionConsentModal.tsx`, mirroring `app/_components/OnboardingModal.tsx` (plain `role="dialog"` div, no new dep). Props: `patientName`, `onDecide(enabled: boolean)`. Renders the copy + two buttons → calls `api.updatePatient(patientId, { data_collection_enabled })` then closes. Test: clicking each button calls `updatePatient` with the right value (mock the api). Commit.

### A3. First-open trigger
In `app/(app)/patients/[id]/page.tsx` (already loads full patient state): if `patient.data_collection_decided_at == null`, render `<DataCollectionConsentModal>`. After a decision, refetch/optimistically update so it doesn't reappear. Commit.

### A4. Settings toggle
Create `app/_components/DataCollectionSettings.tsx` mirroring `EmailAlertsSettings.tsx` (the `role="switch"` button + optimistic toggle + rollback). Add it as a section in `app/(app)/patients/[id]/settings/page.tsx`. Toggling PATCHes `data_collection_enabled`. Commit.

---

## Phase B — iOS caregiver (`nemara-caregiver-app`)

### B1. Extend model
`Models/Patient.swift`: add `dataCollectionEnabled: Bool` + `dataCollectionDecidedAt: Date?` to `Patient` (+ CodingKeys `data_collection_enabled` / `data_collection_decided_at`), and `dataCollectionEnabled: Bool?` to `UpdatePatientRequest` (+ CodingKey + `encodeIfPresent`). Commit.

### B2. Consent sheet
Create `Features/Patient/Settings/DataCollectionConsentSheet.swift`, mirroring `PairingCodeSheet.swift`'s `.sheet()` + state pattern. Renders the copy + Share / Not now → `env.api.updatePatient(patientId, UpdatePatientRequest(dataCollectionEnabled: decision))`, then dismiss. Commit.

### B3. First-open trigger
In `Features/Patient/PatientShell.swift`: a `@State private var showConsent` set on appearance when `env.api.patients.first{ $0.id == patientId }?.dataCollectionDecidedAt == nil`; present `DataCollectionConsentSheet` via `.sheet()`. Commit.

### B4. Settings toggle card
Create `Features/Patient/Settings/DataCollectionCard.swift` mirroring `PatientEmailAlertsCard.swift` (uses `NemaraToggle`, optimistic + rollback via `env.api.updatePatient`). Add it to `PatientSettingsView.swift`. Commit.

---

## Verification checklist
- [ ] Banner shows **only** when `decided_at IS NULL`, on first open, on both surfaces.
- [ ] "Share" and "Not now" both record a decision (`decided_at` set) → banner never re-nags.
- [ ] Settings toggle reflects + changes `data_collection_enabled` anytime, optimistic + rollback.
- [ ] Default is **OFF**; nothing records until explicit opt-in.
- [ ] Copy reviewed by legal (consent-capacity + biometric framing — see recorder plan's risk note).
- [ ] Ships together with the backend recorder (no recording without consent flow live).
