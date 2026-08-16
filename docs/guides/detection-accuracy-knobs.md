# Detection accuracy — the knobs, and how to argue about them

Written 2026-08-06 after a night that produced 24 false safety alerts from one
camera watching one person who slept normally. Its purpose is to stop us
reasoning about accuracy by anecdote. Every claim below is tied to a
measurement, and anything that is not measured is marked as a guess.

The failure that motivated this: between the iPhone and the Tapo we changed
the camera, the models, the call shapes, the scheduler and the prompts, and we
cannot say which change cost us what — because no FP rate was recorded across
any of them. **That is the actual root cause. Everything else is a symptom.**

---

## 0. The rule that would have prevented all of it

> A safety event may not be decided by the cheapest model on a bundled,
> full-frame, low-frame-rate call.

On the 08-05/06 night, every single false alert — 18 calling-for-help, 4 fall,
1 medication — came from one call shape:

    full:video4:clean:lite:scene      on gemini-3.5-flash-lite

Whole frame, 4 fps, no overlay, cheapest tier, one prompt answering all 8
enabled detectors at once. 4,320 of these ran; the person-crop lane ran 63
times (1.4%) and the validated RAPID lane never ran at all.

---

## 1. Evidence knobs — what the model is allowed to see

These decide whether the answer is *possible*, before the model matters.

| knob | values | what it governs |
|---|---|---|
| spatial | `full` / `person` crop / `face` crop | pixels on the subject |
| temporal | `video4` (4 fps) / 8 fps / native | whether a sub-second event exists in the input |
| window | clip length (15-26s) | how much context per question |
| overlay | `clean` / motion-tinted | whether motion is made explicit or must be inferred |
| modality | muxed A/V / video-only / audio-only | whether one channel's noise contaminates the other |
| resolution | `media_res`, ingress res | effective detail after the provider downscales |

**Measured 08-06.** A fall is a sub-second whole-body event. At 4 fps on a full
1920x1080 frame, a person across the room survives the provider's downscale as
a figure a few tens of pixels tall, sampled maybe twice during the fall. The
model is not seeing a fall and judging it; it is seeing a posture change and
completing the story. That is exactly the observed signature: *"crouches down
and falls onto the floor"*, *"bends to clean the table then collapses onto his
hands and knees"* — narration correct up to the verb.

**Rule of thumb:** the evidence must resolve the phenomenon.
- falls → temporal resolution + person crop
- medication (hand to mouth) → close crop; a full frame cannot resolve it
- speech → clean isolated audio, not muxed with video questions

---

## 2. Model knobs

| knob | note |
|---|---|
| tier | `flash-lite` / `flash` / `pro` |
| thinking level | on 3.5-flash-lite, `minimal`/`low` are measured no-ops |
| provider | gemini / openrouter(Qwen) / openai |

**Measured 08-06, audio-only probe.** Same clips, same audio, two tiers:

| clip | flash-lite | flash |
|---|---|---|
| real call for help | verbatim, 10/10, **yes** | verbatim, 9/10, **yes** |
| ordinary conversation ×3 | garbled; one hallucinated "I'm dying" → **yes** | coherent transcript, **no** on all |
| objectively static audio | UNINTELLIGIBLE 0/10 | UNINTELLIGIBLE 1/10, **no** |

One tier up fixes the calling-for-help FPs outright, on the same audio.

---

## 3. Question knobs — how we ask

| knob | values |
|---|---|
| bundling | one prompt for N detectors / one call per detector |
| shape | free narrative then classify / direct classification |
| refusal | is "I cannot tell" a first-class answer? |

**Measured 08-06.** The bundled call is *worse than the sum of its parts*.
Given audio in isolation, even flash-lite answers "no calling for help" on
most of the FP clips, and the Spanish/Russian quotes never reproduce. Those
quotes are an artefact of asking one cheap model to answer eight questions
about a muxed clip in one pass — the shared narrative gets written once, and
every detector reads the same confabulation.

A single narrative serving 8 detectors means one bad sentence fires several
alerts. It is also why the same tick was simultaneously discarded as sleep
evidence and promoted to a fall alert.

---

## 4. Decision knobs — turning an answer into an event

| knob | status |
|---|---|
| k-of-N across ticks | **not used** on the safety path |
| cross-modal corroboration | not used (audio "help" alone can fire) |
| two-stage adjudication | **does not exist** |
| confidence threshold | `sleep_confidence` measured as a *visibility* score, not a determination score — do not reuse this pattern |
| operator gate | built; held all 6 alerts, paged nobody (key absent) |

A single tick can create a safety event today. The 05:49 fall was one tick
between two ticks of "lying asleep under the covers". Any persistence
requirement at all would have killed it.

---

## 5. Absence detectors are a different class — do not conflate

`Missed evening medication` fired at exactly 19:15:00 with
`alert_trigger: absence`, `window_start 18:00`, `deadline 19:15`. It fires when
no medication-taking is *observed* in the window. It is not a perception FP —
it is a scheduled check, and it is only as good as the detector it depends on.
A missed true positive upstream becomes a false alarm here. Judge it against
recall of the `medication` detector, never on its own.

---

## 6. What to measure (the part we have never done)

For every change to any knob above, record per detector:

- FP rate on the labelled negative set
- recall on the labelled positive set
- cost per occupied-room-hour
- latency

`~/Desktop/eval-set` now holds labelled negatives from the 08-05 night
(`GROUND-TRUTH-2026-08-05.md`), including a matched positive for
calling-for-help. **No knob change should land without a before/after on this
set.** Without that, the next hardware or model swap will be as unattributable
as this one was.

---

## 7. If resources were unlimited

In priority order — the first two are most of the win.

1. **Two-stage: triage then adjudicate.** Keep the cheap always-on scene call,
   but it may only ever *propose*. Every safety candidate is re-examined by a
   frontier model, one detector per call, with evidence chosen for the
   phenomenon (person crop at native fps for falls; isolated clean audio for
   speech). Candidates are rare, so the expensive stage costs little. This
   alone kills every FP catalogued on 08-06.

2. **Unbundle the safety detectors.** One question, one prompt, one answer.
   Stop letting a shared narrative fire several alerts.

3. **Cross-modal corroboration for safety.** A fall claim should be supported
   by motion evidence; a call-for-help claim by an audio-only transcript.
   Disagreement means hold, not fire.

4. **Temporal persistence.** k-of-N on the safety path, so no single tick can
   page anyone.

5. **Continuous eval.** The labelled set runs on every prompt, model or
   pipeline change, and the numbers are the merge criterion.

6. **Then, and only then, tune evidence quality** — resolution, fps, crops,
   audio preprocessing. It is the most obvious lever and the least important
   one, because the 08-06 data shows the current inputs already carry the
   answer. A real call for help transcribed at 10/10 through the same
   microphone that produced 18 false ones.
