# Thermal UAV Analysis — Demo Story & Runbook

> The one-line pitch: **an AI analyst you can talk to that watches thermal drone
> footage for you, learns what you care about from a few example boxes (no
> training), and hands you a short, ranked list of the moments that matter —
> each pinned to real pixels so you can trust it.**

Everything below maps to open-source tools that exist today: **SAM 3** (find +
segment + track), **Qwen3-VL** (describe + answer), a small **database** for
recall, and OpenCV overlays for marking. **No image generation** is used or
claimed — marking is drawing the boxes and masks SAM 3 already returns.

---

## 1. Pick one buyer before you present

The stack can serve several verticals, but a demo that tries to do all of them
reads as a bag of features. Choose one story and set the watch-list + severity
language to match:

- **Perimeter / security ISR** — "AI watch officer": flags approaches, new
  vehicles, loitering. Best fit for thermal + people + movement.
- **Search & rescue** — heat signatures at night over wide areas. The most
  sympathetic story; thermal is genuinely ideal.
- **Humanitarian camp monitoring** — count tents, watch a camp grow/move.
  NGO/UN buyers; fits the tents angle.

> Solar / infrastructure inspection is real but a *different* product (defect
> detection on panels, not activity). Don't mix it into this demo.

This runbook is written for **perimeter / camp monitoring**; swap nouns for your
chosen vertical.

---

## 2. The narrative arc (what the audience should feel)

1. **The problem is attention, not vision.** Show ~10 seconds of raw thermal
   feed. It's ugly, jittery, and nobody could watch hours of it. "An operator
   can't watch everything — that's the actual problem."
2. **You teach it by pointing, not training.** Pause a frame, draw a few boxes
   ("these are tents", "this is a person", "that bright blob is a fire"), name
   them, set priority. It's live in seconds. Say the line: *"No dataset, no
   training run — I just showed it."*
3. **It watches so you don't.** Let the clip play. The live view is small; the
   **event feed** on the right fills up with flagged moments. The system is
   doing the watching.
4. **Every flag is trustworthy.** Click an event. It's pinned to the exact
   frame, mask, and track that triggered it, with a short grounded description.
   *"It can't hand-wave — everything points back at real pixels."*
5. **You can interrogate it.** In the detail view, ask a question or two about
   that moment. The answer is scoped to what's on screen.
6. **It remembers.** Ask about earlier in the session ("when did the vehicle
   first appear?"). It pulls from the database and jumps you to that frame.
7. **The deliverable.** Export a one-page **incident timeline** of the clip.
   That artifact — "here's everything that happened" — is the close.

---

## 3. Console layout (the UI in one picture)

```
┌──────────────┬───────────────────────────────┬────────────────────────────┐
│  LEFT RAIL   │           CENTER              │       RIGHT RAIL (HERO)    │
│              │                               │                            │
│ Stream       │   Live feed (small)           │  Event feed                │
│ selector     │   + SAM 3 overlays:           │  (reverse chronological)   │
│              │     masks / boxes on           │                            │
│ Watch-list   │     priority objects,          │  ┌──────────────────────┐  │
│  ● tent      │     faint markers on           │  │ [thumb] 14:22:07      │  │
│  ● person    │     generic objects,           │  │ person → tent cluster │  │
│  ● vehicle   │     motion trails.             │  │ severity ●  [confirm] │  │
│  ● fire      │                               │  └──────────────────────┘  │
│              │   Attention cue on the         │  ┌──────────────────────┐  │
│ [+ add ctx]  │   most-salient object.         │  │ [thumb] 14:21:40 ...  │  │
│ [ library ]  │                               │  └──────────────────────┘  │
├──────────────┴───────────────────────────────┴────────────────────────────┤
│  TIMELINE STRIP  ▪──────▪────────▪──────────▪───────▪  (event ticks)        │
└─────────────────────────────────────────────────────────────────────────────┘
```

Key UX principle: **the live video is glanceable; the event feed is the hero.**
If the video is big and centered you've built a fancy CCTV monitor. If the event
feed is primary, you've built an AI analyst. The whole thesis (reduce attention
load) becomes visible design.

### Region notes
- **Left — context / watch-list.** First-class panel, not a settings dialog.
  Each concept: name, color, exemplar thumbnails, priority (alert/watch/ignore),
  toggle. Two entry points: *draw new* (pause → box → name) and *reuse* (apply a
  saved concept pack in one click).
- **Center — live view.** Small. Overlays let the audience *see the machine
  thinking*. New events do **not** interrupt the video — a card slides into the
  right rail.
- **Right — event feed.** Each card: thumbnail, timestamp, one grounded line,
  severity color, concept tag, and a quick verdict (**dismiss / confirm /
  escalate**). Clicking opens the detail/investigation view — the only place
  chat lives, always scoped to a clicked moment (never a free-floating box).
- **Bottom — timeline strip.** Session as a ribbon of incident ticks; jump to
  any. This is also what you export.

---

## 4. Tiers of analysis (surface them in the UI)

| Tier | Trigger | Work | Shown as |
|---|---|---|---|
| 1 — Passive | every sampled frame | SAM 3 detect/track, overlays, motion | live overlays |
| 2 — Event | a track raises an event | Qwen3-VL writes the card's one-liner | event card text |
| 3 — On demand | operator clicks a card | full reasoning + Q&A + cross-reference | "deep-analyzed" badge |

A card gaining a **deep-analyzed** badge after you investigate it makes the
escalating intelligence feel real and honestly reflects what's running.

---

## 5. Step-by-step demo script

**Setup (before the room):**
- vLLM/SGLang serving Qwen3-VL is up; SAM 3 weights loaded; DB running.
- A 2-minute thermal clip where your marked concepts are reasonably visible.
- 3–5 example boxes per concept already prepared as a fallback, in case live
  drawing under pressure is fiddly.

**Live run (~4 minutes):**
1. Play 10s of raw feed. State the problem: attention, not vision.
2. Pause. Draw boxes for `tent`, `person`, `vehicle`, `fire`. Name + prioritize.
   → *"No training. I just pointed."*
3. Resume. Narrate the event feed filling. Point out overlays tracking a person.
4. Click the **person → tent cluster** event. Show the pinned frame/mask/track.
5. Ask one question in the scoped chat (e.g. "how many people near the tents?").
6. Ask a **memory** question ("when did the vehicle first appear?") → jumps to
   the earlier frame from the DB.
7. Export the **incident timeline**. Hand it over as the deliverable. Done.

---

## 6. What is real vs stubbed for the demo

**Build for real:** three-zone layout; context panel with live box-drawing;
SAM 3 overlays; event feed populating from real detections/events; one card →
detail view with a scoped VLM Q&A; timeline export.

**Stub / fake convincingly:** context library packs; triage-feedback learning;
multi-stream switching; the full agent-over-database tool suite (a couple of
canned queries is enough to tell the story).

---

## 7. On-stage safety (don't get embarrassed)

- **Script around a favorable clip.** Ugly thermal + text-only prompts will
  miss. Lean on the **example boxes**, not text prompts, for recall.
- **Keep VLM lines terse.** A short grounded sentence reads as competent; a
  paragraph invites the audience to spot where it's wrong.
- **Don't promise instant chat.** It's chat with an index updated every few
  seconds. Say that.
- **Never imply generated frames.** The trust story dies the moment the system
  looks like it invents imagery. Marking = overlays on real pixels, always.
- **Have the fallback boxes ready** so a fumbled live drawing doesn't stall you.

---

## 8. Tech behind each beat (cheat sheet)

| Demo beat | What's actually running |
|---|---|
| Draw boxes → concept goes live | boxes stored as `context_exemplars`; fed to SAM 3 as image exemplars + DINOv2 prototypes |
| Overlays tracking objects | SAM 3 masks + IDs (Tier 1), drawn with OpenCV/supervision |
| Event card appears | geometric event (new/approach/low-conf) → Qwen3-VL one-liner (Tier 2) |
| Click → detail + Q&A | Qwen3-VL over the event's frame/crop (Tier 3), grounded to track ID |
| "When did X appear?" | DB query over `events`/`tracks`, jump to `frames.thumb_path` |
| Export timeline | render `events` for the session to a one-pager |
