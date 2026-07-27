# `frontend/` — UAV Analysis Console

The operator console: **React + Vite + TypeScript + Tailwind + shadcn/ui**,
implementing the three-zone layout from `docs/DEMO.md` §3 — left watch-list,
small glanceable center feed with SAM-3-style overlays, hero event feed on the
right, timeline strip at the bottom, and a detail view with scoped Q&A.

## Why this stack

- **shadcn/ui over MUI** — components are source you own (vendored under
  `src/components/ui/`, MIT-licensed, no CLI/network dependency), Radix-backed
  for accessibility, and fully restyleable for a dense, bespoke dark ISR
  console rather than a generic "Material" look.
- **Tailwind** — utility classes keep the dense card/panel layout consistent
  without a growing CSS file; the whole theme (colors, radii) is CSS variables
  in `src/index.css` so light/dark and rebranding are one-file changes.
- **Vite** — instant HMR in dev, a small static bundle for prod (served by
  nginx per the deployment plan), zero runtime dependency on a bundler server.
- **Zustand** — the event feed, watch-list, and live tracks are simple client
  state; Zustand avoids Redux boilerplate for a single-operator console.
- **Canvas, not `<video>`, for the live feed** — the backend will draw SAM 3
  overlays server-side and push frames as JPEGs over WebSocket; a `<canvas>`
  is the simplest fully-offline low-latency path (no WebRTC signaling to run
  locally). `lib/feed.ts` documents the swap to a real `WsFeed`.
- **No external fonts/CDN** — the UI uses system UI fonts by default so it
  renders identically fully air-gapped; see `index.css`.

## Run it

```bash
npm install       # one-time, needs internet (or an offline npm cache/mirror)
npm run dev        # http://localhost:5173
npm run typecheck
npm run build       # -> dist/, static assets for nginx
```

`vite.config.ts` proxies `/api` and `/ws` to `localhost:8000` (the future
FastAPI backend), so once the backend exists, the console talks to it over the
same origin in dev with no CORS config.

## Current state: simulated, backend-ready

There is no backend yet. `src/lib/feed.ts` runs an in-browser **`MockFeed`**
that generates plausible tracked objects (person/vehicle/tent/fire), KLV-style
telemetry, and geometric events (first-appearance, approach, low-confidence) —
enough to demo the full UX end to end today. It implements one `Feed`
interface; swapping in a real backend is writing a `WsFeed` that opens `/ws`
and calls the same `onTracks` / `onTelemetry` / `onEvent` handlers — no other
component changes.

Similarly, `mockAnswer()` in the same file stands in for the Tier-3 VLM Q&A in
the detail view until `POST /api/events/:id/ask` exists.

## Layout

```
src/
├─ components/
│  ├─ ui/            # vendored shadcn primitives (button, card, dialog, ...)
│  ├─ TopBar.tsx      # status bar: profile, model health, clock
│  ├─ LeftRail.tsx    # stream selector + watch-list (draw-to-teach entry point)
│  ├─ CenterFeed.tsx  # canvas: live frame + SAM-3-style mask/box/trail overlays
│  ├─ EventFeed.tsx   # hero: reverse-chronological event cards
│  ├─ EventCard.tsx   # thumbnail, grounded line, severity, triage actions
│  ├─ TimelineStrip.tsx  # session ribbon of event ticks, exports later
│  └─ DetailView.tsx  # click-through: pinned frame + scoped chat (Tier 3)
├─ store/useStore.ts  # zustand: streams, concepts, tracks, events, chat
├─ lib/
│  ├─ types.ts        # shared domain types (mirrors the backend's schema)
│  ├─ feed.ts          # Feed interface + MockFeed (swap point for the real WS)
│  └─ utils.ts
└─ index.css           # theme tokens (CSS vars), dark ISR palette
```

## Design notes (from `docs/DEMO.md`)

- **The event feed is the hero**, not the video — the whole thesis (reduce
  attention load) is visible in the layout: center feed is small, right rail
  is wide.
- **Tiers are surfaced honestly**: overlays = Tier 1, the card's one-liner =
  Tier 2, a "deep-analyzed" badge appears only after the operator opens an
  event and Tier-3 reasoning runs (`DetailView` sets this after the dialog
  opens, standing in for the real VLM call).
- **Chat only lives in the detail view**, always scoped to one event/track —
  never a free-floating assistant box.
- **Draw-to-teach** is on the center feed (`Draw concept` pauses playback and
  opens a name/priority form) so "no training, just pointing" is the literal
  interaction, not a settings dialog.
