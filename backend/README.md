# `backend/` — the orchestrator

Ties together the UDP video+telemetry stream (`uav_udp_stream/`), perception
(SAM 3 + DINOv3, `model_servers/perception/`), and the VLM (`model_servers/`'s
`vllm`/`vlm` services) into the Tier-1/Tier-2 pipeline described in
`docs/Architecture_and_Technology.md`, and exposes it to `frontend/` over a
WebSocket + REST API.

## What's real here, and what isn't

Built and tested against live perception (real SAM 3 + DINOv3 calls, not
mocked) and a real vLLM instance:

- Tier-1: samples the video stream, runs SAM 3 detection + DINOv3 exemplar
  matching, tracks identities frame-to-frame (see `app/tracker.py` — SAM 3's
  own video-predictor API would replace this eventually, see its docstring),
  broadcasts over `/ws`.
- Tier-2: fires a "first appearance" event through the VLM for a one-line
  description the moment a track is created.
- Context authoring: `POST /concepts` + `POST /concepts/{id}/exemplars` — the
  same DINOv3-embedding mechanism as
  `model_servers/perception/tools/embed_concept.py`, live instead of a script.

**Not done — do not assume these work:**

- **No database.** `app/store.py` is an in-memory dict; concepts and their
  exemplars are lost on restart. There is no Postgres/pgvector anywhere in
  this repo yet (see `model_servers/README.md`'s data-model gap).
- **The frontend doesn't call any of this yet.** `frontend/src/lib/feed.ts`
  still wires `MockFeed` into `App.tsx`. `/ws`'s message shape matches the
  `Feed` interface's `onTracks`/`onTelemetry`/`onEvent` handlers exactly (see
  `app/pipeline.py`'s docstring), but nothing implements `WsFeed` yet.
- **Tier-3 chat has no backend at all.** `frontend/src/components/DetailView.tsx`
  calls `mockAnswer()` directly — Q&A never went through the `Feed`
  abstraction to begin with, so wiring it needs a new endpoint (and a
  decision about scoping: send the pinned frame + track history + question,
  probably `POST /events/{id}/ask`) plus a frontend change, not just a
  backend one.
- **"Draw concept" in `CenterFeed.tsx` doesn't draw anything.** It's a
  name-only text input; clicking "Add & go live" fabricates a concept with
  `exemplars: 1` client-side and calls nothing. Real box-drawing on the
  canvas, wired to `POST /concepts/{id}/exemplars`, is frontend work that
  hasn't been done.
- **Track IDs are approximate.** `app/tracker.py`'s greedy IOU matcher is a
  deliberate stand-in for SAM 3's real video-predictor + persistent
  `inference_state` (see `model_servers/perception/interface.py`).
- **Only "first appearance" fires events.** The "approach" and
  "low-confidence" event kinds `frontend/src/lib/types.ts` defines (and
  `MockFeed` simulates) have no real trigger logic yet.

## Running it

Needs `model_servers/`' services up first — see that directory's README.
Then, from the repo root:

```bash
docker compose up -d backend       # builds from repo root context, see ../docker-compose.yml
docker compose logs -f backend
curl localhost:8000/health
```

`STREAM_URL` (default `udp://host.docker.internal:5600`) must point at a
live UDP source — e.g. `uav_udp_stream/stream_udp.py` or
`make_video_stream.py` sending to that port. Without one, `/health` and the
REST endpoints still work; the pipeline thread logs a connection failure and
retries every 3s rather than crashing the app (see `app/pipeline.py`).

## Layout

```
backend/
├── Dockerfile          # build context = repo root (needs perception/ + uavstream/)
├── requirements.txt
└── app/
    ├── main.py         # FastAPI app: lifespan, /ws, /concepts REST
    ├── config.py        # env-driven settings
    ├── schemas.py        # wire models — kept in lockstep with frontend/src/lib/types.ts by hand
    ├── ingest.py         # PyAV video decode + uavstream.klv telemetry decode
    ├── tracker.py        # greedy IOU tracker — frame-to-frame identity stand-in
    ├── pipeline.py        # Tier-1/2 loop: ingest -> perception -> tracker -> VLM -> on_message
    ├── store.py          # in-memory concept/exemplar store (no DB yet)
    └── vlm_client.py      # OpenAI-compatible /v1/chat/completions calls
```
