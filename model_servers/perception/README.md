# `perception/` — SAM 3 + DINOv3, in-memory

This package holds the **real-time perception** models. It is **imported and run
in-process by the backend** (not served over HTTP on the live path) — see the
rationale in `../README.md` §1.2. One process, both models co-resident on the GPU,
the frame tensor handed straight from SAM 3 to DINOv3.

## Interface

`interface.py` exposes one class:

```python
from perception import Perception

percept = Perception.from_env()                    # loads SAM 3 + DINOv3 once
dets    = percept.detect_track(frame, concepts, exemplars)   # Tier-1, per frame
embs    = percept.embed(frame, dets)               # DINOv3 on SAM 3's crops
```

- `detect_track` returns boxes + masks (RLE) + stable track IDs.
- `embed` returns a vector per detection for pgvector similarity / exemplar match.

## Backends (`PERCEPTION_BACKEND`)

| Value | Detect + track | Concept prompts | Availability |
|---|---|---|---|
| `sam3` (default) | SAM 3 native | text + example boxes | preferred; confirm weights |
| `sam2_gdino` | SAM 2.1 + GroundingDINO (or YOLO-World) | text via GDINO, boxes via DINOv3 match | fully available offline today |

The backend switch is invisible to the rest of the system — same return types.

## Weights (env, set by the profile)

- `SAM_WEIGHTS=/models/sam3`
- `DINOV3_WEIGHTS=/models/dinov3`
- fallback: `SAM2_WEIGHTS=/models/sam2`, `GDINO_WEIGHTS=/models/groundingdino`

## Debug API (optional, never on the live path)

`service.py` wraps the same class in a thin FastAPI (`:8090`) for poking from curl
or other clients. Start it with `docker compose --profile debug up -d perception-api`.
Do **not** route the live pipeline through it — the whole point of this package is
to avoid that hop.
