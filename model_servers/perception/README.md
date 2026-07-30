# `perception/` — SAM 3 + DINOv3

This package holds the **real-time perception** models: SAM 3 (detect/segment,
concept-promptable) feeding DINOv3 (embeds SAM 3's crops for verification /
exemplar matching). It's imported **in-process by `backend/`** (see
`../../backend/app/pipeline.py`) — not served over HTTP on the live path, per
the rationale in `../README.md` §1.3. `perception-api` (this directory's own
`Dockerfile`/`service.py`, `:8090`) is now back to being what it was always
meant to be — a debug/testing surface, not the live path.

## Interface

`interface.py` exposes one class:

```python
from interface import Perception

percept = Perception.from_env()                 # loads SAM 3 + DINOv3 once
dets    = percept.detect_track(frame, concepts)  # Tier-1, per frame
dets    = percept.embed(frame, dets)             # DINOv3 on SAM 3's crops
```

`Concept.exemplar_boxes` carries the example boxes (no separate `exemplars`
argument) — see the dataclass in `interface.py`.

- `detect_track` returns boxes + masks (RLE) + a track ID per detection.
- `embed` returns a vector per detection for pgvector similarity / exemplar match.

**Known gaps, flagged in `interface.py`:** SAM 3's exemplar-box prompting
(`Sam3Processor.add_geometric_prompt`) is confirmed to exist but isn't wired
in — this project's exemplar matching goes through DINOv3 similarity instead
(see "Context authoring" below). And `track_id` here is just an incrementing
counter, not a real persistent identity, because `detect_track()` calls SAM
3's single-image API fresh per frame with no memory between calls — real
tracking needs SAM 3's video-predictor API with one `inference_state` kept
alive per stream. `backend/app/tracker.py` papers over this today with a
simple greedy IOU matcher; see that file's docstring.

## Backends (`PERCEPTION_BACKEND`)

| Value | Detect + track | Concept prompts | Availability |
|---|---|---|---|
| `sam3` (default) | SAM 3 native | text prompt confirmed; exemplar boxes not yet wired (see above) | gated on HF — request access to `facebookresearch/sam3` |
| `sam2_gdino` | SAM 2.1 + GroundingDINO (or YOLO-World) | text via GDINO, boxes via DINOv3 match | not implemented yet — raises `NotImplementedError` |

The backend switch is invisible to the rest of the system — same return types.

## Weights (env, set by the profile)

- `SAM_WEIGHTS=/models/sam3`
- `DINOV3_WEIGHTS=/models/dinov3-vitl16`
- fallback: `SAM2_WEIGHTS=/models/sam2`, `GDINO_WEIGHTS=/models/groundingdino`

## Context authoring (concept packs) — before `backend/` exists

Until there's a real UI for drawing exemplar boxes, `tools/` gives you the same
result from the command line, in two steps because box-drawing needs a display
and embedding needs the GPU/model stack — usually different machines/contexts:

```bash
# 1) HOST — draw a box on each reference image (needs `pip install opencv-python`)
python tools/select_boxes.py --label tent --images tent1.jpg tent2.jpg tent3.jpg
#   -> concepts/tent.boxes.json

# 2) CONTAINER — turn boxes into a concept pack (DINOv3 embeddings)
docker compose --profile debug up -d perception-api
docker compose exec perception-api python tools/embed_concept.py \
    --boxes concepts/tent.boxes.json --label tent --priority watch
#   -> concepts/tent.json  (label, priority, text_prompt, reference_embeddings)

# 3) CONTAINER — try it against a real frame
docker compose exec perception-api python tools/test_inference.py \
    --frame sample.jpg --concepts concepts/tent.json concepts/person.json \
    --out annotated.jpg
```

**Why this doesn't use SAM 3's exemplar-box prompting directly:** that API
isn't confirmed (see the gap noted above). Instead, `test_inference.py` uses
SAM 3's **text prompt** for broad recall, then DINOv3 for the actual "does it
look like my examples" gate via cosine similarity against `reference_embeddings`
(`--threshold`, default 0.55). This is deliberately more reliable than depending
on an unverified API, at the cost of needing a reasonably distinctive text
prompt per concept. Concept packs (`concepts/*.json`) are small text files and
are checked into git — they're the reusable "concept pack" asset the
architecture doc describes, not disposable output.

## Debug API

`service.py` wraps the same class in a FastAPI (`:8090`) for poking at SAM 3 +
DINOv3 directly with curl, independent of the full ingest/tracking/VLM
pipeline in `backend/`. Start it with
`docker compose --profile debug up -d perception-api` — see `../README.md` §6
for curl examples. Never route the live pipeline through it; `backend/`
imports `Perception` in-process instead.
