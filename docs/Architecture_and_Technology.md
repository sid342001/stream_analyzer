# Thermal UAV Analysis System
## Architecture & Technology Reference

*Open-source stack · single-GPU deployment · draft for demo planning*

---

## 1. Purpose and the outcome we sell

This system turns a firehose of hard-to-watch thermal UAV video into a short, ranked list of moments a human needs to look at — each described in plain language, pinned to the exact frame and object that triggered it, and available to question on demand. The value is not "AI that sees." The value is collapsing an analyst's attention load: instead of watching everything, the operator works a queue of what the system flagged.

The capability that makes this new, rather than a repackaged CCTV analytic, is zero-shot generalization. The operator specifies what matters using a few example boxes and short concept names — no training run, no labeled dataset. If those things appear, attention snaps to them; everything else in the scene is still perceived, but held at low priority. "Give it context and it generalizes" is the product.

---

## 2. Component roles — what each piece is for

The system has three functional layers. The most common point of confusion is the split between SAM 3 and the vision-language model, so it is stated plainly: **SAM 3 is the eyes, Qwen3-VL is the brain, and they are not interchangeable.**

**SAM 3 performs localization.** Given concept names plus example boxes, it detects and segments every matching instance in each frame and tracks them across frames with stable identities. It runs on the fast path, on (a sampling of) every frame, and produces boxes, masks, and track IDs. This is precisely what a VLM cannot do reliably: a VLM is too slow per frame and weak at precise, consistent multi-object localization and tracking. SAM 3 answers *where is it and is it the same one as before.*

**Qwen3-VL performs reasoning.** It takes the regions SAM 3 found — as crops or as a marked frame — and produces the plain-language description, the event summary, and answers to operator questions. It runs on the slow path, only when an event warrants it or the operator asks. It answers *what is this and what is happening.*

**A DINOv2 embedding bank is an optional verification layer.** It embeds each tracked crop and matches it, thermal-to-thermal, against the operator's labeled examples, which catches or relabels shaky detections without depending on a text prompt working on ugly infrared.

**Image generation is deliberately excluded from the core.** "Marking" a detection is simply drawing the box and mask SAM 3 already returns — a free overlay operation. Generating imagery in an analysis or monitoring product is a liability: the entire value rests on what is shown being trustworthy evidence tied to real pixels and a timestamp. A generative model has one legitimate, back-office role only — synthesizing example images to bootstrap a rare concept with no real samples — and even there, thermal realism from RGB-trained models is weak, so it is a crutch, not a pillar.

| Component | Technology | Role | License |
|---|---|---|---|
| Perception | SAM 3 (Promptable Concept Segmentation) | Detect + segment + track all instances of the concepts, from text and/or example boxes. Fast path, per frame. | SAM License |
| Reasoning / chat | Qwen3-VL-30B-A3B (MoE, FP8) | Describe events, summarize scene, answer operator questions. Slow path, event- or query-triggered. | Apache 2.0 |
| Verifier (optional) | DINOv2 embeddings + cosine match | Thermal-to-thermal confirmation / relabeling of detections against labeled examples. | Permissive |
| Serving | vLLM or SGLang | OpenAI-compatible endpoint for the VLM so the agent calls it as a tool. | Apache 2.0 |
| Marking | OpenCV / supervision overlays | Draw masks, boxes, labels, motion trails. No model needed. | Permissive |
| Image generation | Excluded from core (e.g. Qwen-Image, offline only) | Optional offline synthesis of rare-concept examples. Not in the live path or the demo. | Apache 2.0 |

All models run comfortably on a single H100 (80 GB) provided the diffusion model is not resident: SAM 3 (~5 GB) + DINOv2 (~3 GB) + Qwen3-VL FP8 (~32 GB) plus buffers leaves ample headroom. Keeping any image generator out of the live loop is what preserves that budget.

---

## 3. The context model

Context is a lightweight, editable watch-list authored by the operator, not a training set. It has two layers that coexist, which is what delivers "guide, not whitelist."

**Priority set.** The concepts the operator explicitly marked — solar panels, fire signatures, people, vehicles — each with a few example boxes. These get low detection thresholds, always-on tracking, and drive alerts. This is "if it appears, focus here."

**Open-vocabulary awareness.** SAM 3 and the VLM still perceive the rest of the scene generically. Novel or unexpected objects are not invisible; they sit at low priority until something makes them interesting — motion, heat, or a person interacting with them. This is "do not only look for them," and it comes essentially for free because the underlying models are open-vocabulary rather than a closed, trained detector.

Context also separates by scope. **Object context** (what a tent, fire, or vehicle looks like) is authored by drawing example boxes and is consumed mainly by SAM 3. **Stream context** (this is a thermal feed, white-hot polarity, EO/IR sensor) is a property of the stream, not an object to be boxed, and is consumed as a textual briefing by Qwen3-VL so its descriptions are correct for the domain. The same raw context the operator provides is therefore split by type: example boxes to the eyes, a scene briefing to the brain.

Context is authored two ways. For a new concept, the operator pauses on a frame, drags boxes around examples, names it, and sets a priority — it is live immediately. For reuse, saved concept packs (for example, "desert camp: tents, vehicles, people, cook-fires") can be applied to any stream in one click. That library is both a strong demo beat and a durable product asset.

---

## 4. Multi-rate pipeline and tiers of analysis

Nothing heavy runs on every frame. Work is staged into three tiers so the console stays real-time on one GPU, and the tier a given result came from is surfaced in the UI so the escalating intelligence is honest and visible.

| Tier | Runs on | What it does | Cost |
|---|---|---|---|
| 1 — Passive | Every (sampled) frame | SAM 3 detection, tracking, overlays; motion analytics (speed, heading). | Cheap |
| 2 — Event | When a track raises an event | Qwen3-VL writes the one-line grounded description on the event card. | Moderate |
| 3 — On demand | Operator clicks into an event | Full scene reasoning, Q&A, cross-referencing earlier events and tracks. | Heavy |

Events are geometric and cheap to raise: a concept appears for the first time, a detection is low-confidence and needs adjudication, or a subject (person) is closing on a target cluster (tents, vehicles). "How is this person walking" is answered by the trajectory in Tier 1 — speed and heading from the track — not by asking the VLM to infer motion from a still.

---

## 5. Data model and database

Everything the system perceives is persisted so that later agentic tooling can retrieve and reason over it: pull the relevant frames, detections, tracks, events, and prior descriptions for a stream and time range, run visual question answering on specific frames, and return grounded results. The recommended store is PostgreSQL with the pgvector extension (so embeddings live alongside the relational data and similarity search is one query); SQLite is adequate for the demo. Full-resolution frames are kept on object storage or disk; the database holds paths, thumbnails, and metadata, not raw video.

| Table | Purpose |
|---|---|
| `streams` | One row per feed. Source type (udp/file), sensor type, palette, polarity, notes. |
| `context_concepts` | The watch-list. Label, color, priority (alert/watch/ignore), optional stream scope or library pack. |
| `context_exemplars` | The example boxes the operator drew: image path, box, crop, and embedding. |
| `frames` | Selectively stored frames (keyframes and event frames): stream, index, timestamp, thumbnail path. |
| `detections` | Per-frame SAM 3 output: frame, concept, track, box, mask (RLE/path), score. |
| `tracks` | Per-instance identity over time: concept, first/last frame, motion summary. |
| `events` | Flagged moments: kind, severity, timestamp, frame, VLM description, operator verdict, tier reached. |
| `vlm_outputs` | Every VLM call: prompt, response, model, latency — for audit and reuse. |
| `conversations` / `messages` | Q&A scoped to an event; messages carry grounding references to frames/detections. |

Sketch of the core schema (PostgreSQL + pgvector):

```sql
CREATE TABLE streams (
  id BIGSERIAL PRIMARY KEY, name TEXT, source_type TEXT,
  sensor_type TEXT, palette TEXT, polarity TEXT, created_at TIMESTAMPTZ DEFAULT now());

CREATE TABLE context_concepts (
  id BIGSERIAL PRIMARY KEY, stream_id BIGINT REFERENCES streams(id),
  label TEXT, color TEXT, priority TEXT CHECK (priority IN ('alert','watch','ignore')));

CREATE TABLE context_exemplars (
  id BIGSERIAL PRIMARY KEY, concept_id BIGINT REFERENCES context_concepts(id),
  image_path TEXT, box INT[4], embedding vector(768));

CREATE TABLE frames (
  id BIGSERIAL PRIMARY KEY, stream_id BIGINT REFERENCES streams(id),
  frame_index BIGINT, ts TIMESTAMPTZ, thumb_path TEXT);

CREATE TABLE tracks (
  id BIGSERIAL PRIMARY KEY, stream_id BIGINT, concept_id BIGINT,
  first_frame_id BIGINT, last_frame_id BIGINT, motion JSONB);

CREATE TABLE detections (
  id BIGSERIAL PRIMARY KEY, frame_id BIGINT REFERENCES frames(id),
  track_id BIGINT REFERENCES tracks(id), concept_id BIGINT,
  box INT[4], mask_rle TEXT, score REAL, embedding vector(768));

CREATE TABLE events (
  id BIGSERIAL PRIMARY KEY, stream_id BIGINT, track_id BIGINT,
  kind TEXT, severity TEXT, ts TIMESTAMPTZ, frame_id BIGINT,
  description TEXT, verdict TEXT, tier INT);

CREATE TABLE vlm_outputs (
  id BIGSERIAL PRIMARY KEY, event_id BIGINT, frame_id BIGINT,
  prompt TEXT, response TEXT, model TEXT, latency_ms INT, created_at TIMESTAMPTZ DEFAULT now());

CREATE TABLE conversations (
  id BIGSERIAL PRIMARY KEY, event_id BIGINT REFERENCES events(id), created_at TIMESTAMPTZ DEFAULT now());

CREATE TABLE messages (
  id BIGSERIAL PRIMARY KEY, conversation_id BIGINT REFERENCES conversations(id),
  role TEXT, content TEXT, ref_frame_id BIGINT, ref_detection_id BIGINT, created_at TIMESTAMPTZ DEFAULT now());
```

The later agent operates over this schema through a small set of tools rather than free-form SQL: fetch events for a stream and time window, fetch a track's history, retrieve a frame, run visual question answering on a frame, and similarity-search detections by embedding (pgvector). Because every message and event references the frame and detection it came from, the agent's answers stay grounded — it can always point the operator at the pixels behind a claim.

---

## 6. Thermal realities and honest limitations

- Thermal is out-of-distribution for these RGB-trained models. Zero-shot text prompts on grainy infrared are flaky; the operator's example boxes are what make it work. Demo on a clip where marked concepts are reasonably visible, and let exemplars do the heavy lifting.
- SAM 3 video is near real-time for roughly five concurrent objects; crowded scenes slow it, which is exactly why the multi-rate staging matters.
- "Chat with the live feed" is chat with a running index that updates every few seconds, not frame-perfect dialogue. Still a strong demo — just not instant.
- VLMs will confidently mis-describe a blurry blob. Grounding is the mitigation and the trust story: every sentence is pinned to a box, track, and timestamp the operator can verify at a glance. Keep on-card descriptions terse.
- The stack is one product per buyer. Search-and-rescue, perimeter security, and camp monitoring are different stories; solar/infrastructure inspection is a genuinely different product (defect detection, not activity). Scope to one vertical for the demo.

---

## 7. Demo scope vs full system

For the demo, build: the three-zone console; a working context panel where drawing boxes on a paused frame makes a concept go live; SAM 3 overlays on the center feed; an event feed that populates as the clip plays; and one event card that opens a detail view with a scoped Q&A. The context library, triage-feedback learning, multi-stream orchestration, and the full agent-over-database tooling can be visually present but lightly stubbed, then hardened after the stack proves out on real footage.
