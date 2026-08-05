"""Wire-format models — one source of truth, kept in lockstep with
frontend/src/lib/types.ts by hand (there is no codegen here; if you change a
field on either side, change it on both and grep for the old name).

All models serialize with `by_alias=True` so the JSON keys match the
frontend's camelCase exactly, even though Python code uses snake_case.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Priority = Literal["alert", "watch", "ignore"]
Severity = Literal["critical", "warning", "info"]
Verdict = Literal["pending", "dismissed", "confirmed", "escalated"]
Tier = Literal[1, 2, 3]


class Camel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class Point(Camel):
    x: float
    y: float


class TrackedObject(Camel):
    """Matches frontend `TrackedObject`. x/y/w/h are normalized 0..1 within the frame."""

    id: int
    concept: str
    color: str
    x: float
    y: float
    w: float
    h: float
    vx: float
    vy: float
    score: float
    trail: list[Point] = Field(default_factory=list)


class Telemetry(Camel):
    lat: float
    lon: float
    alt: float
    hdg: float
    speed: float


class EventItem(Camel):
    id: str
    ts: int  # epoch millis, matches JS Date.now()
    concept: str
    concept_color: str = Field(alias="conceptColor")
    severity: Severity
    kind: str
    description: str
    verdict: Verdict
    tier: Tier
    deep_analyzed: bool = Field(alias="deepAnalyzed")
    # None for scene-overview events (see pipeline.py's _queue_scene_analysis)
    # — a periodic whole-frame narrative isn't about any one tracked object.
    # Only set for events sourced from a specific track.
    track_id: int | None = Field(alias="trackId")
    telemetry: Telemetry
    # `image` is a base64 JPEG data URL — the analyzed frame for a scene
    # note, or a margin crop for a track-scoped event (NOT the full frame —
    # bounds memory); `box` is already in that image's OWN pixel
    # coordinates, not the source frame's, so nothing downstream needs
    # offset math. Empty/None when there's no object box (scene notes) or
    # the crop failed — callers must handle the missing case rather than
    # assume it's always present.
    image: str | None = None
    box: list[int] = Field(default_factory=list)


class Concept(Camel):
    """Matches frontend `Concept` (the watch-list entry)."""

    id: str
    label: str
    color: str
    priority: Priority
    exemplars: int
    enabled: bool


class ChatMessage(Camel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    grounded: str | None = None
