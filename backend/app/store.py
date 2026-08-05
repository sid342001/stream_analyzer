"""In-memory concept/exemplar store.

There is no database yet (no Postgres/pgvector service exists anywhere in
this repo — see model_servers/README.md's data-model gap, noted but never
built). This store is intentionally just a dict behind a lock: it is lost on
restart. Swap it for a real repository backed by Postgres+pgvector when that
service exists; the call sites (app/main.py, app/pipeline.py) only touch the
methods below, so the storage swap should stay localized to this file.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field

from app.schemas import Concept

_DEFAULT_COLORS = ["#38bdf8", "#a78bfa", "#34d399", "#f87171", "#facc15", "#fb923c"]


@dataclass(frozen=True)
class Exemplar:
    """One drawn/uploaded reference for a concept — everything downstream
    needs: the DINOv3 embedding for the match gate, and the original
    (thumbnail-sized) image + box so the VLM can be shown "what this concept
    looks like" (see vlm_client.py) and so YOLOE-26 can bake a visual prompt
    from it (see perception/interface.py's detect_by_exemplar)."""

    image: str                     # base64 JPEG data URL, resized (~320px long side)
    box: list[int]                 # [x1,y1,x2,y2] in image's own pixel coords
    embedding: list[float]


@dataclass
class ConceptRecord:
    """Everything the backend needs for one watch-list entry — the public
    `Concept` fields the frontend expects, plus what detection actually runs
    on (text_prompt for SAM 3, exemplars for the DINOv3 match gate and (once
    enabled) YOLOE-26 visual-prompt recall; see
    model_servers/perception/README.md's "Context authoring" section)."""

    id: str
    label: str
    color: str
    priority: str
    enabled: bool
    text_prompt: str
    exemplars: list[Exemplar] = field(default_factory=list)

    def to_public(self) -> Concept:
        return Concept(
            id=self.id,
            label=self.label,
            color=self.color,
            priority=self.priority,  # type: ignore[arg-type]
            exemplars=len(self.exemplars),
            enabled=self.enabled,
        )


class ConceptStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, ConceptRecord] = {}
        self._color_cycle = itertools.cycle(_DEFAULT_COLORS)
        self._next_id = 1

    def seed(self, labels: list[str]) -> None:
        """Populate an initial watch-list so detection has something to run
        against before an operator has authored anything — e.g. from
        STREAM_URL / STARTUP_CONCEPTS config, not hardcoded here."""
        with self._lock:
            for label in labels:
                self._add_locked(label=label, priority="watch")

    def list(self) -> list[ConceptRecord]:
        with self._lock:
            return list(self._records.values())

    def enabled(self) -> list[ConceptRecord]:
        with self._lock:
            return [c for c in self._records.values() if c.enabled]

    def add(self, label: str, priority: str = "watch") -> ConceptRecord:
        with self._lock:
            return self._add_locked(label, priority)

    def _add_locked(self, label: str, priority: str) -> ConceptRecord:
        concept_id = f"c_{self._next_id}"
        self._next_id += 1
        record = ConceptRecord(
            id=concept_id,
            label=label,
            color=next(self._color_cycle),
            priority=priority,
            enabled=True,
            text_prompt=label,
        )
        self._records[concept_id] = record
        return record

    def get(self, concept_id: str) -> ConceptRecord | None:
        with self._lock:
            return self._records.get(concept_id)

    def get_by_label(self, label: str) -> ConceptRecord | None:
        """Used by /api/chat, which only has the event's concept *label*
        (from the frontend store), not its id — linear scan is fine at this
        scale, matches the rest of this store's simplicity."""
        with self._lock:
            for record in self._records.values():
                if record.label == label:
                    return record
            return None

    def add_exemplar(self, concept_id: str, image: str, box: list[int],
                     embedding: list[float]) -> ConceptRecord | None:
        with self._lock:
            record = self._records.get(concept_id)
            if record is not None:
                # Copy-on-write, not .append(): the detection thread iterates
                # this list while matching exemplars, and appending in place
                # mutates a list another thread is mid-read on. Rebinding means
                # readers always hold a consistent immutable snapshot.
                record.exemplars = record.exemplars + [Exemplar(image=image, box=box, embedding=embedding)]
            return record

    def set_enabled(self, concept_id: str, enabled: bool) -> ConceptRecord | None:
        with self._lock:
            record = self._records.get(concept_id)
            if record is not None:
                record.enabled = enabled
            return record

    def set_priority(self, concept_id: str, priority: str) -> ConceptRecord | None:
        with self._lock:
            record = self._records.get(concept_id)
            if record is not None:
                record.priority = priority
            return record


concepts = ConceptStore()
