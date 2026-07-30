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


@dataclass
class ConceptRecord:
    """Everything the backend needs for one watch-list entry — the public
    `Concept` fields the frontend expects, plus what detection actually runs
    on (text_prompt for SAM 3, reference_embeddings for the DINOv3 match gate;
    see model_servers/perception/README.md's "Context authoring" section)."""

    id: str
    label: str
    color: str
    priority: str
    enabled: bool
    text_prompt: str
    reference_embeddings: list[list[float]] = field(default_factory=list)

    def to_public(self) -> Concept:
        return Concept(
            id=self.id,
            label=self.label,
            color=self.color,
            priority=self.priority,  # type: ignore[arg-type]
            exemplars=len(self.reference_embeddings),
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

    def add_exemplar_embedding(self, concept_id: str, embedding: list[float]) -> ConceptRecord | None:
        with self._lock:
            record = self._records.get(concept_id)
            if record is not None:
                record.reference_embeddings.append(embedding)
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
