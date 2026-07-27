"""In-memory perception: SAM 3 (+ DINOv3) loaded once, called in-process.

This is the real-time hot path. It is imported by the backend, NOT served over
HTTP on the live path (see ../README.md §1.2). Two backends hide behind one
interface so SAM 3 and a SAM 2.1 + GroundingDINO fallback are interchangeable.

The heavy model loading is intentionally left as TODOs — this file pins the
contract the backend codes against so the pipeline can be built in parallel with
the model integration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Detection:
    track_id: int
    concept: str
    box: tuple[int, int, int, int]      # x1, y1, x2, y2
    mask_rle: str                       # COCO RLE
    score: float
    embedding: list[float] | None = None


@dataclass
class Concept:
    label: str
    priority: str = "watch"             # alert | watch | ignore
    text_prompt: str | None = None
    exemplar_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)


class Perception:
    """Loads SAM 3 + DINOv3 onto the GPU once; exposes in-process inference."""

    def __init__(self, backend: str, sam_weights: str, dinov3_weights: str,
                 device: str = "cuda") -> None:
        self.backend = backend
        self.device = device
        self.sam_weights = sam_weights
        self.dinov3_weights = dinov3_weights
        self._load()

    @classmethod
    def from_env(cls) -> "Perception":
        return cls(
            backend=os.environ.get("PERCEPTION_BACKEND", "sam3"),
            sam_weights=os.environ.get("SAM_WEIGHTS", "/models/sam3"),
            dinov3_weights=os.environ.get("DINOV3_WEIGHTS", "/models/dinov3"),
        )

    def _load(self) -> None:
        if self.backend == "sam3":
            ...  # TODO: load SAM 3 (concept-promptable segmentation + tracker)
        elif self.backend == "sam2_gdino":
            ...  # TODO: load SAM 2.1 + GroundingDINO/YOLO-World
        else:
            raise ValueError(f"unknown PERCEPTION_BACKEND: {self.backend}")
        ...      # TODO: load DINOv3 from self.dinov3_weights

    def detect_track(self, frame, concepts: list[Concept]) -> list[Detection]:
        """Tier-1: detect + segment + track every concept instance in `frame`.

        `frame` is a decoded HxWx3 array (or GPU tensor) handed straight from the
        pipeline — no encode/decode round-trip.
        """
        raise NotImplementedError

    def embed(self, frame, detections: list[Detection]) -> list[Detection]:
        """Attach a DINOv3 embedding to each detection's crop (in-place)."""
        raise NotImplementedError
