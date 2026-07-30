"""Frame-to-frame identity: a small greedy IOU tracker.

`perception.Perception.detect_track()` calls SAM 3's single-image API fresh
per frame — it has no memory, so every detection gets a brand-new track_id
(see model_servers/perception/interface.py's own docstring on this). Without
something stitching detections together across frames, every object would
look like a "first appearance" on every single frame, which is useless for
Tier-2 event logic and for drawing a trail in the UI.

This is a deliberately simple stand-in, not a replacement for SAM 3's real
video-predictor API (`build_sam3_video_predictor` + a persistent
`inference_state` — see interface.py). Swap to that if/when per-stream state
management is wired up; until then, greedy IOU matching on same-concept boxes
between consecutive frames is enough to make events and trails meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    concept: str
    box: tuple[int, int, int, int]  # last-seen pixel box, for IOU matching next frame
    norm_x: float
    norm_y: float
    score: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    trail: list[tuple[float, float]] = field(default_factory=list)
    missed_frames: int = 0
    is_new: bool = True  # True only on the frame the track was created


class Tracker:
    """Call `update()` once per frame with that frame's raw detections."""

    def __init__(self, iou_threshold: float = 0.3, max_missed_frames: int = 10) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(
        self,
        detections: list[tuple[str, tuple[int, int, int, int], float]],
        frame_w: int,
        frame_h: int,
    ) -> list[Track]:
        """`detections` is (concept, pixel_box, score) tuples for one frame.

        Returns the current set of live tracks (newly created, matched, or
        still-coasting-within-max_missed_frames), each with `.is_new` set
        correctly for that call — callers use `is_new` to fire "first
        appearance" events exactly once per track.
        """
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self._tracks.keys())
        matches: list[tuple[int, int]] = []  # (track_id, detection_index)

        # greedy best-IOU-first matching, same concept only
        pairs = []
        for tid in unmatched_tracks:
            for di in unmatched_dets:
                concept, box, _score = detections[di]
                if concept != self._tracks[tid].concept:
                    continue
                iou = _iou(self._tracks[tid].box, box)
                if iou >= self.iou_threshold:
                    pairs.append((iou, tid, di))
        pairs.sort(reverse=True)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for iou, tid, di in pairs:
            if tid in matched_tracks or di in matched_dets:
                continue
            matches.append((tid, di))
            matched_tracks.add(tid)
            matched_dets.add(di)

        unmatched_dets = [d for d in unmatched_dets if d not in matched_dets]
        unmatched_tracks = [t for t in unmatched_tracks if t not in matched_tracks]

        for tid, di in matches:
            concept, box, score = detections[di]
            track = self._tracks[tid]
            x1, y1, x2, y2 = box
            nx, ny = (x1 + x2) / 2 / frame_w, (y1 + y2) / 2 / frame_h
            track.vx, track.vy = nx - track.norm_x, ny - track.norm_y
            track.norm_x, track.norm_y = nx, ny
            track.box = box
            track.score = score
            track.missed_frames = 0
            track.is_new = False
            track.trail.append((nx, ny))
            if len(track.trail) > 40:
                track.trail.pop(0)

        for di in unmatched_dets:
            concept, box, score = detections[di]
            x1, y1, x2, y2 = box
            nx, ny = (x1 + x2) / 2 / frame_w, (y1 + y2) / 2 / frame_h
            track = Track(track_id=self._next_id, concept=concept, box=box, norm_x=nx, norm_y=ny, score=score)
            track.trail.append((nx, ny))
            self._tracks[track.track_id] = track
            self._next_id += 1

        for tid in unmatched_tracks:
            track = self._tracks[tid]
            track.missed_frames += 1
            track.is_new = False
            if track.missed_frames > self.max_missed_frames:
                del self._tracks[tid]

        return list(self._tracks.values())
