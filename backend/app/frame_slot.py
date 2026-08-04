"""Single-slot frame handoff between the ingest thread and its consumers.

The pipeline must never build a frame backlog: a queue of stale video frames
is worse than useless, because every frame it holds is one the consumer will
render or analyze *late*. So this is a rendezvous, not a queue — there is
exactly one slot, and a new frame simply replaces whatever was in it.

It is *demand-signalled*: a consumer marks that it wants a frame, and the
producer only bothers converting/offering one when someone is actually
waiting. That is what lets `ingest` skip the expensive `to_ndarray()` colour
conversion entirely for frames nobody wants (see pipeline.py).

Why not `queue.Queue(maxsize=1)` with drop-on-full: the usual idiom is

    try: q.put_nowait(x)
    except queue.Full:
        try: q.get_nowait()
        except queue.Empty: pass
        q.put_nowait(x)          # <-- can raise Full *again*

which is racy — between the producer's get and its put, a consumer can refill
the slot and the second put_nowait raises, losing the frame and raising on the
ingest thread. A Condition gives take-and-clear atomicity instead.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameItem:
    """One decoded video frame, shared read-only across threads.

    `rgb` is safe to share without copying or locking: PyAV's `to_ndarray()`
    allocates a fresh buffer per call (swscale writes into new memory rather
    than a recycled decoder buffer), and every consumer in this codebase only
    reads it — cv2.imencode/cvtColor allocate their own outputs, PIL's
    fromarray copies, and crop slicing produces read-only views. The array is
    marked non-writeable on construction so that contract is enforced rather
    than merely documented.
    """

    rgb: np.ndarray
    width: int
    height: int
    seq: int
    captured: float  # time.monotonic() at decode, NOT at handoff


class DemandSlot:
    """One-frame rendezvous with demand signalling. Thread-safe."""

    __slots__ = ("_cv", "_item", "_wanted", "_closed", "offered", "refused")

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._item: FrameItem | None = None
        self._wanted = False
        self._closed = False
        # approximate counters for the perf log — not locked, don't build logic on them
        self.offered = 0
        self.refused = 0

    def wanted(self) -> bool:
        """Lock-free hint: is a consumer currently waiting?

        Only a hint — `offer()` re-validates under the lock. A stale True costs
        at most one wasted colour conversion; it can never cause incorrectness.
        """
        return self._wanted and not self._closed

    def offer(self, item: FrameItem) -> bool:
        """Producer side. Never blocks. Returns True if a consumer took delivery."""
        with self._cv:
            if not self._wanted or self._closed:
                self.refused += 1
                return False
            self._item = item
            self._wanted = False
            self.offered += 1
            self._cv.notify_all()
            return True

    def take(self, timeout: float = 0.25) -> FrameItem | None:
        """Consumer side. Registers demand, waits, then takes *and clears*.

        Take-and-clear under the lock is what guarantees a consumer that
        outpaces the producer can never process the same frame twice: once
        this returns, the slot is empty again.

        The timeout is a liveness backstop so the caller can re-check its stop
        event if the stream goes silent — not a polling interval.
        """
        with self._cv:
            self._wanted = True
            if self._item is None and not self._closed:
                self._cv.wait(timeout)
            item, self._item = self._item, None
            return item

    def close(self) -> None:
        """Wake every waiter so threads can observe their stop event and exit."""
        with self._cv:
            self._closed = True
            self._wanted = False
            self._item = None
            self._cv.notify_all()
