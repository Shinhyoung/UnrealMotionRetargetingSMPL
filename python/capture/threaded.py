"""Threaded frame reader that decouples capture from inference.

Rationale (CLAUDE.md 마일스톤 7): when inference (~74ms) is slower than the
camera's frame interval (~33ms @ 30 FPS), the source's internal FIFO fills up
and every returned frame is *stale* by the time we get it. Displayed latency
and effective FPS both suffer.

This wrapper spawns a daemon thread that continuously pulls from the source
and keeps only the **latest** frame. ``read()`` returns whatever is newest —
older frames are silently dropped. This lifts displayed FPS to camera rate
(~30) even when detection is slower, at the cost of detection cadence.

Semantics:
  * ``read()`` blocks until the first frame is available, then returns
    immediately with the most recent one.
  * Downstream tracker+smoother keeps person_ids stable across dropped frames.
  * ``release()`` joins the reader thread cleanly.
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import numpy as np


class ThreadedCapture:
    """Wrap any ``cv2.VideoCapture``-like object (``read`` + ``release``)."""

    def __init__(self, wrapped, poll_interval_s: float = 0.001, with_depth: bool = False):
        self._wrapped = wrapped
        self._latest: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_seq: int = 0
        self._consumed_seq: int = -1
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._source_exhausted = threading.Event()
        self._poll_interval_s = float(poll_interval_s)
        self._with_depth = bool(with_depth) and hasattr(wrapped, "read_with_depth")

        self._thread = threading.Thread(
            target=self._loop, name="ThreadedCapture", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        # RealSense (and similar hardware sources) can drop the first few
        # ``wait_for_frames`` calls while streaming spins up. Treat isolated
        # failures as warm-up noise; only give up after a sustained streak.
        _MAX_CONSECUTIVE_FAILURES = 60
        failures = 0
        while not self._stop.is_set():
            try:
                if self._with_depth:
                    ok, frame, depth = self._wrapped.read_with_depth()
                else:
                    ok, frame = self._wrapped.read()
                    depth = None
            except Exception as e:  # pragma: no cover — source-specific
                print(f"[threaded_capture] read raised: {e}")
                self._source_exhausted.set()
                return
            if not ok:
                failures += 1
                if failures > _MAX_CONSECUTIVE_FAILURES:
                    self._source_exhausted.set()
                    return
                continue
            failures = 0
            with self._lock:
                self._latest = frame
                self._latest_depth = depth
                self._latest_seq += 1

    def read(self) -> Tuple[bool, np.ndarray]:
        """Return (ok, frame). Blocks until a frame is available; drops stale ones."""
        while not self._stop.is_set():
            with self._lock:
                if self._latest is not None and self._latest_seq != self._consumed_seq:
                    self._consumed_seq = self._latest_seq
                    return True, self._latest
            if self._source_exhausted.is_set():
                return False, np.empty((0, 0, 3), dtype=np.uint8)
            time.sleep(self._poll_interval_s)
        return False, np.empty((0, 0, 3), dtype=np.uint8)

    def read_with_depth(self) -> Tuple[bool, np.ndarray, Optional[np.ndarray]]:
        """Return (ok, color, depth). Depth is None if not enabled."""
        while not self._stop.is_set():
            with self._lock:
                if self._latest is not None and self._latest_seq != self._consumed_seq:
                    self._consumed_seq = self._latest_seq
                    return True, self._latest, self._latest_depth
            if self._source_exhausted.is_set():
                return False, np.empty((0, 0, 3), dtype=np.uint8), None
            time.sleep(self._poll_interval_s)
        return False, np.empty((0, 0, 3), dtype=np.uint8), None

    def dropped_since_last_read(self) -> int:
        """How many frames the reader thread produced since the last ``read()``.

        Useful for logging staleness. 0 means we picked up the same frame the
        capture just delivered; N>0 means N-1 frames were skipped as stale.
        """
        with self._lock:
            return max(0, self._latest_seq - self._consumed_seq - 1)

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self._wrapped.release()
        except Exception:  # pragma: no cover
            pass
