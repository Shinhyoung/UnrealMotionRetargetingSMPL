"""Unit tests for ThreadedCapture.

Uses a synthetic ``cv2.VideoCapture``-shaped source that produces frames at a
controllable rate. Verifies:
  * newest-frame semantics (stale frames get dropped)
  * blocking read until first frame lands
  * end-of-source signalling
  * release joins the reader thread
"""
import threading
import time

import numpy as np
import pytest

from capture.threaded import ThreadedCapture


class _RateLimitedSource:
    def __init__(self, interval_s: float, n_frames: int = 100):
        self._interval = interval_s
        self._n = n_frames
        self._i = 0
        self._last_read_ts = 0.0
        self._lock = threading.Lock()

    def read(self):
        # Simulate camera FPS: block until enough time has passed since last frame.
        now = time.time()
        wait = self._interval - (now - self._last_read_ts)
        if wait > 0:
            time.sleep(wait)
        with self._lock:
            if self._i >= self._n:
                return False, np.empty((0, 0, 3), dtype=np.uint8)
            frame = np.full((4, 4, 3), self._i, dtype=np.uint8)
            self._i += 1
            self._last_read_ts = time.time()
            return True, frame

    def release(self):
        pass


def test_read_returns_latest_frame_and_drops_stale():
    src = _RateLimitedSource(interval_s=0.01, n_frames=200)
    tc = ThreadedCapture(src, poll_interval_s=0.001)
    try:
        # Let the reader buffer up several frames.
        time.sleep(0.05)
        ok, frame = tc.read()
        assert ok
        # The frame we got should be the latest; the reader kept producing so
        # dropped_since_last_read is measured RELATIVE to the read we just did.
        # Immediately reading again shouldn't return the same one — but might
        # briefly if the source hasn't ticked yet.
        prev = int(frame[0, 0, 0])
        # After a delay, next read must return a strictly newer frame.
        time.sleep(0.05)
        ok2, frame2 = tc.read()
        assert ok2
        assert int(frame2[0, 0, 0]) > prev
    finally:
        tc.release()


def test_release_joins_thread():
    src = _RateLimitedSource(interval_s=0.005, n_frames=1000)
    tc = ThreadedCapture(src)
    time.sleep(0.02)
    tc.release()
    assert not tc._thread.is_alive()


def test_end_of_source_signals_read_failure():
    src = _RateLimitedSource(interval_s=0.001, n_frames=3)
    tc = ThreadedCapture(src)
    try:
        # Drain until source exhausted and reader signals.
        deadline = time.time() + 2.0
        got_none = False
        while time.time() < deadline:
            ok, _ = tc.read()
            if not ok:
                got_none = True
                break
        assert got_none, "reader should signal ok=False after source exhausts"
    finally:
        tc.release()


def test_dropped_since_last_read_counter():
    src = _RateLimitedSource(interval_s=0.005, n_frames=200)
    tc = ThreadedCapture(src)
    try:
        # Prime the reader.
        ok, _ = tc.read()
        assert ok
        # Sleep longer than one interval so extra frames pile up.
        time.sleep(0.05)
        dropped = tc.dropped_since_last_read()
        assert dropped >= 3, f"expected >=3 dropped frames, got {dropped}"
        ok2, _ = tc.read()
        assert ok2
        # After consuming, dropped counter should reset.
        dropped_after = tc.dropped_since_last_read()
        assert dropped_after == 0 or dropped_after < dropped
    finally:
        tc.release()
