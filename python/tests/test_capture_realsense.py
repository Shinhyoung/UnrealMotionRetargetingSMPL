"""RealSense capture adapter tests.

The real hardware is not always attached (CI, dev laptop). These tests use a
faked ``pyrealsense2`` module — inject it into ``sys.modules`` before
``RealSenseCapture`` imports it, so we exercise the adapter's control flow
without needing a device.
"""
import sys
import types

import numpy as np
import pytest


def _install_fake_pyrealsense2(monkeypatch, *, fail_on_start: bool = False,
                                good_frames: int = 3):
    """Inject a stub pyrealsense2 module into sys.modules."""
    fake = types.ModuleType("pyrealsense2")

    class Stream:
        color = "color"
        depth = "depth"

    class Format:
        bgr8 = "bgr8"
        z16 = "z16"

    class CameraInfo:
        name = "name"
        serial_number = "serial"

    fake.stream = Stream
    fake.format = Format
    fake.camera_info = CameraInfo

    class _Frame:
        def __init__(self, data):
            self._data = data
        def __bool__(self):
            return True
        def get_data(self):
            return self._data

    class _Frames:
        def __init__(self, color, depth=None):
            self._color = color
            self._depth = depth
        def get_color_frame(self):
            return self._color
        def get_depth_frame(self):
            return self._depth

    class _Profile:
        def get_device(self):
            class Dev:
                def get_info(self, key):
                    return "FakeDev" if key == "name" else "FAKESERIAL"
            return Dev()

    class _Pipeline:
        def __init__(self):
            self._n = 0
        def start(self, cfg):
            if fail_on_start:
                raise RuntimeError("no device (fake)")
            return _Profile()
        def wait_for_frames(self, timeout_ms=1000):
            self._n += 1
            if self._n > good_frames:
                raise RuntimeError("timeout")
            color = _Frame(np.full((720, 1280, 3), self._n, dtype=np.uint8))
            depth = _Frame(np.full((720, 1280), self._n * 10, dtype=np.uint16))
            return _Frames(color, depth)
        def stop(self):
            pass

    class _Config:
        def enable_stream(self, *a, **k): pass
        def enable_device(self, *a, **k): pass

    class _Align:
        def __init__(self, target): pass
        def process(self, frames): return frames

    fake.pipeline = _Pipeline
    fake.config = _Config
    fake.align = _Align

    monkeypatch.setitem(sys.modules, "pyrealsense2", fake)


def test_realsense_reads_color_frames(monkeypatch):
    _install_fake_pyrealsense2(monkeypatch, good_frames=2)
    from capture.realsense import RealSenseCapture
    cap = RealSenseCapture(width=1280, height=720, fps=30)
    ok, f = cap.read()
    assert ok
    assert f.shape == (720, 1280, 3)
    ok, _ = cap.read()
    assert ok
    ok, _ = cap.read()
    assert not ok  # 3rd frame throws timeout in stub
    cap.release()


def test_realsense_release_is_idempotent(monkeypatch):
    _install_fake_pyrealsense2(monkeypatch)
    from capture.realsense import RealSenseCapture
    cap = RealSenseCapture()
    cap.release()
    cap.release()  # must not raise


def test_realsense_start_failure_raises(monkeypatch):
    _install_fake_pyrealsense2(monkeypatch, fail_on_start=True)
    from capture.realsense import RealSenseCapture, RealSenseCaptureError
    with pytest.raises(RealSenseCaptureError):
        RealSenseCapture()


def test_realsense_depth_requires_enable(monkeypatch):
    _install_fake_pyrealsense2(monkeypatch)
    from capture.realsense import RealSenseCapture, RealSenseCaptureError
    cap = RealSenseCapture(enable_depth=False)
    with pytest.raises(RealSenseCaptureError):
        cap.read_with_depth()
    cap.release()


def test_realsense_depth_read(monkeypatch):
    _install_fake_pyrealsense2(monkeypatch)
    from capture.realsense import RealSenseCapture
    cap = RealSenseCapture(enable_depth=True)
    ok, color, depth = cap.read_with_depth()
    assert ok
    assert color.shape == (720, 1280, 3)
    assert depth.shape == (720, 1280)
    cap.release()
