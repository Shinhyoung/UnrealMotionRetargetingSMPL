"""Intel RealSense D455 color-stream capture adapter.

Exposes the same ``read() -> (ok, frame_bgr)`` / ``release()`` surface as
``cv2.VideoCapture`` so ``main.py`` can swap sources without branching.

The pipeline currently uses only the color stream (SAT-HMR is single RGB).
Depth is initialized but not returned by default — enable ``enable_depth=True``
if you want to consume it for e.g. root-position refinement later.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class RealSenseCaptureError(RuntimeError):
    """Raised when a RealSense device cannot be started."""


class RealSenseCapture:
    """cv2.VideoCapture-compatible reader for Intel RealSense (tested on D455)."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        enable_depth: bool = False,
        serial: Optional[str] = None,
    ):
        try:
            import pyrealsense2 as rs  # type: ignore
        except ImportError as e:
            raise RealSenseCaptureError(
                "pyrealsense2 not installed. `pip install pyrealsense2`."
            ) from e

        self._rs = rs
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.enable_depth = bool(enable_depth)
        self.serial = serial

        self._pipeline: Optional["rs.pipeline"] = None
        self._align: Optional["rs.align"] = None
        self._closed = False
        self._start()

    def _start(self) -> None:
        rs = self._rs
        cfg = rs.config()
        if self.serial:
            cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.enable_depth:
            cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        pipeline = rs.pipeline()
        try:
            profile = pipeline.start(cfg)
        except RuntimeError as e:
            raise RealSenseCaptureError(
                f"Failed to start RealSense pipeline "
                f"({self.width}x{self.height}@{self.fps}, depth={self.enable_depth}, "
                f"serial={self.serial!r}): {e}"
            ) from e

        self._pipeline = pipeline
        if self.enable_depth:
            self._align = rs.align(rs.stream.color)

        # Log connected device for traceability.
        try:
            dev = profile.get_device()
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            print(f"[realsense] started {name} (serial={serial}) "
                  f"{self.width}x{self.height}@{self.fps}fps depth={self.enable_depth}")
        except Exception:
            pass

    def read(self) -> Tuple[bool, np.ndarray]:
        """Return (ok, frame_bgr). Never raises — signals failure via ok=False."""
        if self._closed or self._pipeline is None:
            return False, np.empty((0, 0, 3), dtype=np.uint8)
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError:
            return False, np.empty((0, 0, 3), dtype=np.uint8)

        if self._align is not None:
            frames = self._align.process(frames)

        color = frames.get_color_frame()
        if not color:
            return False, np.empty((0, 0, 3), dtype=np.uint8)

        # asanyarray copies into an ndarray; format is bgr8 so no channel swap needed.
        img = np.asanyarray(color.get_data())
        return True, img

    def read_with_depth(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Return (ok, color_bgr, depth_uint16).

        Depth is in millimeters (D455 default). Requires ``enable_depth=True`` at construction.
        """
        if not self.enable_depth:
            raise RealSenseCaptureError("Depth stream was not enabled at construction.")
        if self._closed or self._pipeline is None:
            return False, np.empty((0, 0, 3), dtype=np.uint8), np.empty((0, 0), dtype=np.uint16)
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError:
            return False, np.empty((0, 0, 3), dtype=np.uint8), np.empty((0, 0), dtype=np.uint16)

        if self._align is not None:
            frames = self._align.process(frames)

        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            return False, np.empty((0, 0, 3), dtype=np.uint8), np.empty((0, 0), dtype=np.uint16)
        return True, np.asanyarray(color.get_data()), np.asanyarray(depth.get_data())

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except RuntimeError:
                pass
            self._pipeline = None
