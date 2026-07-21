"""2D debug overlay on the source frame.

Rules (CLAUDE.md §8):
  * Must be flag-controlled and OFF by default.
  * Failures here MUST NOT kill the realtime pipeline — the caller catches exceptions.
  * No dependency from production modules on this file.
  * FPS shown here is measured with debug ON — do not trust it for perf tuning.

Mesh overlay uses **SAT-HMR's own ``vis_meshes_img``** utility when available
(CLAUDE.md §8.3 — reuse first). Requires ``PoseDetection.debug_verts`` /
``debug_intrinsics`` populated by the SATHMRDetector wrapper with
``keep_debug_output=True``.
"""
from __future__ import annotations

import time
from typing import Optional, Sequence

import cv2
import numpy as np


def _color_for_id(person_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color from a person_id."""
    rng = np.random.default_rng(int(person_id) * 9973 + 7)
    return tuple(int(v) for v in rng.integers(64, 256, size=3))


class DebugOverlay:
    def __init__(self, settings, sat_hmr_root: Optional[str] = None,
                 smpl_faces: Optional[np.ndarray] = None):
        self.settings = settings
        self.window = settings.window_name
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

        # Mesh overlay: reuse SAT-HMR's vis_meshes_img if requested. This pulls
        # ``pyrender`` at first render call — failures are swallowed and mesh is
        # disabled per §8.2.
        self._vis_meshes_img = None
        self._smpl_faces = smpl_faces
        if settings.show_mesh:
            if sat_hmr_root and smpl_faces is not None:
                self._try_load_vis_meshes(sat_hmr_root)
            else:
                print("[warn] show_mesh=True but sat_hmr_root/smpl_faces not provided; "
                      "mesh overlay disabled.")

        self._last_time: float | None = None
        self._fps_ema: float = 0.0
        self.should_exit: bool = False   # set to True when user presses q or ESC

    def _try_load_vis_meshes(self, sat_hmr_root: str) -> None:
        import contextlib
        import os
        import sys
        try:
            # SAT-HMR's utils/visualization.py forces PYOPENGL_PLATFORM=egl before
            # importing pyrender, which fails on Windows (no EGL). Pre-load pyrender
            # ourselves in a Windows-compatible way and stash it in sys.modules so
            # SAT-HMR's bare `import pyrender` finds the cached module.
            os.environ.pop("PYOPENGL_PLATFORM", None)
            import pyrender  # noqa: F401 — populates sys.modules
            if sat_hmr_root not in sys.path:
                sys.path.insert(0, sat_hmr_root)
            with contextlib.chdir(sat_hmr_root):
                from utils.visualization import vis_meshes_img
            self._vis_meshes_img = vis_meshes_img
            print("[debug] mesh renderer via SAT-HMR vis_meshes_img loaded.")
        except Exception as e:
            print(f"[warn] vis_meshes_img unavailable ({e}); mesh overlay disabled.")
            self._vis_meshes_img = None

    def render(self, frame_bgr: np.ndarray, detections: Sequence, frame_id: int) -> None:
        canvas = frame_bgr.copy()

        # Mesh overlay first (bottom layer), then bbox + HUD on top.
        if self.settings.show_mesh and self._vis_meshes_img is not None and detections:
            canvas = self._draw_meshes(canvas, detections)

        for det in detections:
            pid = getattr(det, "person_id", None)
            color = _color_for_id(pid if pid is not None else 0)

            bbox = getattr(det, "bbox", None)
            if self.settings.show_bbox and bbox is not None:
                x0, y0, x1, y1 = [int(v) for v in bbox]
                cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
                label = f"ID {pid}" if pid is not None else "ID ?"
                y_text = max(y0 - 8, 20)
                cv2.putText(
                    canvas, label, (x0, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

        if self.settings.show_hud:
            self._draw_hud(canvas, frame_id, len(detections))

        cv2.imshow(self.window, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 27 = ESC
            self.should_exit = True

    def _draw_meshes(self, canvas: np.ndarray, detections: Sequence) -> np.ndarray:
        """Reuse SAT-HMR's mesh visualizer. Silent no-op if a det lacks debug fields."""
        # Collect per-detection verts (in OpenCV camera coord) and a shared intrinsics.
        verts_list = []
        colors = []
        intrinsics = None
        resize_rate = None
        for det in detections:
            v = getattr(det, "debug_verts", None)
            k = getattr(det, "debug_intrinsics", None)
            r = getattr(det, "debug_resize_rate", None)
            if v is None or k is None or r is None:
                continue
            verts_list.append(v)
            colors.append(_rgb_from_id(getattr(det, "person_id", None) or 0))
            intrinsics = k
            resize_rate = r
        if not verts_list:
            return canvas

        # SAT-HMR's vis_meshes_img expects the input-size padded frame layout — the
        # square canvas its intrinsics were trained against. We resize the source
        # to the valid-area size, let vis_meshes_img pad it internally, then crop
        # the valid region out of the returned square image before scaling back.
        # NOTE: SAT-HMR's utils/visualization.py hardcodes PYOPENGL_PLATFORM=egl at
        # import time — Windows has no EGL. Reset before each render so pyrender's
        # OffscreenRenderer picks the pyglet-based Windows path instead.
        import os as _os
        _os.environ.pop("PYOPENGL_PLATFORM", None)
        try:
            H_orig, W_orig = canvas.shape[:2]
            new_w = max(1, int(round(W_orig * resize_rate)))
            new_h = max(1, int(round(H_orig * resize_rate)))
            resized = cv2.resize(canvas, (new_w, new_h))
            meshed_square = self._vis_meshes_img(
                img=resized,
                verts=np.stack(verts_list, axis=0),
                smpl_faces=self._smpl_faces,
                cam_intrinsics=intrinsics,
                colors=colors,
                padding=True,   # let SAT-HMR pad to input_size square (its trained layout)
            )
            # Crop the valid area (top-left of the padded square) and scale back.
            valid = meshed_square[:new_h, :new_w]
            return cv2.resize(valid, (W_orig, H_orig))
        except Exception as e:
            print(f"[warn] mesh overlay render failed: {e}")
            return canvas

    def _draw_hud(self, canvas: np.ndarray, frame_id: int, n_people: int) -> None:
        now = time.time()
        if self._last_time is not None:
            dt = now - self._last_time
            if dt > 0:
                inst = 1.0 / dt
                self._fps_ema = 0.9 * self._fps_ema + 0.1 * inst if self._fps_ema else inst
        self._last_time = now

        text_lines = [
            f"frame={frame_id}  people={n_people}  fps={self._fps_ema:5.1f} (debug)",
            "DEBUG OVERLAY ON — do not trust for perf measurement",
        ]
        y = 26
        for line in text_lines:
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y += 22

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.window)
        except cv2.error:
            pass


def _rgb_from_id(pid: int) -> tuple[float, float, float]:
    """Deterministic 0..1 RGB tuple for SAT-HMR's ``vis_meshes_img`` colors arg."""
    b, g, r = _color_for_id(pid)
    return (r / 255.0, g / 255.0, b / 255.0)
