"""Simple tkinter launcher for the motion-retargeting pipeline.

Pick a detector (SAT-HMR or SMPLest-X), optionally enable the debug overlay,
click Run. Spawns ``main.py`` in a background subprocess with the appropriate
Python interpreter for each detector's env.

Run:
    python launcher.py
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext, ttk


# --- Paths — edit if your setup differs ------------------------------------
PIPELINE_DIR = Path(__file__).resolve().parent
BASE_PY = sys.executable                       # current interpreter (for SAT-HMR)
SMPLESTX_PY = r"C:\Users\ADMIN\anaconda3\envs\smplestx\python.exe"
SMPLESTX_ROOT = r"C:\0.shinhyoung\Project\1.Retargeting\SMPLest-X"
SMPLESTX_CKPT = "smplest_x_h"
PELVIS_REST = "../pelvis_rest.npy"

COMMON_ARGS = [
    "--source", "realsense",
    "--conf-thresh", "0.3",
    "--smpl-native",
    "--pelvis-rest", PELVIS_REST,
    "--depth-root-lock",
    "--swap-lr",
]


def build_command(detector: str, debug: bool, smooth_alpha: float, root_alpha: float,
                  max_persons: int):
    if detector == "smplest-x":
        py = SMPLESTX_PY
        extra = [
            "--detector", "smplest-x",
            "--smplest-x-root", SMPLESTX_ROOT,
            "--smplest-x-ckpt", SMPLESTX_CKPT,
            "--joint-format", "smplx",
        ]
    else:
        py = BASE_PY
        extra = []
    cmd = [py, "-u", "main.py"] + COMMON_ARGS + [
        "--smooth-alpha", str(smooth_alpha),
        "--smooth-root-alpha", str(root_alpha),
        "--max-persons", str(max_persons),
    ] + extra
    if debug:
        cmd += ["--debug", "--show-mesh"]
    return cmd, py


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.proc: subprocess.Popen | None = None
        self.log_queue: queue.Queue = queue.Queue()
        self.reader_thread: threading.Thread | None = None
        root.title("Motion Retargeting Launcher")
        root.geometry("640x520")

        # --- Detector ---
        det_frame = ttk.LabelFrame(root, text="Detector", padding=10)
        det_frame.pack(fill="x", padx=10, pady=6)
        self.detector = tk.StringVar(value="sat-hmr")
        ttk.Radiobutton(det_frame, text="SAT-HMR (SMPL 24 joints, body only, faster)",
                        variable=self.detector, value="sat-hmr",
                        command=self._sync_detector).pack(anchor="w")
        ttk.Radiobutton(det_frame, text="SMPLest-X (SMPL-X 55 joints, body + hands, slower)",
                        variable=self.detector, value="smplest-x",
                        command=self._sync_detector).pack(anchor="w")

        # --- Persons ---
        pp_frame = ttk.LabelFrame(root, text="Multi-Person (SAT-HMR only, max 3)", padding=10)
        pp_frame.pack(fill="x", padx=10, pady=6)
        ttk.Label(pp_frame, text="Max persons:").pack(side="left")
        self.max_persons = tk.IntVar(value=1)
        self.persons_spin = ttk.Spinbox(pp_frame, from_=1, to=3, width=4,
                                        textvariable=self.max_persons)
        self.persons_spin.pack(side="left", padx=6)
        ttk.Label(pp_frame, text="(place N SMPLProceduralActor instances in UE, "
                                 "each with PersonId = 1..N)").pack(side="left", padx=(10, 0))

        # --- Debug overlay ---
        dbg_frame = ttk.LabelFrame(root, text="Options", padding=10)
        dbg_frame.pack(fill="x", padx=10, pady=6)
        self.debug = tk.BooleanVar(value=False)
        ttk.Checkbutton(dbg_frame, text="Debug overlay window (OpenCV mesh preview)",
                        variable=self.debug).pack(anchor="w")

        # --- Smoothing ---
        sm_frame = ttk.LabelFrame(root, text="Smoothing (0.05 = heavy, 1.0 = none)", padding=10)
        sm_frame.pack(fill="x", padx=10, pady=6)
        ttk.Label(sm_frame, text="Pose α:").grid(row=0, column=0, sticky="w")
        self.smooth_alpha = tk.DoubleVar(value=0.5)
        ttk.Spinbox(sm_frame, from_=0.05, to=1.0, increment=0.05, width=6,
                    textvariable=self.smooth_alpha).grid(row=0, column=1, padx=6)
        ttk.Label(sm_frame, text="Root α:").grid(row=0, column=2, sticky="w", padx=(20, 0))
        self.root_alpha = tk.DoubleVar(value=0.3)
        ttk.Spinbox(sm_frame, from_=0.05, to=1.0, increment=0.05, width=6,
                    textvariable=self.root_alpha).grid(row=0, column=3, padx=6)

        # --- Buttons ---
        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill="x", padx=10, pady=10)
        self.run_btn = ttk.Button(btn_frame, text="실행 (Run)", command=self.on_run)
        self.run_btn.pack(side="left", padx=(0, 6))
        self.stop_btn = ttk.Button(btn_frame, text="정지 (Stop)", command=self.on_stop, state="disabled")
        self.stop_btn.pack(side="left")

        # --- Log output ---
        log_frame = ttk.LabelFrame(root, text="Log", padding=6)
        log_frame.pack(fill="both", expand=True, padx=10, pady=6)
        self.log = scrolledtext.ScrolledText(log_frame, wrap="none", height=12)
        self.log.pack(fill="both", expand=True)

        # --- Status bar ---
        self.status = ttk.Label(root, text="Ready", anchor="w", relief="sunken")
        self.status.pack(fill="x", side="bottom")

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._sync_detector()

    def _sync_detector(self):
        # SMPLest-X is single-person (crop-based per detection). Force 1 and disable spinbox.
        if self.detector.get() == "smplest-x":
            self.max_persons.set(1)
            self.persons_spin.config(state="disabled")
        else:
            self.persons_spin.config(state="normal")

    def _log(self, msg):
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    def on_run(self):
        if self.proc and self.proc.poll() is None:
            self._log("[launcher] pipeline already running")
            return
        cmd, py = build_command(
            self.detector.get(),
            self.debug.get(),
            float(self.smooth_alpha.get()),
            float(self.root_alpha.get()),
            int(self.max_persons.get()),
        )
        if not os.path.isfile(py):
            self._log(f"[error] Python not found: {py}")
            self.status.config(text="Python interpreter missing")
            return
        self._log(f"[launcher] $ {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd, cwd=str(PIPELINE_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        self.status.config(text=f"Running (PID {self.proc.pid})")
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        # Background thread reads subprocess stdout → puts lines on queue.
        # UI polls the queue non-blockingly from the tk event loop.
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()
        self._drain_queue()

    def _reader_loop(self):
        """Runs in background thread. Reads stdout until proc exits."""
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for line in iter(proc.stdout.readline, ""):
            self.log_queue.put(line.rstrip())
        proc.stdout.close()
        # Sentinel so UI knows to check exit code.
        self.log_queue.put(None)

    def _drain_queue(self):
        """UI-thread poll of log queue (non-blocking)."""
        proc_done = False
        # Pull up to 200 lines per tick — bounded so UI stays responsive.
        for _ in range(200):
            try:
                item = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                proc_done = True
                break
            self._log(item)
        if proc_done or (self.proc and self.proc.poll() is not None):
            code = self.proc.returncode if self.proc else -1
            self._log(f"[launcher] pipeline exited (code {code})")
            self.status.config(text=f"Stopped (exit {code})")
            self.run_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.proc = None
            return
        # Schedule next drain.
        self.log.after(100, self._drain_queue)

    def on_stop(self):
        if not self.proc or self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self._log("[launcher] stopped")
        self.status.config(text="Stopped")
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.proc = None

    def _on_close(self):
        # Kill child process without blocking the UI thread. terminate() is
        # best-effort; then hard-kill after a short wait.
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            except Exception:
                pass
        self.root.quit()      # break mainloop
        self.root.destroy()   # tear down widgets


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
