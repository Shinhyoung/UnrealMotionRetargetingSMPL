"""UDP unicast sender.

Fire-and-forget. Failures are logged and swallowed — we prefer to keep the
realtime loop alive over crashing on transient network errors (CLAUDE.md §5).
"""
from __future__ import annotations

import socket


class UDPSender:
    def __init__(self, host: str, port: int):
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Non-blocking is not necessary; sendto on UDP does not block meaningfully.
        self._closed = False

    def send(self, data: bytes) -> bool:
        if self._closed:
            return False
        try:
            self.sock.sendto(data, self.addr)
            return True
        except OSError as e:
            # Route/host errors etc. shouldn't kill the pipeline.
            print(f"[warn] UDP send failed: {e}")
            return False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self.sock.close()
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
