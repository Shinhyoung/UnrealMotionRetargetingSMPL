"""Tiny UDP dump: listen on host:port and print packet stats.

Used to sanity-check that main.py --mock is really sending. Not part of the
pipeline. Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

# make /python importable when this file is run standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from network.packet import parse_packet


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9527)
    p.add_argument("--max-packets", type=int, default=0,
                   help="Stop after N packets (0 = run forever).")
    p.add_argument("--timeout", type=float, default=5.0,
                   help="Socket recv timeout seconds; 0 = blocking.")
    args = p.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    if args.timeout > 0:
        sock.settimeout(args.timeout)
    print(f"[udp_dump] listening on {args.host}:{args.port}")

    n = 0
    t0 = time.time()
    try:
        while True:
            try:
                data, _addr = sock.recvfrom(65535)
            except socket.timeout:
                print("[udp_dump] recv timeout, exiting")
                break
            try:
                frame_id, persons = parse_packet(data)
            except ValueError as e:
                print(f"[udp_dump] malformed packet ({len(data)} bytes): {e}")
                continue
            ids = [p[0] for p in persons]
            print(f"[udp_dump] frame={frame_id:6d}  people={len(persons)}  ids={ids}  bytes={len(data)}")
            n += 1
            if args.max_packets and n >= args.max_packets:
                break
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        elapsed = time.time() - t0
        pps = n / elapsed if elapsed > 0 else 0
        print(f"[udp_dump] received {n} packets in {elapsed:.2f}s ({pps:.1f} pps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
