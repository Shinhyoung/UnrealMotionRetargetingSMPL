"""UDP packet (de)serialization.

Wire format is fixed by ``docs/protocol.md``. Little-endian throughout.
"""
from __future__ import annotations

import struct
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from config.bone_mapping import NUM_SMPL_JOINTS

_HEADER_FMT = "<IB"                                # frame_id (uint32) + person_count (uint8)
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)        # 5 bytes
_PERSON_ID_FMT = "<H"                              # uint16
_PERSON_ID_SIZE = struct.calcsize(_PERSON_ID_FMT)  # 2 bytes
_POS_SIZE = 3 * 4                                  # 3 float32
_QUATS_SIZE = NUM_SMPL_JOINTS * 4 * 4              # 24 * 4 float32 = 384 bytes
PER_PERSON_SIZE = _PERSON_ID_SIZE + _POS_SIZE + _QUATS_SIZE  # 398 bytes


# person tuple: (person_id: int, root_translation: array-like (3,), quats: array-like (24, 4) xyzw)
PersonPayload = Tuple[int, np.ndarray, np.ndarray]


def build_packet(frame_id: int, persons: Sequence[PersonPayload]) -> bytes:
    """Serialize a single frame's payload for UDP transmission.

    Values must already be in UE coordinate system (see ``transform``).
    ``persons`` may be empty (a valid heartbeat frame).
    """
    if not (0 <= frame_id < 2**32):
        raise ValueError(f"frame_id out of uint32 range: {frame_id}")
    if len(persons) > 255:
        raise ValueError(f"person_count out of uint8 range: {len(persons)}")

    parts: List[bytes] = [struct.pack(_HEADER_FMT, frame_id, len(persons))]
    for pid, root, quats in persons:
        if not (0 <= pid < 2**16):
            raise ValueError(f"person_id out of uint16 range: {pid}")
        root_arr = np.ascontiguousarray(np.asarray(root, dtype=np.float32).reshape(3))
        quats_arr = np.ascontiguousarray(
            np.asarray(quats, dtype=np.float32).reshape(NUM_SMPL_JOINTS * 4)
        )
        parts.append(struct.pack(_PERSON_ID_FMT, pid))
        parts.append(root_arr.tobytes())
        parts.append(quats_arr.tobytes())
    return b"".join(parts)


def parse_packet(data: bytes) -> Tuple[int, List[Tuple[int, np.ndarray, np.ndarray]]]:
    """Inverse of ``build_packet``. Provided for unit tests and Python-side receivers."""
    if len(data) < _HEADER_SIZE:
        raise ValueError(f"Packet too small: {len(data)} bytes")
    frame_id, person_count = struct.unpack_from(_HEADER_FMT, data, 0)
    expected = _HEADER_SIZE + person_count * PER_PERSON_SIZE
    if len(data) != expected:
        raise ValueError(
            f"Packet size mismatch: got {len(data)} bytes, expected {expected}"
        )

    persons: List[Tuple[int, np.ndarray, np.ndarray]] = []
    offset = _HEADER_SIZE
    for _ in range(person_count):
        (pid,) = struct.unpack_from(_PERSON_ID_FMT, data, offset)
        offset += _PERSON_ID_SIZE
        root = np.frombuffer(data, dtype=np.float32, count=3, offset=offset).copy()
        offset += _POS_SIZE
        quats = np.frombuffer(
            data, dtype=np.float32, count=NUM_SMPL_JOINTS * 4, offset=offset
        ).reshape(NUM_SMPL_JOINTS, 4).copy()
        offset += _QUATS_SIZE
        persons.append((pid, root, quats))
    return frame_id, persons
