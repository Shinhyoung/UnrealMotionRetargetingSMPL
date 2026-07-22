"""UDP packet (de)serialization.

Wire format (v2 — variable joint count per person, added 2026-07-22):
    frame_id : uint32
    person_count : uint8
    per person:
        person_id       : uint16
        joint_count     : uint8               # 24 for SMPL, 55 for SMPL-X
        root_translation: float32 x 3
        quats           : float32 x (joint_count * 4)  # xyzw

Backward compat: existing 24-joint pipelines still work — joint_count is
just per-person metadata now.
"""
from __future__ import annotations

import struct
from typing import List, Sequence, Tuple

import numpy as np


_HEADER_FMT = "<IB"                                # frame_id (u32) + person_count (u8)
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)        # 5 bytes
_PERSON_HEAD_FMT = "<HB"                           # pid (u16) + joint_count (u8)
_PERSON_HEAD_SIZE = struct.calcsize(_PERSON_HEAD_FMT)  # 3 bytes
_POS_SIZE = 3 * 4                                  # 3 float32


# person tuple: (person_id: int, root_translation: (3,), quats: (J, 4) xyzw)
PersonPayload = Tuple[int, np.ndarray, np.ndarray]


def build_packet(frame_id: int, persons: Sequence[PersonPayload]) -> bytes:
    if not (0 <= frame_id < 2**32):
        raise ValueError(f"frame_id out of uint32 range: {frame_id}")
    if len(persons) > 255:
        raise ValueError(f"person_count out of uint8 range: {len(persons)}")

    parts: List[bytes] = [struct.pack(_HEADER_FMT, frame_id, len(persons))]
    for pid, root, quats in persons:
        if not (0 <= pid < 2**16):
            raise ValueError(f"person_id out of uint16 range: {pid}")
        quats_arr = np.ascontiguousarray(np.asarray(quats, dtype=np.float32))
        if quats_arr.ndim != 2 or quats_arr.shape[1] != 4:
            raise ValueError(f"quats must be (J, 4), got shape {quats_arr.shape}")
        joint_count = quats_arr.shape[0]
        if not (0 <= joint_count < 256):
            raise ValueError(f"joint_count out of uint8 range: {joint_count}")
        root_arr = np.ascontiguousarray(np.asarray(root, dtype=np.float32).reshape(3))
        parts.append(struct.pack(_PERSON_HEAD_FMT, pid, joint_count))
        parts.append(root_arr.tobytes())
        parts.append(quats_arr.reshape(-1).tobytes())
    return b"".join(parts)


def parse_packet(data: bytes) -> Tuple[int, List[Tuple[int, np.ndarray, np.ndarray]]]:
    if len(data) < _HEADER_SIZE:
        raise ValueError(f"Packet too small: {len(data)} bytes")
    frame_id, person_count = struct.unpack_from(_HEADER_FMT, data, 0)

    persons: List[Tuple[int, np.ndarray, np.ndarray]] = []
    offset = _HEADER_SIZE
    for _ in range(person_count):
        if offset + _PERSON_HEAD_SIZE > len(data):
            raise ValueError(f"Packet truncated at person header (offset {offset})")
        pid, joint_count = struct.unpack_from(_PERSON_HEAD_FMT, data, offset)
        offset += _PERSON_HEAD_SIZE
        quat_bytes = joint_count * 4 * 4
        if offset + _POS_SIZE + quat_bytes > len(data):
            raise ValueError(f"Packet truncated at person body (offset {offset})")
        root = np.frombuffer(data, dtype=np.float32, count=3, offset=offset).copy()
        offset += _POS_SIZE
        quats = np.frombuffer(
            data, dtype=np.float32, count=joint_count * 4, offset=offset
        ).reshape(joint_count, 4).copy()
        offset += quat_bytes
        persons.append((pid, root, quats))
    return frame_id, persons
