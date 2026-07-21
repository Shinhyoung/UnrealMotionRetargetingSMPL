import numpy as np
import pytest

from config.bone_mapping import NUM_SMPL_JOINTS
from network.packet import PER_PERSON_SIZE, build_packet, parse_packet


def _make_person(pid=1, seed=0):
    rng = np.random.default_rng(seed)
    root = rng.normal(0, 1, 3).astype(np.float32)
    quats = rng.normal(0, 1, (NUM_SMPL_JOINTS, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    return (pid, root, quats)


def test_empty_packet_roundtrip():
    data = build_packet(42, [])
    frame_id, persons = parse_packet(data)
    assert frame_id == 42
    assert persons == []


def test_single_person_roundtrip():
    pid, root, quats = _make_person(pid=7)
    data = build_packet(1000, [(pid, root, quats)])
    assert len(data) == 5 + PER_PERSON_SIZE
    frame_id, persons = parse_packet(data)
    assert frame_id == 1000
    assert len(persons) == 1
    pid_r, root_r, quats_r = persons[0]
    assert pid_r == pid
    assert np.allclose(root_r, root)
    assert np.allclose(quats_r, quats)


def test_six_person_roundtrip():
    persons = [_make_person(pid=i + 1, seed=i) for i in range(6)]
    data = build_packet(0xFFFFFFFF, persons)
    assert len(data) == 5 + 6 * PER_PERSON_SIZE
    frame_id, out = parse_packet(data)
    assert frame_id == 0xFFFFFFFF
    assert len(out) == 6
    for (pid, root, quats), (pid_r, root_r, quats_r) in zip(persons, out):
        assert pid == pid_r
        assert np.allclose(root, root_r)
        assert np.allclose(quats, quats_r)


def test_frame_id_out_of_range():
    with pytest.raises(ValueError):
        build_packet(2**32, [])
    with pytest.raises(ValueError):
        build_packet(-1, [])


def test_person_id_out_of_range():
    _, root, quats = _make_person()
    with pytest.raises(ValueError):
        build_packet(0, [(2**16, root, quats)])


def test_parse_rejects_truncated_data():
    good = build_packet(1, [_make_person(pid=1)])
    with pytest.raises(ValueError):
        parse_packet(good[:-4])


def test_wire_format_endianness_and_layout():
    """Guard the wire format: first 5 bytes must be `<IB` (little-endian)."""
    import struct
    data = build_packet(0x01020304, [])
    frame_bytes = data[:4]
    person_count_byte = data[4]
    assert struct.unpack("<I", frame_bytes)[0] == 0x01020304
    assert person_count_byte == 0
