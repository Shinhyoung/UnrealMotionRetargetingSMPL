"""Per-bone rest-orientation correction (방법 A).

Motivation
----------
The naive path (``smpl_quat_to_ue_quat``) only applies the SMPL→UE basis
change ``M R M^-1`` per joint. This treats each SMPL joint's rest local frame
as identical to the corresponding UE bone's rest local frame — which is
**false** for UE's mannequin (upperarm points along -Y, spine has a slight
tilt, etc.). The symptom is limbs jumping to the wrong orientation at rest
and picking up rotations about the wrong axes.

Derivation
----------
We want each UE bone's post-send world orient to equal ``M W_smpl[i] M^-1``
composed on top of the UE mannequin's rest world orient — i.e. SMPL delta
applied to UE rest::

    W_ue_target[i] = (M · W_smpl_chain[i] · M^-1) · W_ref[i]

Since UE evaluates ``W_ue[i] = W_ue[parent] · Q_send[i]`` and the parent is
already correctly retargeted, we get::

    Q_send[i] = W_ref[parent(i)]^-1 · (M · R_smpl[i] · M^-1) · W_ref[i]

Where ``M R_smpl[i] M^-1`` is what ``smpl_quat_to_ue_quat`` already computes.

At rest (``R_smpl[i] = I``): ``Q_send[i] = W_ref[parent]^-1 · W_ref[i] =
L_ref[i]`` — UE local rest, so the mannequin holds T-pose. ✓

File format
-----------
``.npz`` with two arrays, indexed by SMPL joint 0..23:

* ``parent_world_inv`` : (24, 4) xyzw quats, ``inverse(W_ref[parent(i)])``.
  For joint 0 (pelvis, parent = -1) uses identity.
* ``own_world`` : (24, 4) xyzw quats, ``W_ref[i]``.

Slots for SMPL joints that have no UE counterpart (``SMPL_TO_UE_BONE[i] ==
""``, i.e. hands 22 & 23) hold identities and are inert — the pipeline still
sends them but they never map to a bone.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from config.bone_mapping import NUM_SMPL_JOINTS
from transform.rotation import normalize_quat, quat_multiply_xyzw


IDENTITY_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def identity_correction() -> tuple[np.ndarray, np.ndarray]:
    """Return (parent_world_inv, own_world) filled with identity quats."""
    parent = np.tile(IDENTITY_QUAT_XYZW, (NUM_SMPL_JOINTS, 1)).astype(np.float64)
    own = parent.copy()
    return parent, own


def load_correction(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load per-joint correction quaternions from a .npz file.

    Returns identity correction if path is empty or file is missing.
    Both arrays must be shape (24, 4), xyzw, unit quaternions.
    """
    if not path:
        return identity_correction()
    p = Path(path)
    if not p.is_file():
        return identity_correction()
    data = np.load(str(p))
    if "parent_world_inv" not in data or "own_world" not in data:
        raise ValueError(
            f"Correction file {p} missing required arrays "
            "'parent_world_inv' and 'own_world'."
        )
    parent = np.asarray(data["parent_world_inv"], dtype=np.float64)
    own = np.asarray(data["own_world"], dtype=np.float64)
    expected = (NUM_SMPL_JOINTS, 4)
    if parent.shape != expected or own.shape != expected:
        raise ValueError(
            f"Correction arrays must be {expected}; got parent={parent.shape}, own={own.shape}."
        )
    return normalize_quat(parent), normalize_quat(own)


def apply_bone_correction(
    quats_ue_conjugated: np.ndarray,
    parent_world_inv: np.ndarray,
    own_world: np.ndarray,
) -> np.ndarray:
    """Apply ``Q_send[i] = parent_world_inv[i] · quats_ue_conjugated[i] · own_world[i]``.

    ``quats_ue_conjugated`` is the per-joint rotation after
    ``smpl_quat_to_ue_quat`` (i.e. already ``M R_smpl M^-1``). All inputs are
    xyzw, shape (24, 4).
    """
    q = np.asarray(quats_ue_conjugated, dtype=np.float64)
    p = np.asarray(parent_world_inv, dtype=np.float64)
    o = np.asarray(own_world, dtype=np.float64)
    if not (q.shape == p.shape == o.shape):
        raise ValueError(
            f"Shape mismatch: q={q.shape} parent_inv={p.shape} own={o.shape}"
        )
    return normalize_quat(quat_multiply_xyzw(quat_multiply_xyzw(p, q), o))
