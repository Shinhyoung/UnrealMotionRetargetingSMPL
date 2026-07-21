"""SMPL 24-joint definitions and UE mannequin bone-name mapping.

See ``docs/bone_mapping.md`` for the reasoning and joint semantics.
Coding rule (CLAUDE.md §7): don't hardcode indices elsewhere — import from here.
"""
from __future__ import annotations

NUM_SMPL_JOINTS = 24

# Joint index → SMPL joint name.
SMPL_JOINT_NAMES = (
    "pelvis",      # 0
    "L_hip",       # 1
    "R_hip",       # 2
    "spine1",      # 3
    "L_knee",      # 4
    "R_knee",      # 5
    "spine2",      # 6
    "L_ankle",     # 7
    "R_ankle",     # 8
    "spine3",      # 9
    "L_foot",      # 10
    "R_foot",      # 11
    "neck",        # 12
    "L_collar",    # 13
    "R_collar",    # 14
    "head",        # 15
    "L_shoulder",  # 16
    "R_shoulder",  # 17
    "L_elbow",     # 18
    "R_elbow",     # 19
    "L_wrist",     # 20
    "R_wrist",     # 21
    "L_hand",      # 22
    "R_hand",      # 23
)

# Kinematic parent for each joint (root = -1).
SMPL_PARENTS = (
    -1,  # 0  pelvis
    0,   # 1  L_hip
    0,   # 2  R_hip
    0,   # 3  spine1
    1,   # 4  L_knee
    2,   # 5  R_knee
    3,   # 6  spine2
    4,   # 7  L_ankle
    5,   # 8  R_ankle
    6,   # 9  spine3
    7,   # 10 L_foot
    8,   # 11 R_foot
    9,   # 12 neck
    9,   # 13 L_collar
    9,   # 14 R_collar
    12,  # 15 head
    13,  # 16 L_shoulder
    14,  # 17 R_shoulder
    16,  # 18 L_elbow
    17,  # 19 R_elbow
    18,  # 20 L_wrist
    19,  # 21 R_wrist
    20,  # 22 L_hand
    21,  # 23 R_hand
)

# SMPL joint index → UE5 mannequin bone name. Empty string = no counterpart, skip.
SMPL_TO_UE_BONE = {
    0:  "pelvis",
    1:  "thigh_l",
    2:  "thigh_r",
    3:  "spine_01",
    4:  "calf_l",
    5:  "calf_r",
    6:  "spine_02",
    7:  "foot_l",
    8:  "foot_r",
    9:  "spine_03",
    10: "ball_l",
    11: "ball_r",
    12: "neck_01",
    13: "clavicle_l",
    14: "clavicle_r",
    15: "head",
    16: "upperarm_l",
    17: "upperarm_r",
    18: "lowerarm_l",
    19: "lowerarm_r",
    20: "hand_l",
    21: "hand_r",
    22: "",   # SMPL "hand" = wrist-child. No UE mannequin counterpart.
    23: "",
}
