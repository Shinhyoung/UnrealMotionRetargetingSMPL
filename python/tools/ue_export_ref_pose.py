"""UE Editor Python script — export SK_Mannequin rest-pose bone transforms.

**Run inside UE Editor**:
    1. UE Editor 상단 메뉴 ``Tools → Execute Python Script``
    2. 이 파일 경로 선택
    3. Output Log 에서 저장 경로 확인 (기본: ``<project_root>/ue_ref_pose.json``)

The script spawns a temporary ``SkeletalMeshActor`` in the current editor world,
assigns it ``SKM_Manny``, and reads every bone's rest-pose world / parent-space
transform through ``USkeletalMeshComponent`` (the reliable Python API path in
UE 5.3). It then destroys the temp actor and writes JSON.

Output is consumed by ``tools/compute_bone_correction.py`` to build the
per-bone twist correction that aligns SMPL local frames to UE Mannequin local
frames (방법 A — CLAUDE.md §4.5, memory: UE 통합 진행 상태).
"""
import json
import os
import sys

try:
    import unreal
except ImportError:
    sys.stderr.write("This script must be run from inside Unreal Editor's Python.\n")
    sys.exit(1)


SKELETAL_MESH_PATH = "/Game/Characters/Mannequins/Meshes/SKM_Manny"
OUTPUT_PATH = "C:/0.shinhyoung/Project/1.Retargeting/MRetargeting/ue_ref_pose.json"


def _vec_to_xyz(v):
    return [float(v.x), float(v.y), float(v.z)]


def _quat_to_xyzw(q):
    return [float(q.x), float(q.y), float(q.z), float(q.w)]


def _get_editor_world():
    # UE 5.3 preferred: UnrealEditorSubsystem. Fallback to EditorLevelLibrary.
    try:
        subsys = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
        world = subsys.get_editor_world()
        if world is not None:
            return world
    except Exception:
        pass
    return unreal.EditorLevelLibrary.get_editor_world()


def _spawn_actor(world, mesh):
    # Prefer EditorActorSubsystem in 5.3+.
    try:
        subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        actor = subsys.spawn_actor_from_class(
            unreal.SkeletalMeshActor, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0)
        )
    except Exception:
        actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
            unreal.SkeletalMeshActor, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0)
        )
    if actor is None:
        raise RuntimeError("Failed to spawn temporary SkeletalMeshActor.")
    comp = actor.skeletal_mesh_component
    if comp is None:
        raise RuntimeError("Spawned actor has no SkeletalMeshComponent.")
    # SetSkeletalMeshAsset is the 5.3 name; SetSkinnedAssetAndUpdate on older.
    if hasattr(comp, "set_skeletal_mesh_asset"):
        comp.set_skeletal_mesh_asset(mesh)
    elif hasattr(comp, "set_skinned_asset_and_update"):
        comp.set_skinned_asset_and_update(mesh)
    else:
        comp.set_skeletal_mesh(mesh, True)
    return actor, comp


def _destroy_actor(actor):
    try:
        subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        subsys.destroy_actor(actor)
    except Exception:
        try:
            unreal.EditorLevelLibrary.destroy_actor(actor)
        except Exception:
            pass


def _all_bone_names(comp):
    # UE 5.3: SkeletalMeshComponent has get_num_bones + get_bone_name.
    try:
        n = comp.get_num_bones()
        return [str(comp.get_bone_name(i)) for i in range(n)]
    except Exception:
        # Fallback: enumerate via bone_space_transforms length (may not exist).
        try:
            transforms = comp.get_editor_property("bone_space_transforms")
            return [str(comp.get_bone_name(i)) for i in range(len(transforms))]
        except Exception:
            # Last-ditch: hard-coded UE5 mannequin bones we actually need.
            return [
                "root", "pelvis",
                "spine_01", "spine_02", "spine_03", "spine_04", "spine_05",
                "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l",
                "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r",
                "neck_01", "neck_02", "head",
                "thigh_l", "calf_l", "foot_l", "ball_l",
                "thigh_r", "calf_r", "foot_r", "ball_r",
            ]


def _get_bone_world_transform(comp, bone_name):
    """Return the bone's ref-pose world transform (component-space actually,
    since actor is at origin with identity rotation, component-space == world
    orient/position)."""
    # UE 5.3 signature: get_bone_transform(bone_name)  → world by default.
    try:
        return comp.get_bone_transform(bone_name)
    except Exception:
        pass
    # Some engine versions expose two-arg variants:
    try:
        space = unreal.RelativeTransformSpace.RTS_WORLD
        return comp.get_bone_transform(bone_name, space)
    except Exception:
        return None


def _get_parent_bone_name(comp, bone_name):
    # SkeletalMeshComponent doesn't always expose parent lookup in Python.
    # USkeleton does, so we go through the mesh's skeleton.
    mesh = comp.get_skeletal_mesh_asset() if hasattr(comp, "get_skeletal_mesh_asset") \
        else comp.get_editor_property("skeletal_mesh")
    if mesh is None:
        return ""
    skel = mesh.skeleton if hasattr(mesh, "skeleton") else mesh.get_editor_property("skeleton")
    if skel is None:
        return ""
    try:
        parent = skel.get_parent_name(bone_name)
    except Exception:
        try:
            idx = skel.get_reference_pose_bone_index(bone_name)  # unlikely API
            parent = skel.get_bone_name(skel.get_parent_index(idx))
        except Exception:
            parent = ""
    return str(parent) if parent else ""


def dump_skeleton():
    mesh = unreal.load_asset(SKELETAL_MESH_PATH)
    if mesh is None:
        raise RuntimeError(f"SkeletalMesh not found: {SKELETAL_MESH_PATH}")

    world = _get_editor_world()
    if world is None:
        raise RuntimeError("No editor world available. Open a level first.")

    actor, comp = _spawn_actor(world, mesh)
    try:
        bone_names = _all_bone_names(comp)
        bones_data = {}
        for name in bone_names:
            xf_world = _get_bone_world_transform(comp, name)
            if xf_world is None:
                continue
            parent_name = _get_parent_bone_name(comp, name)
            entry = {
                "parent": parent_name,
                "world_translation": _vec_to_xyz(xf_world.translation),
                "world_rotation_xyzw": _quat_to_xyzw(xf_world.rotation),
            }
            bones_data[name] = entry
    finally:
        _destroy_actor(actor)

    if not bones_data:
        raise RuntimeError("No bones exported — check the Python API compatibility.")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_mesh": SKELETAL_MESH_PATH,
                "note": "world_* are in UE component-space; actor spawned at origin "
                        "so component == world (orient/pos both).",
                "bones": bones_data,
            },
            f,
            indent=2,
        )
    unreal.log(f"[ue_export_ref_pose] wrote {len(bones_data)} bones -> {OUTPUT_PATH}")


dump_skeleton()
