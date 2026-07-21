"""Blender headless script — build SMPL skeleton FBX from smpl_rest.json + smpl_tpose.obj.

Run from command line (Blender must be installed):

    blender --background --python tools/make_smpl_fbx_blender.py -- \\
        --skeleton-json ../smpl_rest.json \\
        --mesh-obj      ../smpl_tpose.obj \\
        --output-fbx    ../SMPL_Skeleton.fbx

The resulting FBX contains:
- Armature with 24 SMPL bones at their bind-pose positions
- SMPL T-pose mesh, bound to the armature with automatic weights

Import to UE 5.3: File → Import Into Level → select SMPL_Skeleton.fbx.
UE will create ``SK_SMPL_Skeleton`` (SkeletalMesh) + ``Skeleton_SMPL_Skeleton``
(Skeleton) assets.

Notes
-----
- Y-up is preserved on export (SMPL convention). UE will re-interpret via its
  own axis conventions on import.
- Automatic vertex weights are approximate (Blender's "auto weights"). Good
  enough for IK Retargeter which only needs joint transforms.
"""
import argparse
import json
import sys
from pathlib import Path

# Blender-only imports (fail gracefully outside Blender)
try:
    import bpy
    import mathutils
except ImportError:
    sys.stderr.write("This script must be run inside Blender via `blender --background --python`.\n")
    sys.exit(1)


def _parse_argv():
    # Blender passes everything after `--` to us
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--skeleton-json", required=True)
    p.add_argument("--mesh-obj", required=True)
    p.add_argument("--output-fbx", required=True)
    p.add_argument("--scale", type=float, default=1.0,
                   help="Uniform scale applied to armature + mesh (default 1.0 = keep SMPL meters). "
                        "Set 100.0 to bake SMPL meters → UE centimeters (avoids UE import-scale step).")
    p.add_argument("--weights", default="",
                   help="Path to smpl_weights.npz (from dump_smpl_rest.py). When set, "
                        "uses EXACT SMPL LBS weights instead of Blender's auto-weights, "
                        "matching SMPL's canonical skinning.")
    return p.parse_args(argv)


def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)


GLOBAL_SCALE = 1.0  # patched by main() from --scale


def _smpl_yup_to_blender_zup(pos):
    """SMPL Y-up right-hand → Blender Z-up right-hand.

    Blender OBJ importer auto-converts Y-up meshes to Z-up. We must apply the
    same conversion to the armature bone positions or the skeleton ends up
    lying down while the mesh stands.

    Mapping (Blender defaults: X-right, Y-forward, Z-up):
      blender.x = +smpl.x   (right unchanged)
      blender.y = -smpl.z   (forward: SMPL person's back = +Z → Blender -Y)
      blender.z = +smpl.y   (up)
    """
    return mathutils.Vector((pos[0] * GLOBAL_SCALE,
                             -pos[2] * GLOBAL_SCALE,
                             pos[1] * GLOBAL_SCALE))


def _build_armature(skeleton: dict):
    bpy.ops.object.armature_add(location=(0, 0, 0))
    armature_obj = bpy.context.active_object
    armature_obj.name = "SMPL_Armature"
    armature = armature_obj.data

    bpy.ops.object.mode_set(mode="EDIT")
    # Remove the default bone Blender adds
    for b in list(armature.edit_bones):
        armature.edit_bones.remove(b)

    # Build bones with natural orientation (each points toward its primary
    # child). This gives Blender/FBX/UE bone directions that MATCH the SMPL
    # rest bone directions, which is what our C++ RT Motion Retarget node
    # (swing algorithm) needs to align source and target bones by direction.
    bones = skeleton["bones"]
    edit_bones = {}
    for entry in bones:
        name = entry["name"]
        pos = _smpl_yup_to_blender_zup(entry["position_world"])
        parent_name = entry["parent_name"]

        bone = armature.edit_bones.new(name)
        bone.head = pos
        bone.tail = pos + mathutils.Vector((0.0, 0.0, 0.05))  # placeholder
        bone.roll = 0.0
        if parent_name and parent_name in edit_bones:
            bone.parent = edit_bones[parent_name]
        edit_bones[name] = bone

    # Second pass: set each bone's tail to first-child head. Leaves keep a
    # small up-offset so the bone has non-zero length.
    for entry in bones:
        name = entry["name"]
        bone = edit_bones[name]
        children = [b for b in edit_bones.values() if b.parent == bone]
        if children:
            bone.tail = children[0].head
        if (bone.tail - bone.head).length < 1e-4:
            bone.tail = bone.head + mathutils.Vector((0.0, 0.0, 0.05))

    bpy.ops.object.mode_set(mode="OBJECT")
    return armature_obj


def _import_obj(obj_path: str):
    # Blender 4.x uses wm.obj_import; 3.x uses import_scene.obj
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=obj_path)
    else:
        bpy.ops.import_scene.obj(filepath=obj_path)
    # Return the freshly-imported mesh object
    for obj in bpy.context.selected_objects:
        if obj.type == "MESH":
            return obj
    # Fallback: last mesh in scene
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            return obj
    raise RuntimeError("Failed to import OBJ mesh.")


def _parent_with_auto_weights(mesh_obj, armature_obj):
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")


def _parent_with_smpl_weights(mesh_obj, armature_obj, skeleton, weights_npz_path):
    """Parent mesh to armature and assign EXACT SMPL LBS weights.

    Skips Blender's auto-weight heuristic. Uses per-vertex per-bone weights
    loaded from the SMPL .pkl (via dump_smpl_rest.py --output-weights).
    """
    import numpy as np

    data = np.load(weights_npz_path)
    W = data["weights"]  # shape (V, 24)
    n_verts = len(mesh_obj.data.vertices)
    if W.shape[0] != n_verts:
        raise RuntimeError(
            f"weights vertex count {W.shape[0]} != mesh vertex count {n_verts}"
        )

    bone_names = [b["name"] for b in skeleton["bones"]]
    if W.shape[1] != len(bone_names):
        raise RuntimeError(
            f"weights bone count {W.shape[1]} != skeleton bone count {len(bone_names)}"
        )

    # Parent mesh to armature WITHOUT auto-weights (empty groups only).
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.parent_set(type="ARMATURE_NAME")   # creates empty vertex groups per bone

    # Now assign per-vertex weights. Only add non-trivial weights (>threshold)
    # to keep the FBX small and Blender happy.
    threshold = 1e-4
    print(f"[make_smpl_fbx] assigning SMPL LBS weights ({n_verts} verts × {len(bone_names)} bones) ...")
    for bone_idx, bone_name in enumerate(bone_names):
        vg = mesh_obj.vertex_groups.get(bone_name)
        if vg is None:
            vg = mesh_obj.vertex_groups.new(name=bone_name)
        col = W[:, bone_idx]
        # Efficient: iterate over non-zero indices only
        nz = np.where(col > threshold)[0]
        for vi in nz:
            vg.add([int(vi)], float(col[vi]), "REPLACE")
    print(f"[make_smpl_fbx] weights assigned.")


def _export_fbx(output_path: str):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=True,
        apply_scale_options="FBX_SCALE_ALL",
        axis_forward="-Z",
        axis_up="Y",
        add_leaf_bones=False,
        bake_anim=False,
        # Ensure skinned mesh & armature exported correctly
        object_types={"ARMATURE", "MESH"},
    )


def main():
    args = _parse_argv()

    global GLOBAL_SCALE
    GLOBAL_SCALE = float(args.scale)

    with open(args.skeleton_json, "r", encoding="utf-8") as f:
        skeleton = json.load(f)

    print(f"[make_smpl_fbx] skeleton: {len(skeleton['bones'])} bones")
    print(f"[make_smpl_fbx] mesh:     {args.mesh_obj}")
    print(f"[make_smpl_fbx] output:   {args.output_fbx}")
    print(f"[make_smpl_fbx] scale:    {GLOBAL_SCALE}")

    _clear_scene()

    armature_obj = _build_armature(skeleton)
    mesh_obj = _import_obj(args.mesh_obj)
    # Scale imported mesh to match armature scale.
    if GLOBAL_SCALE != 1.0:
        mesh_obj.scale = (GLOBAL_SCALE, GLOBAL_SCALE, GLOBAL_SCALE)
        bpy.ops.object.select_all(action="DESELECT")
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if args.weights:
        _parent_with_smpl_weights(mesh_obj, armature_obj, skeleton, args.weights)
    else:
        _parent_with_auto_weights(mesh_obj, armature_obj)
    _export_fbx(args.output_fbx)

    print(f"[make_smpl_fbx] wrote {args.output_fbx}")


if __name__ == "__main__":
    main()
