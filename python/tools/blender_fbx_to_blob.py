"""Convert a Mixamo (or any T-pose) FBX + SMPL rest npz into an smpl_model.bin blob.

Runs INSIDE Blender in headless mode. Automates the full workflow:
  1. Import FBX (auto-detect meter vs cm scale from mesh height).
  2. Try to map source armature bones (Mixamo names) → SMPL joint names.
     If mapping succeeds, use the SOURCE character's actual joint positions
     for the SMPL armature. If not (unknown rig), fall back to PKL rest_joints.
  3. Delete source armature, join all mesh parts, clear source vertex groups.
  4. Build a 24-bone SMPL armature at the resolved joint positions.
  5. Bind mesh to the SMPL armature with automatic weights (bone-heat).
  6. Extract vertices / faces / weights.
  7. Write blob with rest_joints = character-actual (or PKL fallback) and
     parents = SMPL kinematic structure.

Using the character's ACTUAL joint positions avoids pivot mismatch during LBS
(each joint rotation now applies around the correct pivot, so mesh segments
stay together at joints instead of visibly separating).
"""
import os
import struct
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


SMPL_JOINT_NAMES = [
    "pelvis", "L_hip", "R_hip", "spine1",
    "L_knee", "R_knee", "spine2",
    "L_ankle", "R_ankle", "spine3",
    "L_foot", "R_foot", "neck",
    "L_collar", "R_collar", "head",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hand", "R_hand",
]
NUM_JOINTS = 24

# Mixamo bone name → SMPL joint name
MIXAMO_TO_SMPL = {
    "Hips":               "pelvis",
    "LeftUpLeg":          "L_hip",
    "RightUpLeg":         "R_hip",
    "Spine":              "spine1",
    "LeftLeg":            "L_knee",
    "RightLeg":           "R_knee",
    "Spine1":             "spine2",
    "LeftFoot":           "L_ankle",
    "RightFoot":          "R_ankle",
    "Spine2":             "spine3",
    "LeftToeBase":        "L_foot",
    "RightToeBase":       "R_foot",
    "Neck":               "neck",
    "LeftShoulder":       "L_collar",
    "RightShoulder":      "R_collar",
    "Head":               "head",
    "LeftArm":            "L_shoulder",
    "RightArm":           "R_shoulder",
    "LeftForeArm":        "L_elbow",
    "RightForeArm":       "R_elbow",
    "LeftHand":           "L_wrist",
    "RightHand":          "R_wrist",
    "LeftHandMiddle1":    "L_hand",
    "RightHandMiddle1":   "R_hand",
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:]
    if len(argv) < 3:
        print("usage: blender --background --python blender_fbx_to_blob.py -- "
              "<mixamo.fbx> <smpl_rest.npz> <output.bin>",
              file=sys.stderr)
        sys.exit(1)
    return argv[0], argv[1], argv[2]


def load_smpl_rest_fallback(smpl_rest_npz_path):
    """Fallback rest joints from SMPL PKL (via wrapper's npz)."""
    data = np.load(smpl_rest_npz_path)
    return data["rest_joints"].astype(np.float32), data["parents"].astype(np.int32)


def smpl_to_blender(v):
    """SMPL Y-up → Blender Z-up. Mixamo characters face Blender -Y after FBX
    import, while SMPL rest expects character facing +Z → flip Z sign.

    smpl.x → bl.x
    smpl.y → bl.z   (up)
    smpl.z → -bl.y  (SMPL forward is Blender's backward for Mixamo imports)
    """
    return Vector((float(v[0]), -float(v[2]), float(v[1])))


def blender_to_smpl(v):
    """Inverse of smpl_to_blender."""
    return (float(v.x), float(v.z), -float(v.y))


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_fbx(path, scale=1.0):
    bpy.ops.import_scene.fbx(
        filepath=path,
        global_scale=scale,
        use_anim=False,
        ignore_leaf_bones=True,
        automatic_bone_orientation=True,
    )


def get_meshes():
    return [o for o in bpy.data.objects if o.type == "MESH"]


def get_armatures():
    return [o for o in bpy.data.objects if o.type == "ARMATURE"]


def select_only(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def apply_transforms(obj, location=True, rotation=True, scale=True):
    select_only(obj)
    bpy.ops.object.transform_apply(location=location, rotation=rotation, scale=scale)


def strip_prefix(name):
    """Return bone name without the 'mixamorig:' (or any 'foo:') prefix."""
    return name.split(":")[-1]


def build_mixamo_to_smpl_index(armature):
    """For every source bone, resolve the nearest ancestor SMPL joint index.

    Fingers, twist bones, etc. that aren't in MIXAMO_TO_SMPL still contribute
    to their nearest mapped ancestor (e.g. all fingers → L_hand / R_hand).
    Returns dict: source_bone_name → smpl_joint_index (0..23).
    """
    smpl_idx = {n: i for i, n in enumerate(SMPL_JOINT_NAMES)}
    out = {}
    for bone in armature.data.bones:
        b = bone
        while b is not None:
            smpl_name = MIXAMO_TO_SMPL.get(strip_prefix(b.name))
            if smpl_name is not None:
                out[bone.name] = smpl_idx[smpl_name]
                break
            b = b.parent
    return out


def try_extract_mixamo_joints(armature):
    """Return (rest_joints_smpl (24,3) float32, parents (24,) int32) or None.

    Reads the source armature's bone HEAD positions in world space, maps
    bone names → SMPL joint indices, and builds a parent array from the
    parent bone chain (using the same mapping).
    """
    world = armature.matrix_world
    bone_by_name = {b.name: b for b in armature.data.bones}
    smpl_head = {}       # smpl_name → Blender world Vector
    smpl_bone = {}       # smpl_name → source bone
    for b in armature.data.bones:
        stripped = strip_prefix(b.name)
        smpl_name = MIXAMO_TO_SMPL.get(stripped)
        if smpl_name is not None and smpl_name not in smpl_head:
            smpl_head[smpl_name] = world @ b.head_local
            smpl_bone[smpl_name] = b

    missing = [n for n in SMPL_JOINT_NAMES if n not in smpl_head]
    if missing:
        print(f"     mixamo-mapping: missing {len(missing)} SMPL joints: {missing}")
        return None

    # Build parents by walking each SMPL joint's source bone up to find an ancestor
    # that also maps to some SMPL joint (its SMPL parent).
    smpl_idx = {n: i for i, n in enumerate(SMPL_JOINT_NAMES)}
    parents = np.full(NUM_JOINTS, -1, dtype=np.int32)
    for i, name in enumerate(SMPL_JOINT_NAMES):
        b = smpl_bone[name].parent
        while b is not None:
            stripped = strip_prefix(b.name)
            parent_smpl = MIXAMO_TO_SMPL.get(stripped)
            if parent_smpl is not None:
                parents[i] = smpl_idx[parent_smpl]
                break
            b = b.parent

    # Convert to SMPL Y-up
    rest_joints = np.empty((NUM_JOINTS, 3), dtype=np.float32)
    for i, name in enumerate(SMPL_JOINT_NAMES):
        rest_joints[i] = blender_to_smpl(smpl_head[name])

    return rest_joints, parents


def clean_and_join_meshes(keep_vertex_groups=False):
    """Delete armatures, join all meshes. Returns single mesh.

    If keep_vertex_groups=True the Mixamo vertex groups on each mesh are
    preserved through the join (needed when we plan to remap them to SMPL).
    """
    for m in get_meshes():
        for mod in list(m.modifiers):
            if mod.type == "ARMATURE":
                m.modifiers.remove(mod)
    meshes = get_meshes()
    if meshes:
        bpy.ops.object.select_all(action="DESELECT")
        for m in meshes:
            m.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")

    for a in get_armatures():
        bpy.data.objects.remove(a, do_unlink=True)

    meshes = get_meshes()
    for m in meshes:
        apply_transforms(m)

    if len(meshes) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for m in meshes:
            m.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()

    joined = get_meshes()[0]
    if not keep_vertex_groups:
        joined.vertex_groups.clear()
    return joined


def build_smpl_armature(rest_joints_smpl, parents):
    """Create a 24-bone SMPL armature at the given rest joint positions."""
    heads_bl = [smpl_to_blender(rest_joints_smpl[i]) for i in range(NUM_JOINTS)]

    children_of = {i: [] for i in range(NUM_JOINTS)}
    for i in range(NUM_JOINTS):
        if parents[i] >= 0:
            children_of[int(parents[i])].append(i)

    bpy.ops.object.armature_add(location=(0.0, 0.0, 0.0))
    armature = bpy.context.object
    armature.name = "SMPL_Armature"
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = armature.data.edit_bones
    for eb in list(edit_bones):
        edit_bones.remove(eb)

    created = []
    for i, name in enumerate(SMPL_JOINT_NAMES):
        b = edit_bones.new(name)
        b.head = heads_bl[i]
        b.tail = heads_bl[i] + Vector((0.0, 0.0, 0.05))
        created.append(b)

    for i in range(NUM_JOINTS):
        if parents[i] >= 0:
            created[i].parent = created[int(parents[i])]

    for i in range(NUM_JOINTS):
        kids = children_of[i]
        if kids:
            created[i].tail = heads_bl[kids[0]]
        else:
            if parents[i] >= 0:
                dir_vec = heads_bl[i] - heads_bl[int(parents[i])]
                length = max(dir_vec.length * 0.5, 0.02)
                created[i].tail = heads_bl[i] + dir_vec.normalized() * length
            else:
                created[i].tail = heads_bl[i] + Vector((0.0, 0.0, 0.1))

    for b in edit_bones:
        if (b.tail - b.head).length < 1e-4:
            b.tail = b.head + Vector((0.0, 0.0, 0.02))

    bpy.ops.object.mode_set(mode="OBJECT")
    return armature


def get_z_extent(obj):
    world = obj.matrix_world
    if obj.type == "MESH":
        coords = [world @ v.co for v in obj.data.vertices]
    elif obj.type == "ARMATURE":
        coords = []
        for b in obj.data.bones:
            coords.append(world @ b.head_local)
            coords.append(world @ b.tail_local)
    else:
        raise TypeError(f"Unsupported obj type: {obj.type}")
    zs = [c.z for c in coords]
    return min(zs), max(zs)


def normalize_mesh_scale(mesh):
    """Ensure mesh height is in a sensible range (0.5–3m). Returns applied scale factor.

    Blender's FBX importer applies unit conversion inconsistently for Mixamo:
    some characters end up in meters, others in centimeters, sometimes 100x off.
    """
    apply_transforms(mesh)
    mn, mx = get_z_extent(mesh)
    h = mx - mn
    if h < 1e-6:
        return 1.0
    target = 1.7
    if h < 0.05 or h > 10.0:
        factor = target / h
        print(f"  auto-normalizing mesh scale: height={h:.4f}m → {target}m (factor={factor:.4f})")
        mesh.scale = (factor, factor, factor)
        apply_transforms(mesh, location=False, rotation=False, scale=True)
        return factor
    return 1.0


def normalize_armature_scale(armature, mesh):
    """Match armature Z-height to mesh Z-height, keeping feet aligned."""
    apply_transforms(armature)
    m_mn, m_mx = get_z_extent(mesh)
    a_mn, a_mx = get_z_extent(armature)
    m_h = m_mx - m_mn
    a_h = a_mx - a_mn
    if a_h < 1e-6 or m_h < 1e-6:
        return
    if abs(a_h - m_h) / m_h > 0.1:
        factor = m_h / a_h
        print(f"  scaling armature: height {a_h:.3f}m → {m_h:.3f}m (factor={factor:.4f})")
        armature.scale = (factor, factor, factor)
        apply_transforms(armature, location=False, rotation=False, scale=True)


def bind_with_automatic_weights(mesh, armature):
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")


def extract_mesh_geometry(mesh):
    """Extract vertices, triangulated faces, per-vertex UV, per-triangle material id, material names."""
    world = mesh.matrix_world

    V = len(mesh.data.vertices)
    verts = np.empty((V, 3), dtype=np.float32)
    for i, v in enumerate(mesh.data.vertices):
        verts[i] = blender_to_smpl(world @ v.co)

    faces = []
    face_mat_ids = []
    for poly in mesh.data.polygons:
        idx = list(poly.vertices)
        mat = poly.material_index
        if len(idx) == 3:
            faces.append(idx); face_mat_ids.append(mat)
        elif len(idx) == 4:
            faces.append([idx[0], idx[1], idx[2]]); face_mat_ids.append(mat)
            faces.append([idx[0], idx[2], idx[3]]); face_mat_ids.append(mat)
        else:
            for k in range(1, len(idx) - 1):
                faces.append([idx[0], idx[k], idx[k + 1]])
                face_mat_ids.append(mat)
    faces = np.array(faces, dtype=np.int32)
    face_mat_ids = np.array(face_mat_ids, dtype=np.int32)

    material_names = [
        (slot.material.name if slot.material else f"slot_{i}")
        for i, slot in enumerate(mesh.material_slots)
    ]
    if not material_names:
        material_names = ["default"]
        face_mat_ids[:] = 0

    uvs = np.zeros((V, 2), dtype=np.float32)
    if mesh.data.uv_layers.active:
        uv_data = mesh.data.uv_layers.active.data
        seen = np.zeros(V, dtype=bool)
        for poly in mesh.data.polygons:
            for li in range(poly.loop_start, poly.loop_start + poly.loop_total):
                vi = mesh.data.loops[li].vertex_index
                if not seen[vi]:
                    uv = uv_data[li].uv
                    uvs[vi] = (uv.x, 1.0 - uv.y)
                    seen[vi] = True
        print(f"  UVs extracted from '{mesh.data.uv_layers.active.name}'")
    else:
        print(f"  WARNING: mesh has no UV layer — textures will not map")

    print(f"  V={V}, F={len(faces)}, materials={len(material_names)}: {material_names}")
    return verts, faces, uvs, face_mat_ids, material_names


def extract_weights_mixamo_remap(mesh, bone_to_smpl_idx):
    """Read Mixamo vertex weights and remap into (V, 24) SMPL weight matrix.

    Multiple Mixamo bones (e.g. all fingers) can map to the same SMPL joint —
    their weights accumulate. Rows are normalized to sum to 1.
    """
    V = len(mesh.data.vertices)
    weights = np.zeros((V, NUM_JOINTS), dtype=np.float32)
    unassigned = 0
    unmapped = set()
    for vi, v in enumerate(mesh.data.vertices):
        row_total = 0.0
        for g in v.groups:
            vg_name = mesh.vertex_groups[g.group].name
            smpl_idx = bone_to_smpl_idx.get(vg_name)
            if smpl_idx is not None:
                weights[vi, smpl_idx] += g.weight
                row_total += g.weight
            else:
                unmapped.add(vg_name)
        if row_total < 1e-6:
            weights[vi, 0] = 1.0
            unassigned += 1
        else:
            weights[vi] /= row_total
    if unmapped:
        print(f"  {len(unmapped)} unmapped vertex groups (weights discarded), e.g. {list(unmapped)[:3]}")
    print(f"  weights: {unassigned}/{V} unassigned verts → pelvis")
    return weights


def extract_weights_automatic(mesh):
    """Read weights from SMPL-named vertex groups (post automatic-bind fallback)."""
    bone_to_idx = {n: i for i, n in enumerate(SMPL_JOINT_NAMES)}
    V = len(mesh.data.vertices)
    weights = np.zeros((V, NUM_JOINTS), dtype=np.float32)
    unassigned = 0
    for vi, v in enumerate(mesh.data.vertices):
        row_total = 0.0
        for g in v.groups:
            name = mesh.vertex_groups[g.group].name
            if name in bone_to_idx:
                weights[vi, bone_to_idx[name]] = g.weight
                row_total += g.weight
        if total := row_total:  # noqa: E231
            weights[vi] /= total
        else:
            weights[vi, 0] = 1.0
            unassigned += 1
    print(f"  weights (auto): {unassigned}/{V} unassigned verts → pelvis")
    return weights


def extract_textures(mesh, out_dir):
    """Save all texture images referenced by the mesh's materials to PNG.

    Handles Blender-packed FBX textures. Returns list of saved paths.
    """
    saved = []
    seen = set()

    # Report every material and its texture nodes.
    print(f"  materials on mesh: {len(mesh.material_slots)}")
    for i, slot in enumerate(mesh.material_slots):
        mat = slot.material
        mat_name = mat.name if mat else "<none>"
        uses_nodes = mat.use_nodes if mat else False
        print(f"    [{i}] material='{mat_name}' use_nodes={uses_nodes}")
        if not mat or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "TEX_IMAGE":
                continue
            img = node.image
            if img is None:
                print(f"        node '{node.name}' has no image")
                continue
            if img.name in seen:
                continue
            seen.add(img.name)
            packed = "packed" if img.packed_file else "unpacked"
            size = f"{img.size[0]}x{img.size[1]}"
            print(f"        image '{img.name}' ({size}, {packed})")

            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in img.name)
            if not safe.lower().endswith((".png", ".jpg", ".jpeg", ".tga", ".bmp")):
                safe += ".png"
            out_path = out_dir / safe

            try:
                img.file_format = "PNG"
                img.filepath_raw = str(out_path)
                img.save()
                saved.append(str(out_path))
                print(f"        → saved: {out_path}")
            except Exception as e:
                print(f"        WARNING save failed: {e}")

    # Fallback: dump every non-generated image loaded in this session.
    if not saved:
        print(f"  no textures found via materials — dumping ALL loaded images...")
        for img in bpy.data.images:
            if img.name in ("Render Result", "Viewer Node"):
                continue
            if img.size[0] == 0 or img.size[1] == 0:
                continue
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in img.name)
            if not safe.lower().endswith((".png", ".jpg", ".jpeg", ".tga", ".bmp")):
                safe += ".png"
            out_path = out_dir / safe
            try:
                img.file_format = "PNG"
                img.filepath_raw = str(out_path)
                img.save()
                saved.append(str(out_path))
                print(f"    fallback saved: {out_path}")
            except Exception as e:
                print(f"    fallback save failed for '{img.name}': {e}")

    return saved


def write_blob(out_path, rest_joints, parents, verts, faces, weights, uvs,
               face_mat_ids, material_names):
    V = len(verts)
    F = len(faces)
    NM = len(material_names)
    with open(out_path, "wb") as f:
        f.write(b"SMPB")
        f.write(struct.pack("<I", 3))                # v3: UV + multi-material
        f.write(struct.pack("<I", NUM_JOINTS))
        f.write(struct.pack("<I", V))
        f.write(struct.pack("<I", F))
        f.write(struct.pack("<I", 0))                # padding
        f.write(parents.tobytes(order="C"))
        f.write(rest_joints.tobytes(order="C"))
        f.write(verts.tobytes(order="C"))
        f.write(faces.tobytes(order="C"))
        f.write(weights.tobytes(order="C"))
        f.write(uvs.tobytes(order="C"))
        # v3:
        f.write(struct.pack("<I", NM))
        for name in material_names:
            b = name.encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)
        f.write(face_mat_ids.tobytes(order="C"))
    print(f"  blob: {os.path.getsize(out_path)/1024:.1f} KB, {NM} materials")


def main():
    mixamo_fbx, smpl_rest_npz, out_blob = parse_args()

    print(f"[1/7] Clearing scene, importing FBX: {mixamo_fbx}")
    clear_scene()
    import_fbx(mixamo_fbx, scale=0.01)

    src_armatures = get_armatures()
    if not src_armatures:
        raise RuntimeError("Input FBX has no armature")
    src_armature = src_armatures[0]
    print(f"     imported: {len(get_meshes())} meshes, "
          f"{len(src_armatures)} armatures, source has {len(src_armature.data.bones)} bones")

    for m in get_meshes():
        apply_transforms(m)
    apply_transforms(src_armature)

    print(f"[2/7] Mapping source bones → SMPL joints...")
    mapped = try_extract_mixamo_joints(src_armature)
    if mapped is not None:
        rest_joints, parents = mapped
        bone_to_smpl_idx = build_mixamo_to_smpl_index(src_armature)
        n_resolved = len(bone_to_smpl_idx)
        n_total = len(src_armature.data.bones)
        print(f"     ✓ 24 SMPL joints mapped; {n_resolved}/{n_total} source bones resolved to SMPL")
        use_mixamo_weights = True
    else:
        print(f"     ✗ mapping failed; PKL rest + automatic weights fallback")
        rest_joints, parents = load_smpl_rest_fallback(smpl_rest_npz)
        bone_to_smpl_idx = None
        use_mixamo_weights = False

    print(f"[3/7] Joining meshes (keep_vertex_groups={use_mixamo_weights})...")
    mesh = clean_and_join_meshes(keep_vertex_groups=use_mixamo_weights)
    print(f"     joined: '{mesh.name}' with {len(mesh.data.vertices)} verts, "
          f"{len(mesh.vertex_groups)} vertex groups")

    print(f"[4/7] Normalizing mesh scale...")
    scale_factor = normalize_mesh_scale(mesh)
    if abs(scale_factor - 1.0) > 1e-6:
        rest_joints = rest_joints * scale_factor
        print(f"     scaled rest_joints by {scale_factor:.4f}")

    print(f"[5/7] Extracting weights...")
    if use_mixamo_weights:
        weights = extract_weights_mixamo_remap(mesh, bone_to_smpl_idx)
    else:
        armature = build_smpl_armature(rest_joints, parents)
        normalize_armature_scale(armature, mesh)
        bind_with_automatic_weights(mesh, armature)
        weights = extract_weights_automatic(mesh)

    print(f"[6/7] Extracting geometry (verts, faces, UV, material ids)...")
    verts, faces, uvs, face_mat_ids, material_names = extract_mesh_geometry(mesh)

    print(f"[7/7] Writing blob v3 to {out_blob}")
    write_blob(out_blob, rest_joints, parents, verts, faces, weights, uvs,
               face_mat_ids, material_names)

    # Extract embedded textures alongside the blob.
    out_dir = Path(out_blob).parent
    tex_dir = out_dir / (Path(out_blob).stem + "_textures")
    tex_dir.mkdir(parents=True, exist_ok=True)
    saved = extract_textures(mesh, tex_dir)
    if saved:
        print(f"[textures] wrote {len(saved)} file(s) to {tex_dir}:")
        for p in saved:
            print(f"    {p}")
    else:
        print(f"[textures] none found (mesh may have no image texture)")

    print("[done]")


if __name__ == "__main__":
    main()
