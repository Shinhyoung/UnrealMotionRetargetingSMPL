// Copyright — see /LICENSE for details.

#pragma once

#include "CoreMinimal.h"
#include "Math/Quat.h"
#include "Math/Transform.h"
#include "Math/Vector.h"

/**
 * Loads SMPL model data (rest joints, kinematic tree, mesh vertices, faces,
 * skinning weights) from a compact binary blob produced by
 * ``python/tools/dump_smpl_blob.py`` and evaluates the full SMPL Linear
 * Blend Skinning (LBS) forward pass on the CPU.
 *
 * This bypasses UE's Skeletal system entirely — instead of retargeting
 * SMPL rotations to a UE skeleton (which has its own bone convention),
 * we compute mesh vertices exactly as Python's SMPL forward does, using
 * SMPL's canonical rest orientation (identity local for all joints).
 *
 * Coordinate space: SMPL Y-up meters. Callers should apply their own
 * basis change (SMPL Y-up → UE Z-up) after retrieving vertices.
 */
class MOTIONRETARGET_API FSMPLModel
{
public:
    FSMPLModel() = default;

    /** Load blob file. Returns false with UE_LOG on failure. */
    bool LoadFromFile(const FString& FilePath);

    /** Number of body joints (always 24 for SMPL). */
    int32 GetNumJoints() const { return NumJoints; }
    /** Number of mesh vertices (always 6890 for SMPL). */
    int32 GetNumVertices() const { return NumVerts; }
    /** Number of mesh triangles (always 13776 for SMPL). */
    int32 GetNumFaces() const { return NumFaces; }

    /** Flat triangle indices [f0.a, f0.b, f0.c, f1.a, ...] — 3 * NumFaces long. */
    const TArray<int32>& GetFaces() const { return FacesFlat; }

    /**
     * Compute per-vertex world positions for the given pose. Convenience:
     * runs joint FK then LBS in one call.
     */
    void ComputeVertices(
        const FQuat& GlobalOrient,
        const TArray<FQuat>& BodyPose,
        const FVector& RootTrans,
        TArray<FVector>& OutVertices) const;

    /**
     * Compute per-joint world transforms (SMPL Y-up meters). Exposed so
     * callers can inject post-FK modifications (e.g. foot IK) before LBS.
     */
    void ComputeJointWorlds(
        const FQuat& GlobalOrient,
        const TArray<FQuat>& BodyPose,
        const FVector& RootTrans,
        TArray<FTransform>& OutJointWorlds) const;

    /** LBS pass given pre-computed joint world transforms. */
    void ComputeVerticesFromJoints(
        const TArray<FTransform>& JointWorlds,
        TArray<FVector>& OutVertices) const;

    /** Rest joint positions (SMPL Y-up meters). Read-only. */
    const TArray<FVector>& GetRestJoints() const { return RestJoints; }

    /** Per-vertex UV0. Empty for v1 blobs; length == NumVerts for v2+. */
    const TArray<FVector2D>& GetUV0() const { return UV0; }

    /** Material slot names (v3+). Length == number of materials on original mesh. */
    const TArray<FString>& GetMaterialNames() const { return MaterialNames; }

    /** Per-triangle material id (0..NumMaterials-1). Length == NumFaces. */
    const TArray<int32>& GetFaceMaterialIds() const { return FaceMaterialIds; }

private:
    int32 NumJoints = 0;
    int32 NumVerts = 0;
    int32 NumFaces = 0;

    TArray<int32> Parents;                // NumJoints
    TArray<FVector> RestJoints;           // NumJoints, SMPL Y-up world
    TArray<FVector> RestVerts;            // NumVerts, SMPL Y-up world
    TArray<int32> FacesFlat;              // NumFaces * 3
    TArray<FVector2D> UV0;                // NumVerts (v2+) or empty (v1)
    TArray<FString> MaterialNames;        // Material slot names (v3+)
    TArray<int32> FaceMaterialIds;        // NumFaces, in [0, NumMaterials) (v3+)

    /**
     * Sparse skinning weights stored as CSR-like flat arrays (avoids nested
     * TArray which UE's UPROPERTY-adjacent machinery has trouble with).
     *   InfluenceRowStart[v]   → start index into InfluenceBones / InfluenceWeights
     *   InfluenceRowStart[v+1] → one past end
     * There are NumVerts + 1 entries in InfluenceRowStart so the last row can
     * still be sliced normally.
     */
    TArray<int32> InfluenceRowStart;
    TArray<int32> InfluenceBones;
    TArray<float> InfluenceWeights;

    /** Precomputed inverse translation of each joint at rest (rest orient is identity). */
    TArray<FVector> RestJointsNeg;
};
