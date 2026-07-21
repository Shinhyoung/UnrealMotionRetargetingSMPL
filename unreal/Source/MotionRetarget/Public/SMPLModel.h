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
     * Compute per-vertex world positions for the given pose.
     *
     *   GlobalOrient : pelvis rotation (SMPL "joint 0")
     *   BodyPose     : rotations for joints 1..23 (in order) — must be 23 long
     *   RootTrans    : additional world-space translation applied to pelvis
     *   OutVertices  : filled with 6890 positions in SMPL Y-up meters
     */
    void ComputeVertices(
        const FQuat& GlobalOrient,
        const TArray<FQuat>& BodyPose,
        const FVector& RootTrans,
        TArray<FVector>& OutVertices) const;

private:
    int32 NumJoints = 0;
    int32 NumVerts = 0;
    int32 NumFaces = 0;

    TArray<int32> Parents;                // NumJoints
    TArray<FVector> RestJoints;           // NumJoints, SMPL Y-up world
    TArray<FVector> RestVerts;            // NumVerts, SMPL Y-up world
    TArray<int32> FacesFlat;              // NumFaces * 3

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

    void ComputeJointWorlds(
        const FQuat& GlobalOrient,
        const TArray<FQuat>& BodyPose,
        const FVector& RootTrans,
        TArray<FTransform>& OutJointWorlds) const;
};
