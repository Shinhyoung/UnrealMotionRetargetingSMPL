// Copyright — see /LICENSE for details.

#include "SMPLModel.h"

#include "HAL/PlatformFileManager.h"
#include "Misc/FileHelper.h"

DEFINE_LOG_CATEGORY_STATIC(LogSMPLModel, Log, All);


bool FSMPLModel::LoadFromFile(const FString& FilePath)
{
    TArray<uint8> Bytes;
    if (!FFileHelper::LoadFileToArray(Bytes, *FilePath))
    {
        UE_LOG(LogSMPLModel, Error, TEXT("SMPL blob missing: %s"), *FilePath);
        return false;
    }
    if (Bytes.Num() < 24)
    {
        UE_LOG(LogSMPLModel, Error, TEXT("SMPL blob too small: %d bytes"), Bytes.Num());
        return false;
    }

    const uint8* p = Bytes.GetData();
    auto ReadU32 = [&p]() -> uint32
    {
        uint32 v;
        FMemory::Memcpy(&v, p, 4);
        p += 4;
        return v;
    };

    // Header
    if (FMemory::Memcmp(p, "SMPB", 4) != 0)
    {
        UE_LOG(LogSMPLModel, Error, TEXT("Bad magic in SMPL blob"));
        return false;
    }
    p += 4;
    const uint32 Version = ReadU32();
    if (Version < 1 || Version > 3)
    {
        UE_LOG(LogSMPLModel, Error, TEXT("Unsupported SMPL blob version %u"), Version);
        return false;
    }
    NumJoints = (int32)ReadU32();
    NumVerts = (int32)ReadU32();
    NumFaces = (int32)ReadU32();
    (void)ReadU32();   // padding

    // Parents
    Parents.SetNumUninitialized(NumJoints);
    FMemory::Memcpy(Parents.GetData(), p, sizeof(int32) * NumJoints);
    p += sizeof(int32) * NumJoints;

    // Rest joints (SMPL Y-up meters)
    RestJoints.SetNumUninitialized(NumJoints);
    for (int32 i = 0; i < NumJoints; ++i)
    {
        float xyz[3];
        FMemory::Memcpy(xyz, p, 12);
        p += 12;
        RestJoints[i] = FVector(xyz[0], xyz[1], xyz[2]);
    }

    // Rest vertices
    RestVerts.SetNumUninitialized(NumVerts);
    for (int32 i = 0; i < NumVerts; ++i)
    {
        float xyz[3];
        FMemory::Memcpy(xyz, p, 12);
        p += 12;
        RestVerts[i] = FVector(xyz[0], xyz[1], xyz[2]);
    }

    // Faces (flattened)
    FacesFlat.SetNumUninitialized(NumFaces * 3);
    FMemory::Memcpy(FacesFlat.GetData(), p, sizeof(int32) * NumFaces * 3);
    p += sizeof(int32) * NumFaces * 3;

    // Weights: (NumVerts, NumJoints) float32 dense on disk.
    // Convert to flat CSR-like sparse arrays (avoids nested TArray).
    const float* WeightRows = reinterpret_cast<const float*>(p);
    InfluenceRowStart.SetNumUninitialized(NumVerts + 1);
    InfluenceBones.Reset();
    InfluenceWeights.Reset();
    InfluenceBones.Reserve(NumVerts * 4);      // typical SMPL sparsity
    InfluenceWeights.Reserve(NumVerts * 4);
    InfluenceRowStart[0] = 0;
    for (int32 v = 0; v < NumVerts; ++v)
    {
        for (int32 j = 0; j < NumJoints; ++j)
        {
            const float w = WeightRows[v * NumJoints + j];
            if (w > 1e-4f)
            {
                InfluenceBones.Add(j);
                InfluenceWeights.Add(w);
            }
        }
        InfluenceRowStart[v + 1] = InfluenceBones.Num();
    }
    p += sizeof(float) * NumVerts * NumJoints;

    // Optional per-vertex UV (v2+). Zero-init for v1.
    UV0.SetNumUninitialized(NumVerts);
    if (Version >= 2)
    {
        for (int32 i = 0; i < NumVerts; ++i)
        {
            float uv[2];
            FMemory::Memcpy(uv, p, 8);
            p += 8;
            UV0[i] = FVector2D(uv[0], uv[1]);
        }
    }
    else
    {
        for (int32 i = 0; i < NumVerts; ++i)
        {
            UV0[i] = FVector2D::ZeroVector;
        }
    }

    // Optional multi-material (v3+): names + per-face material id.
    MaterialNames.Reset();
    FaceMaterialIds.SetNumUninitialized(NumFaces);
    if (Version >= 3)
    {
        const uint32 NM = ReadU32();
        MaterialNames.Reserve(NM);
        for (uint32 i = 0; i < NM; ++i)
        {
            const uint32 NameLen = ReadU32();
            MaterialNames.Add(FString(NameLen, (const ANSICHAR*)p));
            p += NameLen;
        }
        FMemory::Memcpy(FaceMaterialIds.GetData(), p, sizeof(int32) * NumFaces);
        p += sizeof(int32) * NumFaces;
    }
    else
    {
        MaterialNames.Add(TEXT("default"));
        for (int32 i = 0; i < NumFaces; ++i) FaceMaterialIds[i] = 0;
    }

    // Precompute negated rest joint positions. At rest all joint rotations
    // are identity, so rest_world_inv is simply translation(-RestJoints[i]).
    // Storing negated position is enough — we combine inline below.
    RestJointsNeg.SetNumUninitialized(NumJoints);
    for (int32 i = 0; i < NumJoints; ++i)
    {
        RestJointsNeg[i] = -RestJoints[i];
    }

    UE_LOG(LogSMPLModel, Display,
           TEXT("SMPL loaded: %d joints, %d verts, %d faces"),
           NumJoints, NumVerts, NumFaces);
    return true;
}


void FSMPLModel::ComputeJointWorlds(
    const FQuat& GlobalOrient,
    const TArray<FQuat>& BodyPose,
    const FVector& RootTrans,
    TArray<FTransform>& OutJointWorlds) const
{
    OutJointWorlds.SetNumUninitialized(NumJoints);

    // Joint 0 (pelvis): full transform combining root translation + global orient.
    OutJointWorlds[0] = FTransform(GlobalOrient, RestJoints[0] + RootTrans);

    // Remaining joints via kinematic chain:
    //   world[i] = local[i] * world[parent]        (UE FTransform composition)
    //   local[i] = FTransform(rotation, rest_offset)
    // Where rest_offset = RestJoints[i] - RestJoints[parent(i)]
    for (int32 i = 1; i < NumJoints; ++i)
    {
        const int32 Parent = Parents[i];
        const FVector RestOffset = RestJoints[i] - RestJoints[Parent];
        const FTransform Local(BodyPose[i - 1], RestOffset);
        OutJointWorlds[i] = Local * OutJointWorlds[Parent];
    }
}


void FSMPLModel::ComputeVertices(
    const FQuat& GlobalOrient,
    const TArray<FQuat>& BodyPose,
    const FVector& RootTrans,
    TArray<FVector>& OutVertices) const
{
    TArray<FTransform> JointWorlds;
    ComputeJointWorlds(GlobalOrient, BodyPose, RootTrans, JointWorlds);
    ComputeVerticesFromJoints(JointWorlds, OutVertices);
}


void FSMPLModel::ComputeVerticesFromJoints(
    const TArray<FTransform>& JointWorlds,
    TArray<FVector>& OutVertices) const
{
    if (NumJoints == 0 || NumVerts == 0)
    {
        OutVertices.Reset();
        return;
    }
    check(JointWorlds.Num() == NumJoints);

    // LBS: v_deformed = sum_j weights[v,j] * joint_world[j].TransformPosition(v_rest - joint_rest[j])
    OutVertices.SetNumUninitialized(NumVerts);
    for (int32 v = 0; v < NumVerts; ++v)
    {
        const FVector& RestPos = RestVerts[v];
        const int32 Start = InfluenceRowStart[v];
        const int32 End = InfluenceRowStart[v + 1];

        FVector Result = FVector::ZeroVector;
        for (int32 k = Start; k < End; ++k)
        {
            const int32 BoneIdx = InfluenceBones[k];
            const float W = InfluenceWeights[k];
            const FVector RestRelative = RestPos + RestJointsNeg[BoneIdx];
            const FVector Transformed = JointWorlds[BoneIdx].TransformPosition(RestRelative);
            Result += W * Transformed;
        }
        OutVertices[v] = Result;
    }
}
