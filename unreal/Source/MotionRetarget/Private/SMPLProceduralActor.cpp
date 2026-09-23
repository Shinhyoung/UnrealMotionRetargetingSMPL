// Copyright — see /LICENSE for details.

#include "SMPLProceduralActor.h"

#include "MotionReceiverSubsystem.h"
#include "ProceduralMeshComponent.h"
#include "Engine/GameInstance.h"
#include "Engine/World.h"
#include "Materials/MaterialInterface.h"

DEFINE_LOG_CATEGORY_STATIC(LogSMPLActor, Log, All);


ASMPLProceduralActor::ASMPLProceduralActor()
{
    PrimaryActorTick.bCanEverTick = true;
    PrimaryActorTick.TickGroup = TG_PrePhysics;

    ProceduralMesh = CreateDefaultSubobject<UProceduralMeshComponent>(TEXT("ProceduralMesh"));
    ProceduralMesh->bUseAsyncCooking = false;
    ProceduralMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    RootComponent = ProceduralMesh;
}


void ASMPLProceduralActor::BeginPlay()
{
    Super::BeginPlay();

    if (!Model.LoadFromFile(SMPLBlobPath))
    {
        UE_LOG(LogSMPLActor, Error, TEXT("Failed to load SMPL blob from %s"), *SMPLBlobPath);
        return;
    }

    // Prepare body pose scratch (23 rotations for joints 1..23).
    BodyPose.SetNumZeroed(Model.GetNumJoints() - 1);
    for (int32 i = 0; i < BodyPose.Num(); ++i)
    {
        BodyPose[i] = FQuat::Identity;
    }

    // Precompute triangle indices per material section.
    // SmplYupMetersToUEcm uses chirality-preserving (det=+1) basis → swap
    // vertex 1 and 2 of every triangle so back-face culling keeps outside visible.
    const TArray<int32>& Faces = Model.GetFaces();
    const TArray<int32>& FaceMatIds = Model.GetFaceMaterialIds();
    const int32 NumMaterials = FMath::Max(1, Model.GetMaterialNames().Num());
    SectionTriangles.SetNum(NumMaterials);
    for (auto& S : SectionTriangles) { S.Reset(); }

    const int32 NumFaces = Faces.Num() / 3;
    for (int32 f = 0; f < NumFaces; ++f)
    {
        const int32 MatId = FaceMatIds.IsValidIndex(f)
            ? FMath::Clamp(FaceMatIds[f], 0, NumMaterials - 1) : 0;
        TArray<int32>& S = SectionTriangles[MatId];
        S.Add(Faces[f * 3 + 0]);
        S.Add(Faces[f * 3 + 2]);   // swapped
        S.Add(Faces[f * 3 + 1]);
    }

    // Also keep flat Triangles (used for normal recomputation across all faces).
    Triangles.SetNumUninitialized(Faces.Num());
    for (int32 t = 0; t < Faces.Num(); t += 3)
    {
        Triangles[t + 0] = Faces[t + 0];
        Triangles[t + 1] = Faces[t + 2];
        Triangles[t + 2] = Faces[t + 1];
    }

    EnsureMeshSectionInitialized();
    bReady = true;
    UE_LOG(LogSMPLActor, Display, TEXT("SMPL Procedural Actor ready (person id=%d)"), PersonId);
}


FVector ASMPLProceduralActor::SmplYupMetersToUEcm(const FVector& V)
{
    // Chirality-preserving basis (det = +1):
    //   ue.x = -smpl.z
    //   ue.y = -smpl.x      (flipped from +smpl.x)
    //   ue.z = +smpl.y
    // A det=-1 reflection (previous +smpl.x) displays the mesh as a MIRROR
    // image, which physically inverts the perceived rotation direction —
    // user turning left → mannequin turning right. Flipping Y sign turns
    // M into a proper rotation so user's yaw matches the mannequin's yaw.
    // Trade-off: L/R body parts now appear on the same side as the user
    // (like a real twin, not a mirror image).
    return FVector(-V.Z, -V.X, V.Y) * 100.0f;
}


void ASMPLProceduralActor::EnsureMeshSectionInitialized()
{
    if (!ProceduralMesh || Model.GetNumVertices() == 0)
    {
        return;
    }

    const int32 N = Model.GetNumVertices();

    // Rest vertices from model (SMPL Y-up) → UE cm.
    UEVerts.SetNumUninitialized(N);
    Normals.SetNumZeroed(N);              // filled once we have vertices
    VertexColors.Init(FLinearColor::White, N);
    Tangents.Empty();

    // UV0: from blob (v2) or zeros (v1). Copy once — static across ticks.
    const TArray<FVector2D>& ModelUV = Model.GetUV0();
    if (ModelUV.Num() == N)
    {
        UV0 = ModelUV;
    }
    else
    {
        UV0.SetNumZeroed(N);
    }

    // Initial pose = identity: compute rest verts.
    Model.ComputeVertices(FQuat::Identity, BodyPose, FVector::ZeroVector, SmplVertsYup);
    for (int32 i = 0; i < N; ++i)
    {
        UEVerts[i] = SmplYupMetersToUEcm(SmplVertsYup[i]);
    }

    // Compute flat normals (per-triangle averaged into vertices).
    for (int32 t = 0; t < Triangles.Num(); t += 3)
    {
        const int32 A = Triangles[t];
        const int32 B = Triangles[t + 1];
        const int32 C = Triangles[t + 2];
        const FVector Edge1 = UEVerts[B] - UEVerts[A];
        const FVector Edge2 = UEVerts[C] - UEVerts[A];
        FVector Nrm = FVector::CrossProduct(Edge1, Edge2);
        Nrm.Normalize();
        Normals[A] += Nrm;
        Normals[B] += Nrm;
        Normals[C] += Nrm;
    }
    for (FVector& Nv : Normals)
    {
        Nv.Normalize();
    }

    // One mesh section per material. Shared vertex array (wastes a bit of
    // memory but avoids per-section vertex remapping every tick).
    const TArray<FString>& MatNames = Model.GetMaterialNames();
    UE_LOG(LogSMPLActor, Display, TEXT("SMPL materials (%d):"), SectionTriangles.Num());
    for (int32 s = 0; s < SectionTriangles.Num(); ++s)
    {
        const FString Name = MatNames.IsValidIndex(s) ? MatNames[s] : FString::Printf(TEXT("slot_%d"), s);
        UE_LOG(LogSMPLActor, Display, TEXT("  [%d] '%s' (%d tris)"), s, *Name, SectionTriangles[s].Num() / 3);
        ProceduralMesh->CreateMeshSection_LinearColor(
            s, UEVerts, SectionTriangles[s], Normals, UV0, VertexColors, Tangents, false);
        if (Materials.IsValidIndex(s) && Materials[s])
        {
            ProceduralMesh->SetMaterial(s, Materials[s]);
        }
    }
}


void ASMPLProceduralActor::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);

    if (!bReady)
    {
        return;
    }

    UWorld* World = GetWorld();
    UGameInstance* GI = World ? World->GetGameInstance() : nullptr;
    if (!GI)
    {
        return;
    }
    UMotionReceiverSubsystem* Subsystem = GI->GetSubsystem<UMotionReceiverSubsystem>();
    if (!Subsystem)
    {
        return;
    }

    FPersonMotionData Data;
    if (!Subsystem->GetLatestMotion(PersonId, Data) || Data.BoneRotations.Num() < Model.GetNumJoints())
    {
        return;
    }

    // In --smpl-native mode:
    //   Data.BoneRotations : quats in SMPL Y-up (raw axis-angle → quat, no basis change).
    //   Length matches Model.GetNumJoints() (24 for SMPL, 55 for SMPL-X).
    //   Data.RootTranslation : SMPL Y-up meters (no basis change, no *100)
    const FQuat& GlobalOrient = Data.BoneRotations[0];
    for (int32 i = 0; i < BodyPose.Num(); ++i)
    {
        BodyPose[i] = Data.BoneRotations[i + 1];
    }
    const FVector RootTrans = Data.RootTranslation;    // SMPL Y-up meters

    // FK → (optional) foot IK → LBS. Splitting the phases lets ApplyFootIK
    // override hip/knee/ankle transforms so LBS renders a planted foot.
    Model.ComputeJointWorlds(GlobalOrient, BodyPose, RootTrans, JointWorlds);
    if (bEnableFootIK)
    {
        ApplyFootIK(DeltaSeconds);
    }
    Model.ComputeVerticesFromJoints(JointWorlds, SmplVertsYup);

    // Convert to UE Z-up cm.
    const int32 N = SmplVertsYup.Num();
    if (UEVerts.Num() != N)
    {
        UEVerts.SetNumUninitialized(N);
    }
    // Mirror root position in UE world without touching mesh geometry.
    // A uniform per-vertex shift is equivalent to mirroring the root across
    // one axis: local mesh offsets between vertices are preserved.
    //   X shift = 2*RootTrans.Z*100 → mannequin's world X flips sign (depth).
    //   Y shift = 2*RootTrans.X*100 → mannequin's world Y flips sign (L/R).
    const float RootXShiftCm = bInvertRootDepth ? (2.0f * RootTrans.Z * 100.0f) : 0.0f;
    const float RootYShiftCm = bInvertRootLR    ? (2.0f * RootTrans.X * 100.0f) : 0.0f;
    float MinZ = TNumericLimits<float>::Max();
    for (int32 i = 0; i < N; ++i)
    {
        UEVerts[i] = SmplYupMetersToUEcm(SmplVertsYup[i]);
        UEVerts[i].X += RootXShiftCm;
        UEVerts[i].Y += RootYShiftCm;
        if (UEVerts[i].Z < MinZ) { MinZ = UEVerts[i].Z; }
    }
    // ponytail: per-frame ground snap. Kills jump/crouch airborne motion;
    // switch to first-frame-only calibration when someone wants those.
    if (bGroundSnap && N > 0)
    {
        for (int32 i = 0; i < N; ++i) { UEVerts[i].Z -= MinZ; }
    }

    UpdateMeshFromSmplVerts();
}


void ASMPLProceduralActor::UpdateMeshFromSmplVerts()
{
    // Recompute normals (flat-per-vertex from adjacent triangles).
    const int32 N = UEVerts.Num();
    for (int32 i = 0; i < N; ++i)
    {
        Normals[i] = FVector::ZeroVector;
    }
    for (int32 t = 0; t < Triangles.Num(); t += 3)
    {
        const int32 A = Triangles[t];
        const int32 B = Triangles[t + 1];
        const int32 C = Triangles[t + 2];
        const FVector Edge1 = UEVerts[B] - UEVerts[A];
        const FVector Edge2 = UEVerts[C] - UEVerts[A];
        FVector Nrm = FVector::CrossProduct(Edge1, Edge2);
        Nrm.Normalize();
        Normals[A] += Nrm;
        Normals[B] += Nrm;
        Normals[C] += Nrm;
    }
    for (FVector& Nv : Normals)
    {
        Nv.Normalize();
    }

    // Update every material section with the same vertex data.
    for (int32 s = 0; s < SectionTriangles.Num(); ++s)
    {
        ProceduralMesh->UpdateMeshSection_LinearColor(
            s, UEVerts, Normals, UV0, VertexColors, Tangents);
    }
}


// SMPL joint indices: 1=L_hip, 4=L_knee, 7=L_ankle, 10=L_foot,
//                     2=R_hip, 5=R_knee, 8=R_ankle, 11=R_foot.
static void SolveLegIK(
    TArray<FTransform>& Joints,
    int32 HipIdx, int32 KneeIdx, int32 AnkleIdx, int32 FootIdx,
    const FVector& Target)
{
    const FVector H = Joints[HipIdx].GetLocation();
    const FVector K_orig = Joints[KneeIdx].GetLocation();
    const FVector A_orig = Joints[AnkleIdx].GetLocation();

    const float L1 = FVector::Dist(H, K_orig);
    const float L2 = FVector::Dist(K_orig, A_orig);
    if (L1 < 1e-4f || L2 < 1e-4f)
    {
        return;
    }

    FVector HT = Target - H;
    float D = HT.Size();
    if (D < 1e-4f)
    {
        return;
    }
    // Clamp target so the leg can reach without singularity.
    const float DMax = L1 + L2 - 1e-3f;
    if (D > DMax)
    {
        HT *= (DMax / D);
        D = DMax;
    }
    const FVector Dir = HT / D;

    // Preserve the current bend direction: knee's projection onto plane
    // perpendicular to Dir stays on the same side of Dir.
    const FVector KneeDirOrig = (K_orig - H).GetSafeNormal();
    FVector BendAxis = FVector::CrossProduct(Dir, KneeDirOrig);
    if (!BendAxis.Normalize())
    {
        // Straight leg — no natural bend axis. Just place ankle at target and bail.
        Joints[AnkleIdx].SetLocation(Target);
        const FVector Delta = Target - A_orig;
        Joints[FootIdx].SetLocation(Joints[FootIdx].GetLocation() + Delta);
        return;
    }

    // Law of cosines: angle at hip between hip→target and hip→knee.
    float CosAlpha = (L1 * L1 + D * D - L2 * L2) / (2.0f * L1 * D);
    CosAlpha = FMath::Clamp(CosAlpha, -1.0f, 1.0f);
    const float Alpha = FMath::Acos(CosAlpha);

    const FQuat RotAtHip(BendAxis, Alpha);
    const FVector KneeDir = RotAtHip.RotateVector(Dir);
    const FVector K_new = H + KneeDir * L1;

    // Update hip via delta rotation (preserves original twist / roll).
    const FVector OldKneeDirFromHip = (K_orig - H).GetSafeNormal();
    const FQuat HipDelta = FQuat::FindBetweenNormals(OldKneeDirFromHip, KneeDir);
    Joints[HipIdx].SetRotation(HipDelta * Joints[HipIdx].GetRotation());
    // Hip location unchanged (fixed by parent chain).

    // Update knee: new location, delta rotation from original ankle direction.
    Joints[KneeIdx].SetLocation(K_new);
    const FVector OldAnkleDirFromKnee = (A_orig - K_orig).GetSafeNormal();
    const FVector NewAnkleDirFromKnee = (Target - K_new).GetSafeNormal();
    const FQuat KneeDelta = FQuat::FindBetweenNormals(OldAnkleDirFromKnee, NewAnkleDirFromKnee);
    Joints[KneeIdx].SetRotation(KneeDelta * Joints[KneeIdx].GetRotation());

    // Snap ankle to target; shift foot by ankle delta (rotation preserved).
    const FVector AnkleDelta = Target - A_orig;
    Joints[AnkleIdx].SetLocation(Target);
    Joints[FootIdx].SetLocation(Joints[FootIdx].GetLocation() + AnkleDelta);
}


void ASMPLProceduralActor::ApplyFootIK(float DeltaSeconds)
{
    if (JointWorlds.Num() < 12)
    {
        return;
    }

    // Per-foot contact update + IK solve. Speed in SMPL Y-up meters/sec.
    auto Update = [&](FFootIKState& State, int32 HipIdx, int32 KneeIdx, int32 AnkleIdx, int32 FootIdx)
    {
        const FVector Cur = JointWorlds[AnkleIdx].GetLocation();

        float Speed = 0.0f;
        if (State.bHasLastPos && DeltaSeconds > 1e-4f)
        {
            Speed = FVector::Dist(Cur, State.LastPos) / DeltaSeconds;
        }
        // EMA smoothing so a single fast frame doesn't release the lock.
        State.SmoothedSpeed = 0.6f * State.SmoothedSpeed + 0.4f * Speed;
        State.LastPos = Cur;
        State.bHasLastPos = true;

        if (State.bPlanted)
        {
            if (State.SmoothedSpeed > FootReleaseSpeed)
            {
                State.bPlanted = false;
            }
        }
        else
        {
            if (State.SmoothedSpeed < FootPlantSpeed)
            {
                State.bPlanted = true;
                State.LockedPos = Cur;
            }
        }

        if (State.bPlanted)
        {
            SolveLegIK(JointWorlds, HipIdx, KneeIdx, AnkleIdx, FootIdx, State.LockedPos);
        }
    };

    Update(LeftFootState, 1, 4, 7, 10);
    Update(RightFootState, 2, 5, 8, 11);
}
