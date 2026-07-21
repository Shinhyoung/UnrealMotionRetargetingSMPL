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

    // Precompute triangle indices (static across ticks).
    // SmplYupMetersToUEcm uses a chirality-preserving (det=+1) basis, which
    // flips face winding relative to SMPL's original mesh. Swap vertices 1 and
    // 2 of every triangle so UE's back-face culling keeps the outside visible.
    const TArray<int32>& Faces = Model.GetFaces();
    Triangles.SetNumUninitialized(Faces.Num());
    for (int32 t = 0; t < Faces.Num(); t += 3)
    {
        Triangles[t + 0] = Faces[t + 0];
        Triangles[t + 1] = Faces[t + 2];   // swapped
        Triangles[t + 2] = Faces[t + 1];   // swapped
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
    UV0.SetNumZeroed(N);
    VertexColors.Init(FLinearColor::White, N);
    Tangents.Empty();

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

    ProceduralMesh->CreateMeshSection_LinearColor(
        0, UEVerts, Triangles, Normals, UV0, VertexColors, Tangents, false /*create collision*/);

    if (MeshMaterial)
    {
        ProceduralMesh->SetMaterial(0, MeshMaterial);
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
    if (!Subsystem->GetLatestMotion(PersonId, Data) || Data.BoneRotations.Num() < 24)
    {
        return;
    }

    // In --smpl-native mode:
    //   Data.BoneRotations : 24 quats in SMPL Y-up (raw axis-angle → quat, no basis change)
    //   Data.RootTranslation : SMPL Y-up meters (no basis change, no *100)
    const FQuat& GlobalOrient = Data.BoneRotations[0];
    for (int32 i = 0; i < BodyPose.Num(); ++i)
    {
        BodyPose[i] = Data.BoneRotations[i + 1];
    }
    const FVector RootTrans = Data.RootTranslation;    // SMPL Y-up meters

    // Compute vertices in SMPL Y-up.
    Model.ComputeVertices(GlobalOrient, BodyPose, RootTrans, SmplVertsYup);

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
    for (int32 i = 0; i < N; ++i)
    {
        UEVerts[i] = SmplYupMetersToUEcm(SmplVertsYup[i]);
        UEVerts[i].X += RootXShiftCm;
        UEVerts[i].Y += RootYShiftCm;
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

    ProceduralMesh->UpdateMeshSection_LinearColor(
        0, UEVerts, Normals, UV0, VertexColors, Tangents);
}
