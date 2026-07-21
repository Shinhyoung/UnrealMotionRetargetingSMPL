// Copyright — see /LICENSE for details.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ProceduralMeshComponent.h"           // FProcMeshTangent, UProceduralMeshComponent
#include "SMPLModel.h"
#include "SMPLProceduralActor.generated.h"

class UMaterialInterface;

/**
 * Renders a real-time SMPL mesh driven by UDP pose data.
 *
 * Bypasses UE's Skeletal system entirely — we run SMPL Linear Blend Skinning
 * on the CPU each tick using :class:`FSMPLModel` and push the resulting
 * vertex positions into a UProceduralMeshComponent. This gives an exact
 * reproduction of the python-side SMPL forward output, which is what the
 * mesh overlay (SAT-HMR debug) shows the user during capture.
 *
 * Consumes ``UMotionReceiverSubsystem::GetLatestMotion(PersonId)`` for its
 * pose data. Python side must send with ``--smpl-native`` so quaternions
 * are raw SMPL Y-up (no basis change).
 */
UCLASS(Blueprintable)
class MOTIONRETARGET_API ASMPLProceduralActor : public AActor
{
    GENERATED_BODY()

public:
    ASMPLProceduralActor();

    /** Which python-side person feeds this actor. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SMPL")
    int32 PersonId = 1;

    /** Path to smpl_model.bin (see ``python/tools/dump_smpl_blob.py``). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SMPL")
    FString SMPLBlobPath = TEXT("c:/0.shinhyoung/Project/1.Retargeting/MRetargeting/smpl_model.bin");

    /** Optional material to apply to the rendered mesh section. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SMPL")
    UMaterialInterface* MeshMaterial = nullptr;

    /**
     * Mirror the root position across the YZ-plane in UE world so the mannequin
     * moves TOWARD the viewport camera when the user walks TOWARD the real
     * camera. Necessary when the UE viewport looks in the -X → +X direction
     * (the default level camera in most setups). Local mesh geometry is left
     * untouched — only the world position is mirrored.
     */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SMPL")
    bool bInvertRootDepth = true;

    /**
     * Mirror the root position across the XZ-plane in UE world. When the user
     * side-steps to their own right, the mannequin moves toward the viewer's
     * right of screen. Independent of mesh chirality — only affects where the
     * mannequin is placed in world, not how the body looks internally.
     */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SMPL")
    bool bInvertRootLR = true;

    /** ProceduralMeshComponent that owns the SMPL mesh. */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SMPL")
    UProceduralMeshComponent* ProceduralMesh = nullptr;

    //~ Begin AActor interface
    virtual void BeginPlay() override;
    virtual void Tick(float DeltaSeconds) override;
    //~ End AActor interface

private:
    /** Loaded SMPL model. */
    FSMPLModel Model;

    /** True once model is loaded and mesh section is created. */
    bool bReady = false;

    /** Scratch buffers reused each tick. */
    TArray<FVector> SmplVertsYup;           // model output (SMPL Y-up meters)
    TArray<FVector> UEVerts;                // basis-changed to UE cm
    TArray<FVector> Normals;                // recomputed each tick (flat for now)
    TArray<int32> Triangles;                // static after init
    TArray<FVector2D> UV0;                  // dummy zero UVs
    TArray<FLinearColor> VertexColors;      // white
    TArray<FProcMeshTangent> Tangents;      // empty

    /** Body pose scratch (23 quats). */
    TArray<FQuat> BodyPose;

    void EnsureMeshSectionInitialized();
    void UpdateMeshFromSmplVerts();

    /** Convert SMPL Y-up meters vector → UE Z-up cm. */
    static FVector SmplYupMetersToUEcm(const FVector& V);
};
