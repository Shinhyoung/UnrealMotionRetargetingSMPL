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

    /**
     * Materials — one per mesh section (matches the source FBX's material slots).
     * Slot names are logged at load time; assign a material to each index in order.
     * Fallback: if empty or has fewer entries than sections, remaining sections
     * use the default material (usually gray).
     */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SMPL")
    TArray<UMaterialInterface*> Materials;

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

    /** Enable foot IK: lock a foot's world position when its vertical velocity is low. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SMPL|FootIK")
    bool bEnableFootIK = true;

    /** Foot planted when its smoothed speed (SMPL meters/sec) drops below this. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SMPL|FootIK", meta = (ClampMin = "0.01"))
    float FootPlantSpeed = 0.15f;

    /** Foot released when smoothed speed rises above this. Must be > FootPlantSpeed. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SMPL|FootIK", meta = (ClampMin = "0.01"))
    float FootReleaseSpeed = 0.4f;

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
    TArray<FTransform> JointWorlds;         // per-joint FK output, mutated by foot IK

    /** Per-material section triangles (indices into global vertex array). */
    TArray<TArray<int32>> SectionTriangles;

    /** Body pose scratch (23 quats). */
    TArray<FQuat> BodyPose;

    /** Per-foot planting state for foot IK. */
    struct FFootIKState
    {
        bool bPlanted = false;
        FVector LockedPos = FVector::ZeroVector;   // SMPL Y-up meters
        FVector LastPos = FVector::ZeroVector;
        float SmoothedSpeed = 0.0f;
        bool bHasLastPos = false;
    };
    FFootIKState LeftFootState;
    FFootIKState RightFootState;

    void EnsureMeshSectionInitialized();
    void UpdateMeshFromSmplVerts();
    /** Modifies JointWorlds in-place: locks ankle at planted position via two-bone IK. */
    void ApplyFootIK(float DeltaSeconds);

    /** Convert SMPL Y-up meters vector → UE Z-up cm. */
    static FVector SmplYupMetersToUEcm(const FVector& V);
};
