// Copyright — see /LICENSE for details.

#pragma once

#include "CoreMinimal.h"
#include "BoneContainer.h"
#include "BoneControllers/AnimNode_SkeletalControlBase.h"
#include "AnimNode_RTMotionRetarget.generated.h"

/**
 * Chain-based FK retargeting AnimNode (Option B).
 *
 * Unlike ``FAnimNode_RTMotion`` which overwrites each UE bone's local rotation
 * with the raw SMPL local rotation (fails when SMPL joint local frames don't
 * align with UE bone local frames), this node does full FK retargeting:
 *
 *   1. Compute source (SMPL) chain FK in SMPL Y-up world space using the
 *      compile-time SMPL rest joint positions (from SMPLRestData.h) plus
 *      the incoming per-joint local rotations.
 *   2. Basis-change each source world orientation to UE Z-up.
 *   3. For each mapped UE bone, compute the retargeted local rotation
 *      such that its new world orientation equals
 *      ``source_world_ue_basis[i] · ue_rest_world[i]`` (SMPL delta applied
 *      to UE rest orientation).
 *
 * This matches the formula UE's IK Retargeter uses for FK retargeting, but
 * runs entirely inside our AnimNode — no IK Rig / IK Retargeter assets
 * needed. Works directly on SKM_Manny.
 *
 * The SMPL rest world orientations are identity (SMPL canonical rest is
 * axis-aligned), so ``source_delta = source_current`` in world space.
 */
USTRUCT(BlueprintInternalUseOnly)
struct MOTIONRETARGET_API FAnimNode_RTMotionRetarget : public FAnimNode_SkeletalControlBase
{
    GENERATED_BODY()

    /**
     * UE bone name for each SMPL joint id (0..23). Pre-populated with the
     * UE5 Mannequin defaults; override entries only for a custom skeleton.
     * Leave a slot empty to skip that SMPL joint (e.g. hand slots 22/23).
     */
    UPROPERTY(EditAnywhere, Category = "MotionRetarget")
    TArray<FBoneReference> SMPLBones;

    FAnimNode_RTMotionRetarget();

    //~ Begin FAnimNode_SkeletalControlBase interface
    virtual void EvaluateSkeletalControl_AnyThread(
        FComponentSpacePoseContext& Output,
        TArray<FBoneTransform>& OutBoneTransforms) override;
    virtual bool IsValidToEvaluate(const USkeleton* Skeleton, const FBoneContainer& RequiredBones) override;
    virtual void InitializeBoneReferences(const FBoneContainer& RequiredBones) override;
    //~ End FAnimNode_SkeletalControlBase interface
};
