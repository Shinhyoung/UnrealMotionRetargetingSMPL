// Copyright — see /LICENSE for details.

#pragma once

#include "CoreMinimal.h"
#include "BoneContainer.h"
#include "BoneControllers/AnimNode_SkeletalControlBase.h"
#include "AnimNode_RTMotion.generated.h"

/**
 * Anim node that overrides the LOCAL rotation of each mapped SMPL bone with
 * the latest quaternion received from :class:`UMotionReceiverSubsystem` via
 * :class:`URTMotionAnimInstance`.
 *
 * Rules (CLAUDE.md §4.6):
 *   - Evaluate on the worker thread (``_AnyThread`` methods).
 *   - Apply the incoming ``FQuat`` directly — never round-trip through Rotator.
 *   - Overrides are LOCAL (parent-relative). The base class handles the
 *     component-space fix-up when we push component-space transforms into
 *     ``OutBoneTransforms``.
 *
 * ``SMPLBones`` is 24 entries long, indexed by SMPL joint id 0..23 (see
 * ``python/config/bone_mapping.py``). Leave an entry's bone name empty to
 * skip it (useful for SMPL 22/23 which have no UE mannequin counterpart).
 */
USTRUCT(BlueprintInternalUseOnly)
struct MOTIONRETARGET_API FAnimNode_RTMotion : public FAnimNode_SkeletalControlBase
{
	GENERATED_BODY()

	/** SMPL joint id -> UE bone name mapping. Exactly 24 entries. */
	UPROPERTY(EditAnywhere, Category = "MotionRetarget", meta = (PinShownByDefault))
	TArray<FBoneReference> SMPLBones;

	FAnimNode_RTMotion();

	//~ Begin FAnimNode_SkeletalControlBase interface
	virtual void EvaluateSkeletalControl_AnyThread(
		FComponentSpacePoseContext& Output,
		TArray<FBoneTransform>& OutBoneTransforms) override;
	virtual bool IsValidToEvaluate(const USkeleton* Skeleton, const FBoneContainer& RequiredBones) override;
	virtual void InitializeBoneReferences(const FBoneContainer& RequiredBones) override;
	//~ End FAnimNode_SkeletalControlBase interface
};
