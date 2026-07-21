// Copyright — see /LICENSE for details.

#pragma once

#include "CoreMinimal.h"
#include "Animation/AnimInstance.h"
#include "RTMotionAnimInstance.generated.h"

/**
 * Anim instance that pulls the latest joint rotations for a given ``PersonId``
 * from :class:`UMotionReceiverSubsystem` on the game thread and caches them.
 * ``FAnimNode_RTMotion`` reads the cache on the worker thread (CLAUDE.md §7).
 *
 * The pattern (write on game thread, read on _AnyThread) is safe because we
 * copy into the anim node under the anim graph's normal proxy sync — the
 * animation system snapshots UAnimInstance UPROPERTYs before dispatching the
 * anim graph, so worker reads see a consistent frame.
 */
UCLASS(Blueprintable, BlueprintType)
class MOTIONRETARGET_API URTMotionAnimInstance : public UAnimInstance
{
	GENERATED_BODY()

public:
	/** Which python-side person feeds this avatar. Set from BP/gameplay. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MotionRetarget")
	int32 PersonId = 0;

	/** Latest 24 SMPL joint local rotations (xyzw ``FQuat``). Empty when no data. */
	UPROPERTY(Transient, BlueprintReadOnly, Category = "MotionRetarget")
	TArray<FQuat> LatestBoneRotations;

	/** Latest root translation (UE cm). */
	UPROPERTY(Transient, BlueprintReadOnly, Category = "MotionRetarget")
	FVector LatestRootTranslation = FVector::ZeroVector;

	/** True this tick if the subsystem returned fresh data. */
	UPROPERTY(Transient, BlueprintReadOnly, Category = "MotionRetarget")
	bool bHasFreshData = false;

	//~ Begin UAnimInstance interface
	virtual void NativeUpdateAnimation(float DeltaSeconds) override;
	//~ End UAnimInstance interface
};
