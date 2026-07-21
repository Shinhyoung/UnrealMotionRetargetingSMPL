// Copyright — see /LICENSE for details.

#include "RTMotionAnimInstance.h"
#include "MotionReceiverSubsystem.h"
#include "Engine/GameInstance.h"
#include "Engine/World.h"

void URTMotionAnimInstance::NativeUpdateAnimation(float DeltaSeconds)
{
	Super::NativeUpdateAnimation(DeltaSeconds);

	UWorld* World = GetWorld();
	UGameInstance* GameInstance = World ? World->GetGameInstance() : nullptr;
	if (!GameInstance)
	{
		bHasFreshData = false;
		return;
	}

	UMotionReceiverSubsystem* Subsystem = GameInstance->GetSubsystem<UMotionReceiverSubsystem>();
	if (!Subsystem)
	{
		bHasFreshData = false;
		return;
	}

	FPersonMotionData Data;
	if (Subsystem->GetLatestMotion(PersonId, Data) && Data.BoneRotations.Num() > 0)
	{
		LatestBoneRotations = Data.BoneRotations;
		LatestRootTranslation = Data.RootTranslation;
		bHasFreshData = true;
	}
	else
	{
		bHasFreshData = false;
	}
}
