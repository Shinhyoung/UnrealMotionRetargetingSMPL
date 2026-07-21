// Copyright — see /LICENSE for details.

#include "AnimNode_RTMotion.h"
#include "RTMotionAnimInstance.h"
#include "Animation/AnimInstanceProxy.h"

FAnimNode_RTMotion::FAnimNode_RTMotion()
{
	// SMPL has 24 joints. Pre-populate with UE5 Mannequin default names per
	// docs/bone_mapping.md so users only need to *verify* the mapping in the
	// AnimGraph rather than key in 24 dropdown selections. Swap names for a
	// custom skeleton before use.
	static const FName UEMannequinDefaults[24] =
	{
		TEXT("pelvis"),      // 0  SMPL pelvis
		TEXT("thigh_l"),     // 1  L_hip
		TEXT("thigh_r"),     // 2  R_hip
		TEXT("spine_01"),    // 3  spine1
		TEXT("calf_l"),      // 4  L_knee
		TEXT("calf_r"),      // 5  R_knee
		TEXT("spine_02"),    // 6  spine2
		TEXT("foot_l"),      // 7  L_ankle
		TEXT("foot_r"),      // 8  R_ankle
		TEXT("spine_03"),    // 9  spine3
		TEXT("ball_l"),      // 10 L_foot
		TEXT("ball_r"),      // 11 R_foot
		TEXT("neck_01"),     // 12 neck
		TEXT("clavicle_l"),  // 13 L_collar
		TEXT("clavicle_r"),  // 14 R_collar
		TEXT("head"),        // 15 head
		TEXT("upperarm_l"),  // 16 L_shoulder
		TEXT("upperarm_r"),  // 17 R_shoulder
		TEXT("lowerarm_l"),  // 18 L_elbow
		TEXT("lowerarm_r"),  // 19 R_elbow
		TEXT("hand_l"),      // 20 L_wrist
		TEXT("hand_r"),      // 21 R_wrist
		NAME_None,           // 22 L_hand — no UE mannequin counterpart, skip
		NAME_None,           // 23 R_hand — no UE mannequin counterpart, skip
	};

	SMPLBones.SetNum(24);
	for (int32 i = 0; i < 24; ++i)
	{
		SMPLBones[i].BoneName = UEMannequinDefaults[i];
	}
}

bool FAnimNode_RTMotion::IsValidToEvaluate(const USkeleton* Skeleton, const FBoneContainer& RequiredBones)
{
	if (SMPLBones.Num() == 0)
	{
		return false;
	}
	for (const FBoneReference& Bone : SMPLBones)
	{
		if (Bone.IsValidToEvaluate(RequiredBones))
		{
			return true;
		}
	}
	return false;
}

void FAnimNode_RTMotion::InitializeBoneReferences(const FBoneContainer& RequiredBones)
{
	for (FBoneReference& Bone : SMPLBones)
	{
		Bone.Initialize(RequiredBones);
	}
}

void FAnimNode_RTMotion::EvaluateSkeletalControl_AnyThread(
	FComponentSpacePoseContext& Output,
	TArray<FBoneTransform>& OutBoneTransforms)
{
	// Anim instance caches the latest rotations pulled from the subsystem.
	UObject* AnimObj = Output.AnimInstanceProxy->GetAnimInstanceObject();
	const URTMotionAnimInstance* RTInstance = Cast<URTMotionAnimInstance>(AnimObj);
	if (!RTInstance || !RTInstance->bHasFreshData)
	{
		return;
	}

	const TArray<FQuat>& Rotations = RTInstance->LatestBoneRotations;
	const int32 Count = FMath::Min(SMPLBones.Num(), Rotations.Num());
	const FBoneContainer& BoneContainer = Output.Pose.GetPose().GetBoneContainer();

	// Build the list of joints we can actually evaluate, then sort by compact
	// pose index ascending. That guarantees we process a parent BEFORE any of
	// its children — critical because we need to feed the child's ComponentSpace
	// re-projection with the *already-overridden* parent transform, not the
	// input pose's original one. The previous (unsorted, per-bone) approach
	// broke deep hierarchies like ``spine_03 → clavicle_l → upperarm_l``: each
	// arm bone was reconstructed against the untouched input spine and ended
	// up rotating around the wrong axes, which looked like "no reaction".
	struct FJointEntry
	{
		int32 JointIdx;
		FCompactPoseBoneIndex CompactIdx;
	};
	TArray<FJointEntry> ValidJoints;
	ValidJoints.Reserve(Count);
	for (int32 i = 0; i < Count; ++i)
	{
		const FBoneReference& BoneRef = SMPLBones[i];
		if (!BoneRef.IsValidToEvaluate(BoneContainer))
		{
			continue;
		}
		const FCompactPoseBoneIndex CompactIdx = BoneRef.GetCompactPoseIndex(BoneContainer);
		if (CompactIdx == INDEX_NONE)
		{
			continue;
		}
		ValidJoints.Add({i, CompactIdx});
	}
	ValidJoints.Sort([](const FJointEntry& A, const FJointEntry& B)
	{
		return A.CompactIdx.GetInt() < B.CompactIdx.GetInt();
	});

	// Parent-compact-idx → new component-space transform, keyed by raw int so
	// TMap doesn't fight with FCompactPoseBoneIndex's operator semantics.
	TMap<int32, FTransform> ProcessedComponent;

	for (const FJointEntry& Entry : ValidJoints)
	{
		const int32 JointIdx = Entry.JointIdx;
		const FCompactPoseBoneIndex CompactIdx = Entry.CompactIdx;

		FTransform LocalTransform = Output.Pose.GetLocalSpaceTransform(CompactIdx);
		LocalTransform.SetRotation(Rotations[JointIdx]);

		FTransform ParentComponent = FTransform::Identity;
		const FCompactPoseBoneIndex ParentIdx = BoneContainer.GetParentBoneIndex(CompactIdx);
		if (ParentIdx != INDEX_NONE)
		{
			if (const FTransform* Cached = ProcessedComponent.Find(ParentIdx.GetInt()))
			{
				ParentComponent = *Cached;
			}
			else
			{
				ParentComponent = Output.Pose.GetComponentSpaceTransform(ParentIdx);
			}
		}

		const FTransform ComponentTransform = LocalTransform * ParentComponent;
		ProcessedComponent.Add(CompactIdx.GetInt(), ComponentTransform);
		OutBoneTransforms.Add(FBoneTransform(CompactIdx, ComponentTransform));
	}

	// Base class expects sorted by compact index; we inserted in that order but
	// re-sort defensively in case of any future edit.
	OutBoneTransforms.Sort(FCompareBoneTransformIndex());
}
