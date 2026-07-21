// Copyright — see /LICENSE for details.

#include "AnimNode_RTMotionRetarget.h"

#include "RTMotionAnimInstance.h"
#include "SMPLRestData.h"
#include "Animation/AnimInstanceProxy.h"

/*
 * Chain-based swing + twist retargeting (option 2).
 *
 * The per-joint FK retargeting (source_delta · target_rest_world) fails when
 * target bones have non-identity rest world orient — the matrix multiply
 * mixes the delta's axes with the rest's axes, producing pitch↔roll swap.
 * UE mannequin's every bone has ~90° rest yaw, so every joint mis-maps.
 *
 * Fix: work with bone DIRECTIONS instead of raw quaternions.
 *
 *   For each joint:
 *     source_rest_dir  = normalize(SMPL_child_pos - SMPL_self_pos)  in UE basis
 *     source_current_dir = source_world_rotation · source_rest_dir  (via FK)
 *     target_rest_dir  = normalize(UE_child_pos - UE_self_pos)  from mannequin
 *
 *   Compute swing (rotation that aligns source_rest_dir to source_current_dir):
 *     axis  = source_rest_dir × source_current_dir  (normalize)
 *     angle = acos(source_rest_dir · source_current_dir)
 *     q_source_swing = quat(axis, angle)
 *
 *   Apply source_swing to target's rest direction to get where target should point:
 *     target_new_dir = q_source_swing · target_rest_dir
 *
 *   Compute target's swing (rotation from target rest dir to new dir):
 *     q_target_swing = quat_between(target_rest_dir, target_new_dir)
 *
 *   Optionally extract source twist (rotation about source_rest_dir) and
 *   re-apply about target's NEW direction.
 *
 *   Compose final: target_new_world = q_target_swing · target_rest_world
 *                (twist added as post-multiply about new dir)
 *
 * Bones with no chain child (feet, head, hand tips) fall back to direct
 * component-space rotation from the parent's processed transform.
 */

namespace
{
    /** Signed angle rotation from unit vector A to unit vector B. */
    FQuat QuatBetweenVectors(const FVector& A, const FVector& B)
    {
        const float Dot = FMath::Clamp<float>(FVector::DotProduct(A, B), -1.0f, 1.0f);
        if (Dot > 0.9999f)
        {
            return FQuat::Identity;
        }
        if (Dot < -0.9999f)
        {
            // 180° rotation — pick any orthogonal axis.
            FVector Orth = FVector::CrossProduct(A, FVector::UpVector);
            if (Orth.SizeSquared() < 1e-6f)
            {
                Orth = FVector::CrossProduct(A, FVector::ForwardVector);
            }
            Orth.Normalize();
            return FQuat(Orth, PI);
        }
        FVector Axis = FVector::CrossProduct(A, B);
        Axis.Normalize();
        const float Angle = FMath::Acos(Dot);
        return FQuat(Axis, Angle);
    }

    /** Convert SMPL Y-up meters position to UE Z-up centimeters vector. */
    FVector SmplPosToUE(float X, float Y, float Z)
    {
        // Basis mapping (see transform/coordinate.py):
        //   ue.x = -smpl.z
        //   ue.y = +smpl.x
        //   ue.z = +smpl.y
        // Positions get *100 for cm, but for DIRECTION vectors (which is what
        // we use here) scale doesn't matter — we normalize.
        return FVector(-Z, X, Y);
    }
}


FAnimNode_RTMotionRetarget::FAnimNode_RTMotionRetarget()
{
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
        NAME_None,           // 22 L_hand
        NAME_None,           // 23 R_hand
    };

    SMPLBones.SetNum(24);
    for (int32 i = 0; i < 24; ++i)
    {
        SMPLBones[i].BoneName = UEMannequinDefaults[i];
    }
}

bool FAnimNode_RTMotionRetarget::IsValidToEvaluate(const USkeleton* Skeleton, const FBoneContainer& RequiredBones)
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

void FAnimNode_RTMotionRetarget::InitializeBoneReferences(const FBoneContainer& RequiredBones)
{
    for (FBoneReference& Bone : SMPLBones)
    {
        Bone.Initialize(RequiredBones);
    }
}

void FAnimNode_RTMotionRetarget::EvaluateSkeletalControl_AnyThread(
    FComponentSpacePoseContext& Output,
    TArray<FBoneTransform>& OutBoneTransforms)
{
    UObject* AnimObj = Output.AnimInstanceProxy->GetAnimInstanceObject();
    const URTMotionAnimInstance* RTInstance = Cast<URTMotionAnimInstance>(AnimObj);
    if (!RTInstance || !RTInstance->bHasFreshData)
    {
        return;
    }

    const TArray<FQuat>& LocalRotations = RTInstance->LatestBoneRotations;
    if (LocalRotations.Num() < SMPLRest::NumJoints)
    {
        return;
    }

    // Step 1: SMPL FK — accumulate world rotations.
    TStaticArray<FQuat, SMPLRest::NumJoints> SourceWorldRotUE;
    for (int32 i = 0; i < SMPLRest::NumJoints; ++i)
    {
        const int32 Parent = SMPLRest::Parents[i];
        if (Parent < 0)
        {
            SourceWorldRotUE[i] = LocalRotations[i];
        }
        else
        {
            SourceWorldRotUE[i] = SourceWorldRotUE[Parent] * LocalRotations[i];
        }
        SourceWorldRotUE[i].Normalize();
    }

    // Step 2: pre-compute SMPL rest bone directions (in UE basis).
    // dir[i] = normalize(child_pos - self_pos) — the axis the bone points
    // along at rest. For leaves we leave a zero vector; those bones skip
    // swing retargeting and use pure orientation copy.
    TStaticArray<FVector, SMPLRest::NumJoints> SourceRestDirUE;
    for (int32 i = 0; i < SMPLRest::NumJoints; ++i)
    {
        const int32 Child = SMPLRest::ChainChild[i];
        if (Child < 0)
        {
            SourceRestDirUE[i] = FVector::ZeroVector;
            continue;
        }
        const FVector Self  = SmplPosToUE(SMPLRest::RestPositionsYupMeters[i][0],
                                          SMPLRest::RestPositionsYupMeters[i][1],
                                          SMPLRest::RestPositionsYupMeters[i][2]);
        const FVector Child_ = SmplPosToUE(SMPLRest::RestPositionsYupMeters[Child][0],
                                           SMPLRest::RestPositionsYupMeters[Child][1],
                                           SMPLRest::RestPositionsYupMeters[Child][2]);
        FVector Dir = Child_ - Self;
        if (!Dir.Normalize())
        {
            Dir = FVector::ZeroVector;
        }
        SourceRestDirUE[i] = Dir;
    }

    // Step 3: gather valid target joints and sort parent-first.
    struct FJointEntry
    {
        int32 JointIdx;
        FCompactPoseBoneIndex CompactIdx;
    };

    const FBoneContainer& BoneContainer = Output.Pose.GetPose().GetBoneContainer();
    TArray<FJointEntry> ValidJoints;
    ValidJoints.Reserve(SMPLRest::NumJoints);
    const int32 SlotCount = FMath::Min<int32>(SMPLRest::NumJoints, SMPLBones.Num());
    for (int32 i = 0; i < SlotCount; ++i)
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

    // Step 4: for each joint, compute retargeted component-space transform.
    TMap<int32, FTransform> ProcessedComponent;

    for (const FJointEntry& Entry : ValidJoints)
    {
        const int32 JointIdx = Entry.JointIdx;
        const FCompactPoseBoneIndex CompactIdx = Entry.CompactIdx;

        // Parent's PROCESSED component transform (if any).
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

        const FTransform TargetRestComponent = Output.Pose.GetComponentSpaceTransform(CompactIdx);
        const FQuat TargetRestWorldQuat = TargetRestComponent.GetRotation();

        // Find target's child bone in UE skeleton (use SMPL chain child's
        // mapped UE bone if available) for target rest direction computation.
        FVector TargetRestDir = FVector::ZeroVector;
        const int32 SmplChild = SMPLRest::ChainChild[JointIdx];
        if (SmplChild >= 0 && SmplChild < SMPLBones.Num())
        {
            const FBoneReference& ChildBoneRef = SMPLBones[SmplChild];
            if (ChildBoneRef.IsValidToEvaluate(BoneContainer))
            {
                const FCompactPoseBoneIndex ChildCompact = ChildBoneRef.GetCompactPoseIndex(BoneContainer);
                if (ChildCompact != INDEX_NONE)
                {
                    const FTransform ChildRestComp = Output.Pose.GetComponentSpaceTransform(ChildCompact);
                    FVector Dir = ChildRestComp.GetLocation() - TargetRestComponent.GetLocation();
                    if (Dir.Normalize())
                    {
                        TargetRestDir = Dir;
                    }
                }
            }
        }

        FQuat TargetNewWorldQuat;

        if (!SourceRestDirUE[JointIdx].IsNearlyZero() && !TargetRestDir.IsNearlyZero())
        {
            // -------- Swing + Twist retargeting --------
            const FVector SrcRestDir = SourceRestDirUE[JointIdx];
            const FVector SrcCurDir = SourceWorldRotUE[JointIdx].RotateVector(SrcRestDir);

            // Source swing: aligns SrcRestDir → SrcCurDir.
            const FQuat SourceSwing = QuatBetweenVectors(SrcRestDir, SrcCurDir);

            // Apply source swing to target's rest direction to find where
            // target should point.
            const FVector TargetNewDir = SourceSwing.RotateVector(TargetRestDir);

            // Target swing: aligns TargetRestDir → TargetNewDir.
            const FQuat TargetSwing = QuatBetweenVectors(TargetRestDir, TargetNewDir);

            // Extract source twist about SrcRestDir (rotation about bone axis).
            // q_twist = q_source * inv(q_source_swing)
            const FQuat SourceTwistWorld = SourceWorldRotUE[JointIdx] * SourceSwing.Inverse();
            // Project this rotation onto the current bone axis to isolate twist.
            // Alternatively use Q's vector part projected on the axis.
            FVector TwistAxisComp = FVector::DotProduct(SourceTwistWorld.GetRotationAxis(),
                                                        SrcCurDir) * SrcCurDir;
            float TwistAngle = SourceTwistWorld.GetAngle();
            // Reconstruct twist about the source's current direction
            const FQuat SourceTwistPure = FQuat(SrcCurDir,
                                                FVector::DotProduct(SourceTwistWorld.GetRotationAxis(), SrcCurDir) >= 0.0f
                                                  ? TwistAngle : -TwistAngle);

            // Apply the same twist about target's new bone direction.
            const FQuat TargetTwist = FQuat(TargetNewDir, SourceTwistPure.GetAngle() *
                                            (FVector::DotProduct(SourceTwistWorld.GetRotationAxis(), SrcCurDir) >= 0.0f ? 1.0f : -1.0f));

            // Compose: apply swing first (align direction), then twist (rotate about new dir), then target rest.
            TargetNewWorldQuat = TargetTwist * TargetSwing * TargetRestWorldQuat;
        }
        else
        {
            // Leaf or degenerate — fall back to direct FK apply.
            TargetNewWorldQuat = SourceWorldRotUE[JointIdx] * TargetRestWorldQuat;
        }
        TargetNewWorldQuat.Normalize();

        // Preserve LOCAL translation so the child follows the (possibly
        // retargeted) parent correctly.
        //
        // UE FTransform::operator* is "apply A first, then B" (opposite of
        // FQuat::operator* which is "apply B first, then A"). So the FQuat
        // result of `ChildLocalRest * ParentComponent` (FTransform) is
        // effectively `parent_rot * child_local_rot` (FQuat), which physically
        // applies child_local first in bone's own frame then parent's world.
        //
        // To make the composed component_rot equal TargetNewWorldQuat:
        //   parent_rot * child_local_rot = target  (FQuat op, apply right first)
        //   child_local_rot = parent_rot^-1 * target
        FTransform ChildLocalRest = Output.Pose.GetLocalSpaceTransform(CompactIdx);
        const FQuat NewLocalRotation = ParentComponent.GetRotation().Inverse() * TargetNewWorldQuat;
        ChildLocalRest.SetRotation(NewLocalRotation);

        const FTransform NewComponent = ChildLocalRest * ParentComponent;
        ProcessedComponent.Add(CompactIdx.GetInt(), NewComponent);
        OutBoneTransforms.Add(FBoneTransform(CompactIdx, NewComponent));
    }

    OutBoneTransforms.Sort(FCompareBoneTransformIndex());
}
