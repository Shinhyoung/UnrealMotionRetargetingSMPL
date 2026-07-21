// Copyright — see /LICENSE for details.

#include "AnimGraphNode_RTMotionRetarget.h"

#define LOCTEXT_NAMESPACE "AnimGraph_RTMotionRetarget"

FText UAnimGraphNode_RTMotionRetarget::GetNodeTitle(ENodeTitleType::Type TitleType) const
{
    return LOCTEXT("NodeTitle", "RT Motion Retarget (SMPL → Manny)");
}

FLinearColor UAnimGraphNode_RTMotionRetarget::GetNodeTitleColor() const
{
    return FLinearColor(0.35f, 0.85f, 0.65f);
}

FText UAnimGraphNode_RTMotionRetarget::GetTooltipText() const
{
    return LOCTEXT(
        "Tooltip",
        "Chain-based FK retargeting from SMPL predictions to UE Mannequin "
        "(bakes UE IK Retargeter algorithm into a single AnimNode — no IK Rig "
        "/ IK Retargeter assets needed).");
}

FText UAnimGraphNode_RTMotionRetarget::GetControllerDescription() const
{
    return LOCTEXT("Desc", "SMPL → Mannequin FK Retarget");
}

#undef LOCTEXT_NAMESPACE
