// Copyright — see /LICENSE for details.

#include "AnimGraphNode_RTMotion.h"

#define LOCTEXT_NAMESPACE "AnimGraph_RTMotion"

FText UAnimGraphNode_RTMotion::GetNodeTitle(ENodeTitleType::Type TitleType) const
{
	return LOCTEXT("NodeTitle", "RT Motion (SAT-HMR)");
}

FLinearColor UAnimGraphNode_RTMotion::GetNodeTitleColor() const
{
	return FLinearColor(0.35f, 0.65f, 0.85f);
}

FText UAnimGraphNode_RTMotion::GetTooltipText() const
{
	return LOCTEXT(
		"Tooltip",
		"Overrides SMPL joint local rotations each frame from the Motion Receiver Subsystem.");
}

FText UAnimGraphNode_RTMotion::GetControllerDescription() const
{
	return LOCTEXT("Desc", "Real-Time Motion (SMPL 24 joints)");
}

#undef LOCTEXT_NAMESPACE
