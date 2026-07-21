// Copyright — see /LICENSE for details.

#pragma once

#include "CoreMinimal.h"
#include "AnimGraphNode_SkeletalControlBase.h"
#include "AnimNode_RTMotionRetarget.h"
#include "AnimGraphNode_RTMotionRetarget.generated.h"

/**
 * Editor wrapper for ``FAnimNode_RTMotionRetarget`` (chain-based FK
 * retargeting). Replaces the direct RT Motion node when you want SMPL →
 * SKM_Manny retargeting done entirely inside our AnimNode (no IK Rig / IK
 * Retargeter assets required).
 */
UCLASS(MinimalAPI)
class UAnimGraphNode_RTMotionRetarget : public UAnimGraphNode_SkeletalControlBase
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, Category = "Settings")
    FAnimNode_RTMotionRetarget Node;

    //~ Begin UEdGraphNode interface
    virtual FText GetNodeTitle(ENodeTitleType::Type TitleType) const override;
    virtual FLinearColor GetNodeTitleColor() const override;
    virtual FText GetTooltipText() const override;
    //~ End UEdGraphNode interface

protected:
    //~ Begin UAnimGraphNode_SkeletalControlBase interface
    virtual const FAnimNode_SkeletalControlBase* GetNode() const override { return &Node; }
    virtual FText GetControllerDescription() const override;
    //~ End UAnimGraphNode_SkeletalControlBase interface
};
