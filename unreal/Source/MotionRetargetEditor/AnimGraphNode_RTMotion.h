// Copyright — see /LICENSE for details.

#pragma once

#include "CoreMinimal.h"
#include "AnimGraphNode_SkeletalControlBase.h"
#include "AnimNode_RTMotion.h"
#include "AnimGraphNode_RTMotion.generated.h"

/**
 * Editor-side wrapper that exposes ``FAnimNode_RTMotion`` in the AnimGraph
 * palette. Runtime lives in the ``MotionRetarget`` module; this file only
 * exists so users can drop the node into an Animation Blueprint.
 */
UCLASS(MinimalAPI)
class UAnimGraphNode_RTMotion : public UAnimGraphNode_SkeletalControlBase
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, Category = "Settings")
	FAnimNode_RTMotion Node;

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
