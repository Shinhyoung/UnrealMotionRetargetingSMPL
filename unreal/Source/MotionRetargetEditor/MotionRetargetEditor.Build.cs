// Copyright — see /LICENSE for details.

using UnrealBuildTool;

public class MotionRetargetEditor : ModuleRules
{
	public MotionRetargetEditor(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] {
			"Core",
			"CoreUObject",
			"Engine",
			"MotionRetarget",
			"AnimGraph",
			"BlueprintGraph"
		});

		PrivateDependencyModuleNames.AddRange(new string[] {
			"UnrealEd",
			"AnimGraphRuntime",
			"AnimationCore",
			"Slate",
			"SlateCore"
		});
	}
}
