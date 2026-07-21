// Copyright — see /LICENSE for details.

using UnrealBuildTool;

public class MotionRetarget : ModuleRules
{
	public MotionRetarget(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] {
			"Core",
			"CoreUObject",
			"Engine",
			"InputCore",
			"Networking",
			"Sockets",
			"AnimationCore",
			"AnimGraphRuntime",
			"ProceduralMeshComponent"
		});

		PrivateDependencyModuleNames.AddRange(new string[] {
		});
	}
}
