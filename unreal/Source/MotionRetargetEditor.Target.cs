// Copyright — see /LICENSE for details.

using UnrealBuildTool;
using System.Collections.Generic;

public class MotionRetargetEditorTarget : TargetRules
{
	public MotionRetargetEditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Editor;
		DefaultBuildSettings = BuildSettingsVersion.V4;
		IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_3;

		ExtraModuleNames.AddRange( new string[] { "MotionRetarget", "MotionRetargetEditor" } );
	}
}
