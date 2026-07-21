# UE 5.3 IK Retargeter 세팅 (방향 2)

방법 A (per-joint bone correction) 이 SMPL/UE joint local frame 축 mismatch 를 근본적으로 못 잡으므로, UE 5 의 **IK Retargeter** 기능으로 넘긴다. Python 은 raw SMPL 관절 회전만 전송하고, UE 가 SMPL skeleton 을 SKM_Manny 로 매핑.

## 자동화 스크립트 있음

**§1 SMPL FBX 생성 후, §2~§4 는 아래 스크립트 한 번 실행으로 자동화됩니다**:
```
UE Editor → Tools → Execute Python Script → unreal/Content/Python/setup_ik_retarget.py
```
스크립트가 하는 것:
- SMPL_Skeleton.fbx import → SK_SMPL_Skeleton
- IK_SMPL IKRig (chains + root 세팅)
- IK_Mannequin IKRig (chains + root)
- RTG_SMPL_to_Manny IKRetargeter (chain auto-mapping)

**§5 (AnimBP + BP_RTCharacter) 는 여전히 수동** — K2 그래프 편집이 Python 에 노출 안 됨. 다만 C++ Character 클래스로 대체 가능 (별도 개선 대상).

## 전체 흐름

```
Python 파이프라인 (--raw-smpl)
  → UDP: SMPL 관절 회전 (24개, basis change 만 적용)
  → UE: BP_RTCharacter
      → SMPL SkeletalMesh (invisible source, ABP_SMPL_Source 로 UDP 수신)
      → SKM_Manny (visible target, ABP_RTMannequin 에서 Retarget Pose from Mesh 노드로 source 복사)
      → IK Retargeter (SMPL → SKM_Manny) 가 자동 매핑
```

## 1. SMPL Skeleton 자산 만들기

세 가지 옵션. **하나만** 선택.

### 옵션 A: 자체 Blender 스크립트 (권장, 가장 재현성 높음)

**필요**: Blender 3.x 또는 4.x (무료, https://www.blender.org/download/)

1. Python 에서 SMPL rest 데이터 생성 (이미 완료된 상태):
   ```powershell
   cd c:\0.shinhyoung\Project\1.Retargeting\MRetargeting\python
   python tools/dump_smpl_rest.py --output-json ../smpl_rest.json --output-obj ../smpl_tpose.obj
   ```
   → 프로젝트 루트에 `smpl_rest.json` (관절 계층) + `smpl_tpose.obj` (T-pose mesh)

2. Blender 로 FBX 생성 (headless):
   ```powershell
   & "C:\Program Files\Blender Foundation\Blender 4.x\blender.exe" --background --python `
       "c:\0.shinhyoung\Project\1.Retargeting\MRetargeting\python\tools\make_smpl_fbx_blender.py" -- `
       --skeleton-json "c:\0.shinhyoung\Project\1.Retargeting\MRetargeting\smpl_rest.json" `
       --mesh-obj      "c:\0.shinhyoung\Project\1.Retargeting\MRetargeting\smpl_tpose.obj" `
       --output-fbx    "c:\0.shinhyoung\Project\1.Retargeting\MRetargeting\SMPL_Skeleton.fbx"
   ```
   → 프로젝트 루트에 `SMPL_Skeleton.fbx`

### 옵션 B: Meshcapade SMPL Blender addon

Meshcapade 가 공식 SMPL Blender addon 배포 (계정 필요, 무료). GUI 로 armature 생성 → FBX export. https://meshcapade.com/

### 옵션 C: 기존 SMPL FBX 다운로드

GitHub 나 학술 프로젝트에서 SMPL 리깅된 FBX 가 공유되어 있음. 라이센스 확인 필수.

## 2. UE 로 FBX Import

1. UE Editor 열기, MotionRetarget.uproject 로드
2. Content Browser 우클릭 → `Import to /Content/...`
3. `SMPL_Skeleton.fbx` 선택
4. Import 옵션:
   - **Skeletal Mesh**: ✓
   - **Import Mesh**: ✓
   - **Skeleton**: (자동 생성)
   - **Import Animations**: ✗ (지금은 필요 없음)
   - **Convert Scene**: ✓
   - **Force Front XAxis**: ✓ (UE 표준 orient)
   - **Use T0As Ref Pose**: ✓
5. Import → `SK_SMPL_Skeleton`, `SK_SMPL_Skeleton_Skeleton`, `SK_SMPL_Skeleton_PhysicsAsset` 생성됨

## 3. IK Rig 두 개 만들기

### 3-1. SMPL 용 IK Rig

1. Content Browser → 우클릭 → `Animation → IK Rig`
2. Preview Skeletal Mesh: `SK_SMPL_Skeleton`
3. 이름: `IK_SMPL`
4. 열어서:
   - Right panel `Retarget Chains` 에서 `+ Add New Chain` 여러 번:
     - `Root` (start: `pelvis`, end: `pelvis`)
     - `Spine` (start: `spine1`, end: `spine3`)
     - `Neck` (start: `neck`, end: `head`)
     - `LeftArm` (start: `L_collar`, end: `L_wrist`)
     - `RightArm` (start: `R_collar`, end: `R_wrist`)
     - `LeftLeg` (start: `L_hip`, end: `L_ankle`)
     - `RightLeg` (start: `R_hip`, end: `R_ankle`)
5. `Retarget Root Bone` 설정: `pelvis` (viewport 에서 pelvis 우클릭 → `Set Retarget Root`)
6. Save.

### 3-2. SKM_Manny 용 IK Rig

기본 UE Mannequin 은 이미 `IK_Mannequin` asset 이 있음 (`/Game/Characters/Mannequins/Rigs/IK_Mannequin`). 재사용.

없거나 재구성 필요하면:
1. `+ Add New` IK Rig
2. Preview: `SKM_Manny`
3. Chains:
   - `Root` (pelvis → pelvis)
   - `Spine` (spine_01 → spine_05)
   - `Neck` (neck_01 → head)
   - `LeftArm` (clavicle_l → hand_l)
   - `RightArm` (clavicle_r → hand_r)
   - `LeftLeg` (thigh_l → foot_l)
   - `RightLeg` (thigh_r → foot_r)
4. Retarget Root: `pelvis`
5. Save.

## 4. IK Retargeter 만들기 (SMPL → Mannequin)

1. Content Browser → 우클릭 → `Animation → IK Retargeter`
2. Source IK Rig: `IK_SMPL`
3. Target IK Rig: `IK_Mannequin`
4. 이름: `RTG_SMPL_to_Manny`
5. 열어서 chain mapping 확인 (자동으로 이름 매칭됨):
   - Root ↔ Root, Spine ↔ Spine, LeftArm ↔ LeftArm, ...
6. Viewport 에서 미리보기: source 는 SMPL T-pose, target 는 마네킹 A-pose 로 표시됨. Preview animation 재생하면 두 skeleton 모두 반응해야 정상
7. Save.

## 5. AnimBP + BP_RTCharacter 재구성

### 5-1. Source AnimBP (SMPL 구동)

1. `SK_SMPL_Skeleton` 우클릭 → `Create → Animation Blueprint`
2. Parent Class: `RTMotionAnimInstance`
3. Skeleton: `SK_SMPL_Skeleton_Skeleton`
4. 이름: `ABP_SMPL_Source`
5. AnimGraph:
   - `Local Reference Pose` → `RT Motion (SAT-HMR)` → `Output Pose`
   - `RT Motion` 의 SMPLBones 는 SMPL 스켈레톤 본 이름으로 재설정:
     - [0] pelvis, [1] L_hip, [2] R_hip, [3] spine1, [4] L_knee, [5] R_knee,
     - [6] spine2, [7] L_ankle, [8] R_ankle, [9] spine3, [10] L_foot, [11] R_foot,
     - [12] neck, [13] L_collar, [14] R_collar, [15] head,
     - [16] L_shoulder, [17] R_shoulder, [18] L_elbow, [19] R_elbow,
     - [20] L_wrist, [21] R_wrist, [22] L_hand, [23] R_hand
6. Compile → Save

### 5-2. Target AnimBP (Mannequin 렌더 + retarget)

`ABP_RTMannequin` 을 수정 (또는 새로 만듦):

1. AnimGraph:
   - `Retarget Pose From Mesh` 노드 배치 (우클릭 검색)
   - Details 패널:
     - `IK Retargeter Asset`: `RTG_SMPL_to_Manny`
     - `Source Mesh Component`: BP_RTCharacter 의 SMPL mesh component (아래 5-3 참조)
   - Output → `Output Pose`
2. Compile → Save

### 5-3. BP_RTCharacter 수정

1. `BP_RTCharacter` 열기
2. Components 패널에서 기존 `Mesh` (SKM_Manny) 유지
3. `Add Component → Skeletal Mesh` 로 두 번째 mesh 추가:
   - 이름: `SMPLMesh`
   - Skeletal Mesh Asset: `SK_SMPL_Skeleton`
   - Anim Class: `ABP_SMPL_Source`
   - **Visibility**: Hidden In Game ✓ (source 는 안 보임)
4. 기존 `Mesh` 컴포넌트:
   - Anim Class: `ABP_RTMannequin` (retarget 노드가 SMPLMesh 를 참조)
5. `Retarget Pose From Mesh` 노드의 `Source Mesh Component` 를 `SMPLMesh` 로 지정
6. Event BeginPlay: 기존 Cast to RTMotionAnimInstance → Set Person Id 는 **SMPLMesh 의 AnimInstance** 로 변경
   ```
   Event BeginPlay
     → SMPLMesh → Get Anim Instance → Cast to RTMotionAnimInstance → Set Person Id = 1
   ```
7. Compile → Save

## 6. 실행

```powershell
cd c:\0.shinhyoung\Project\1.Retargeting\MRetargeting\python
python main.py --source realsense --conf-thresh 0.3 --smooth-alpha 0.5 --raw-smpl
```

- `--raw-smpl` 만 붙임 (bone-correction, pelvis-rest, chirality 다 안 씀)
- UE PIE ▶ Play

## 7. 예상 문제

- **T-pose vs A-pose rest 차이**: SMPL 은 T-pose (팔 수평), SKM_Manny 는 A-pose. IK Retargeter Viewport 에서 두 skeleton 의 rest 를 각각 확인. 필요 시 IK Retargeter 의 `Global Settings → Enable Post Settings → Pose from Retarget` 로 조정
- **Root motion 이슈**: SMPL 의 root position (pelvis) 과 UE actor transform 이 이중 적용될 수 있음. `RT Motion` 노드의 root position 처리 필요 시 SMPL AnimBP 에서 별도 처리
- **Retarget chain mismatch**: source 와 target chain 개수가 다르면 mapping 이 불완전. 팔·다리 chain 길이가 다르면 IK Retargeter 가 자동 interpolate

## 참고

- UE 5 IK Retargeter 공식 문서: https://docs.unrealengine.com/5.3/en-US/ik-rig-animation-retargeting-in-unreal-engine/
- SMPL 관련: https://smpl.is.tue.mpg.de/
