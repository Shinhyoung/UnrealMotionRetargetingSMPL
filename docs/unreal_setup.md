# Unreal 프로젝트 에디터 설정 절차

`unreal/MotionRetarget.uproject` 을 처음 열고 마네킹이 Python 파이프라인 데이터로 움직이는 상태까지 만드는 절차. UE 5.3 기준.

## 0. 사전 준비

- UBT 빌드가 통과한 상태 (스캐폴딩 사이클 완료). 에디터가 열리면 자동으로 컴파일된 DLL 을 로드합니다.
- Python 파이프라인이 D455 로 정상 UDP 송신 확인된 상태.
- 콘솔 명령 `motion.stats` 로 PIE 중에 통계 확인 가능.

## 1. 콘텐츠 마이그레이션 (마네킹 스켈레톤 확보)

프로젝트가 비어있으므로 UE5 Mannequin 을 가져옵니다.

1. `Edit → Plugins` 에서 **"Modeling Tools Editor Mode"** 및 **"Third Person"** 플러그인이 켜져있는지 확인. Third Person 은 콘텐츠 팩 제공.
2. Content Drawer 아래 `+ Add` → `Add Feature or Content Pack` → `Third Person` 선택 → `Add to Project`.
3. 몇 초 후 `Content/Characters/Mannequins/Meshes/SKM_Manny.uasset` 및 `SK_Mannequin.uasset` 등이 임포트됩니다.

## 2. AnimBlueprint 생성

1. Content Drawer 에서 `Content/Characters/Mannequins/Meshes/SK_Mannequin`  (스켈레톤) 우클릭 → `Create → Animation Blueprint`.
2. 뜨는 창에서:
   - **Parent Class**: `RTMotionAnimInstance` (드롭다운에서 검색)
   - **Target Skeleton**: `SK_Mannequin` (자동)
3. 이름: `ABP_RTMannequin`. 원하는 폴더에 저장.

## 3. AnimGraph 구성

`ABP_RTMannequin` 더블클릭 → AnimGraph 탭:

1. **소스 pose 노드 배치** — 그래프 빈 곳 **우클릭** → 검색창에 `Local Reference Pose` 입력 → 클릭해서 배치. 마네킹 T-포즈를 그대로 넘겨주는 소스입니다.
   - UE 5 에는 옛날식 "Palette" 탭이 없고 우클릭 컨텍스트 메뉴가 그 역할.
   - 결과에 안 나오면 상단 **"Context Sensitive"** 체크박스 해제.
2. **RT Motion 노드 배치** — 다시 빈 곳 우클릭 → `RT Motion` 검색 → **"RT Motion (SAT-HMR)"** 클릭해서 배치.
3. **연결**:
   ```
   [Local Reference Pose] → [RT Motion (SAT-HMR)] → [Output Pose]
   ```
   - RT Motion 의 왼쪽 pin `Component Pose` 에 `Local Reference Pose` 출력 연결
   - RT Motion 의 오른쪽 출력을 `Output Pose` (Result) 노드의 왼쪽 입력에 연결
4. 노드 오른쪽 Details 패널에서 **SMPL Bones** 배열 확인:
   - 24개 슬롯에 UE 마네킹 기본 본 이름이 이미 자동 채워져 있습니다 (`pelvis`, `thigh_l`, ...).
   - 인덱스 22, 23 (SMPL `L_hand`/`R_hand`) 은 UE 마네킹에 대응 본이 없어 `None` 으로 유지 — 정상.
   - 커스텀 스켈레톤이라면 여기에서 개별 dropdown 으로 다시 지정.
5. Compile → Save.

## 4. Character Blueprint 생성

1. Content 폴더에서 우클릭 → `Blueprint Class` → `Character`.
2. 이름: `BP_RTCharacter`.
3. 열어서:
   - `Mesh` 컴포넌트 선택 → **Skeletal Mesh Asset**: `SKM_Manny`
   - **Anim Class**: `ABP_RTMannequin_C` (컴파일된 클래스)
4. Character root 를 mesh 아래로 옮겨 발이 지면에 오도록 Transform 조정 (`Location Z = -90`, `Rotation Yaw = -90` 일반적).
5. Compile → Save.

## 5. Level 배치 & Person ID 설정

1. 원하는 Level 열기 (혹은 `File → New Level → Basic` 로 새로).
2. `BP_RTCharacter` 를 Level 로 드래그.
3. Details 패널에서 `Mesh → Anim Class` 가 `ABP_RTMannequin_C` 인지 확인.
4. **Person ID 세팅**: BP_RTCharacter 를 열고 Event Graph 에서 아래 두 줄 추가:
   ```
   Event BeginPlay
     └─ Cast to RTMotionAnimInstance (target: Mesh → Get Anim Instance)
         └─ Set PersonId = 1
   ```
   - 여러 아바타를 배치할 계획이면 인스턴스별로 PersonId 를 다르게 (`1`, `2`, ...) 세팅.
5. Compile → Save Level.

## 6. 실행

1. UE 툴바 ▶ Play (`Alt+P`) 눌러 PIE 시작.
2. Output Log 에서 `LogMotionRetarget: MotionRetarget UDP listener bound on 0.0.0.0:9527` 확인.
3. 다른 터미널에서 Python 실행:
   ```powershell
   cd c:\0.shinhyoung\Project\1.Retargeting\MRetargeting\python
   python main.py --source realsense --conf-thresh 0.3 --smooth-alpha 0.5
   ```
4. UE 뷰포트에 포커스 두고 `~` → `motion.stats` → `recv` 카운터가 올라가면 UDP 흐름 정상.
5. 마네킹이 카메라 앞 사람 자세대로 움직이면 E2E 성공.

## 7. 예상되는 초기 문제 & 다음 조정

- **팔다리가 꺾이거나 T-포즈에서 튀어나온 자세**: Rest-pose offset (`docs/protocol.md` / CLAUDE.md §4.5) 미튜닝 상태. `python/config/settings.py::RestPoseSettings.offset_path` 에 24×4 xyzw quaternion `.npy` 를 붙여야 정합됩니다. 튜닝 툴은 다음 사이클에서.
- **좌우 반전 / 위아래 뒤집힘**: `python/transform/coordinate.py::SMPL_TO_UE_BASIS` 축 매핑 재검토. 지금은 오른손→왼손 chirality flip 상태.
- **뒤통수를 카메라로 향할 때 급격한 회전**: SMPL global_orient 의 gimbal 근처. `smooth_alpha` 를 0.3 근처로 낮추면 완화.

문제 있을 때마다 python 파이프라인의 `--debug --show-mesh` 오버레이로 SAT-HMR 자체가 사람을 제대로 잡는지 먼저 확인 (CLAUDE.md §10 검증 규칙).
