# Unreal Motion Retargeting with SMPL

실시간 다인 모션 리타게팅 파이프라인. RGB+Depth 카메라로 사람의 자세를 추정하고 Unreal Engine 아바타에 매핑합니다.

- **입력**: Intel RealSense D455 (RGB + Depth)
- **자세 추정**: [SAT-HMR](https://github.com/ChiSu001/SAT-HMR) (CVPR 2025, monocular 다인 3D 메시)
- **전송**: UDP (1 프레임 = 1 패킷, 최신 프레임 우선)
- **렌더링**: Unreal Engine 5.3 커스텀 액터가 CPU 에서 SMPL LBS 직접 실행

## 데이터 흐름

```
RealSense (RGB+Depth)
    │
    ▼
[Python] SAT-HMR 추론 → θ (24 axis-angle) + β + camera-space root
    │
    ├─ Depth root-lock: RealSense depth 로 pelvis 3D 위치 back-project
    ├─ Person tracking (frame 간 ID 유지)
    ├─ L/R mirror (뷰포트 관점 정합용)
    └─ EMA 스무딩 (pose α=0.5, root α=0.3)
    │
    ▼
[UDP] 24 quats (xyzw) + root translation (SMPL Y-up meters)
    │
    ▼
[Unreal C++] MotionReceiverSubsystem → ASMPLProceduralActor
    │
    ├─ FSMPLModel LBS (CPU): joints × body pose → 6890 vertices
    ├─ SMPL Y-up → UE Z-up chirality-preserving basis
    ├─ Depth mirror + L/R mirror (UPROPERTY 토글 가능)
    │
    ▼
ProceduralMeshComponent 실시간 갱신
```

## 5개 축 정합 상태 (2026-07-21 기준)

뷰포트를 마네킹 뒤에서 (`-X` 쪽) 바라보는 기준으로 아래 다섯 축 모두 사용자와 같은 방향으로 매핑됩니다:

| 축 | 담당 fix |
|---|---|
| Depth (앞뒤) | `--depth-root-lock` + UE `bInvertRootDepth` |
| 위치 L/R | UE `bInvertRootLR` |
| Yaw (좌우 회전) | Chirality-preserving M (`det=+1`) + winding order swap |
| Tilt (roll) | `--swap-lr` global orient XZ-plane mirror |
| 팔/다리 L/R | `--swap-lr` body pose YZ-plane mirror + index swap |

자세한 배경은 [docs/](docs/) 및 [CLAUDE.md](CLAUDE.md) 참고.

## 요구 사항

**하드웨어**
- Intel RealSense D455 (또는 호환 depth 카메라)
- NVIDIA GPU (테스트: RTX 5070 Blackwell sm_120)

**소프트웨어**
- Windows 11
- CUDA 12.8+
- PyTorch nightly (Blackwell 지원)
- Unreal Engine 5.3
- Visual Studio 2022 Build Tools

## 설치

### 1. SAT-HMR 준비

이 저장소는 SAT-HMR 를 포함하지 않습니다. 별도 clone 후 체크포인트 다운로드:

```powershell
git clone https://github.com/ChiSu001/SAT-HMR.git C:\path\to\SAT-HMR
cd C:\path\to\SAT-HMR
# 저장소 README 지침대로 weights 다운로드
```

Python 파이프라인이 `--sat-hmr-root` 로 이 경로를 참조합니다.

### 2. Python 환경

```powershell
cd python
pip install -r requirements.txt
# Blackwell (sm_120) GPU 인 경우 nightly 필요:
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
python verify_env.py  # torch.cuda.is_available() + sm_120 확인
```

### 3. SMPL 원본 파일

**중요**: SMPL 은 research-only 라이선스입니다. 저장소에 포함되지 않으니 [SMPL 공식 사이트](https://smpl.is.tue.mpg.de/) 에서 직접 받으세요.

`SMPL_NEUTRAL.pkl` 을 준비한 뒤 파생 파일들을 생성:

```powershell
cd python
# UE actor 용 blob (필수)
python tools/dump_smpl_blob.py --smpl-pkl <path>/SMPL_NEUTRAL.pkl --out ../smpl_model.bin

# Blob 검증 (선택)
python tools/verify_smpl_blob.py --smpl-pkl <path>/SMPL_NEUTRAL.pkl
```

### 4. Unreal 프로젝트 빌드

```powershell
# UE 5.3 설치 필요. UBT 로 빌드:
& "C:\Program Files\Epic Games\UE_5.3\Engine\Build\BatchFiles\Build.bat" `
    MotionRetargetEditor Win64 Development `
    -Project="<repo>/unreal/MotionRetarget.uproject"
```

빌드 후 `unreal/MotionRetarget.uproject` 를 UE 에디터로 열고:
1. 레벨에 `SMPLProceduralActor` 배치
2. Details 패널에서 `SMPL Blob Path` 를 `smpl_model.bin` 절대경로로 설정
3. `Invert Root Depth`, `Invert Root LR` 는 기본값 (true) 유지

**Content/** 는 저장소에 포함되지 않습니다. 새 Third Person 템플릿 프로젝트를 만들어 참고하거나 자체 레벨을 구성하세요.

### 5. 캘리브레이션 (선택)

Pelvis 자세 baseline (사용자별로 다름):

```powershell
cd python
python tools/calibrate_pelvis_rest.py --source realsense --out ../pelvis_rest.npy
# 60프레임 동안 upright 자세로 서 있으면 저장됨
```

## 실행

UE 에디터에서 PIE 시작한 뒤:

```powershell
cd python
python -u main.py --source realsense --conf-thresh 0.3 `
    --smooth-alpha 0.5 --smooth-root-alpha 0.3 `
    --smpl-native --pelvis-rest ../pelvis_rest.npy `
    --single-front-person --depth-root-lock --swap-lr
```

주요 플래그:
- `--smpl-native`: SMPL Y-up 원본 quat 전송 (UE 액터가 LBS 실행)
- `--depth-root-lock`: RealSense depth 로 pelvis 3D 위치 back-project (monocular Z 오차 보정)
- `--swap-lr`: L/R 미러 (뷰포트 관점 정합)
- `--single-front-person`: 카메라에 가장 가까운 사람만 (다중 인물 확장 시 제거)
- `--debug --show-mesh`: OpenCV 창에 SAT-HMR mesh 오버레이 (디버그)

전체 플래그 목록: `python main.py --help`

## GUI Launcher

명령어 대신 GUI 로 실행 가능:

```powershell
cd python
python launcher.py     # 또는 pythonw launcher.py (콘솔 없이)
```

- Detector 선택 (SAT-HMR / SMPLest-X)
- Multi-Person 인원 수 (1-3, SAT-HMR 만)
- 디버그 오버레이 on/off
- Smoothing α 조정
- 실행 / 정지 버튼 + 실시간 로그

## 다중 인원 지원 (SAT-HMR)

`--max-persons N` (1-3): 카메라 Z 기준 가까운 순으로 N 명 유지. Person 마다 tracker 가 stable id (1..N) 부여.

**UE 설정**:
1. 레벨에 `SMPLProceduralActor` N 개 인스턴스 배치 (`Ctrl+D` 로 복제)
2. 각 액터 Details → **Person Id** = 1, 2, 3 (서로 다르게)
3. **SMPL Blob Path** 는 모두 동일 (같은 마네킹) — 원하면 인원별 다른 blob/material 가능

## SMPLest-X: SMPL-X (body + hands)

손가락까지 애니메이션. SMPLest-X 는 무거운 모델 (687M) 이라 SAT-HMR 보다 느림 (5-7 FPS bf16).

**한 번만 셋업**:
- SMPLest-X repo clone: [MotrixLab/SMPLest-X](https://github.com/MotrixLab/SMPLest-X)
- SMPL-X 모델 파일 → `SMPLest-X/human_models/human_model_files/smplx/`
- SMPLest-X Huge 가중치 (8.2GB) → `SMPLest-X/pretrained_models/smplest_x_h/smplest_x_h.pth.tar`
- 별도 conda env (smplestx) 권장: Python 3.10 + PyTorch nightly + `pip install -r requirements.txt` + chumpy

**실행**:
```powershell
python -u main.py --source realsense `
    --detector smplest-x --smplest-x-root <path> --smplest-x-ckpt smplest_x_h `
    --joint-format smplx `
    --smpl-native --pelvis-rest ../pelvis_rest.npy --swap-lr --depth-root-lock
```

**손까지 UE 에서 반영**: Ch17 등 캐릭터를 `--smpl-x` 로 blob 재생성 후 액터에 적용.

## 커스텀 캐릭터 사용 (Mixamo 등)

기본 SMPL 마네킹 외에 Mixamo 캐릭터 등 임의 T-pose FBX 를 사용할 수 있습니다. Blender 헤드리스 자동 변환 도구 제공:

```powershell
cd python
python tools/convert_fbx_to_blob.py `
    --fbx "path/to/character.fbx" `
    --smpl-pkl "path/to/SMPL_NEUTRAL.pkl" `
    --out "path/to/character_model.bin"
```

자동으로:
1. FBX 임포트 (Blender 4.5+ 필요, 자동 감지)
2. Mixamo bone 이름 → SMPL 24-joint 매핑 (원본 Mixamo weights 재활용 — bone-heat 자동 weight 대비 훨씬 자연스러운 deformation)
3. 다중 mesh 파트 join + material 정보 보존
4. Mesh scale 정규화, 좌표계 변환 (Blender Z-up → SMPL Y-up)
5. UV 추출
6. Embedded 텍스처 PNG 자동 저장 (`<blob>_textures/` 폴더)
7. Blob v3 파일 write

**SMPL-X 55-joint blob 생성** (손가락까지):
```powershell
python tools/convert_fbx_to_blob.py --fbx <char>.fbx --smpl-pkl <path>/SMPL_NEUTRAL.pkl --out <char>_smplx.bin --smpl-x
```
Mixamo finger bones (Thumb/Index/Middle/Ring/Pinky × 3 joints × 2 hands = 30) 를 MANO 순서로 매핑. Pinky 없는 캐릭터 (Ch14 등) 는 wrist 위치에 fallback.

**UE 에 적용**:
1. `smpl_model_xxx_textures/` 폴더의 PNG 들을 UE Content Browser 로 드래그 (Texture2D 임포트)
2. **Normal map** 파일들은 `Compression Settings = Normalmap` 로 재임포트 (기본 Color 는 아티팩트 유발)
3. 각 material 별로 UE Material 생성 (예: `M_Body`, `M_Hair`) — Diffuse → Base Color, Normal → Normal, 필요시 Roughness/Metallic 연결
4. 액터 Details → SMPL → **Materials** 배열: 각 인덱스에 해당 material 지정 (Output Log 에 표시된 순서대로)
5. **SMPL Blob Path** 를 새 blob 파일로 변경 → PIE 재시작

Mixamo 스타일 카툰 룩 재현: Roughness=1.0 constant + Normal 생략 + Metallic=0.

## Foot IK

기본으로 활성화된 two-bone IK 가 발이 지면에 planting 됐을 때 미끄러짐 방지:
- 액터 Details → **SMPL | FootIK**
  - `Enable Foot IK` (기본 true)
  - `Foot Plant Speed` (기본 0.15 m/s — 이 이하 속도면 planted)
  - `Foot Release Speed` (기본 0.4 m/s — 이 이상이면 발이 떨어짐)

Twist 보존 delta rotation 방식으로 다리 tremor 최소화. 단, 미끄러짐 완전 제거는 아니고 자연스러운 정도.

## Blob 포맷 (SMPB)

| Version | 추가 필드 | 용도 |
|---|---|---|
| v1 | (기본) parents, rest_joints, rest_verts, faces, weights | 초기 SMPL 렌더 |
| v2 | + UVs (per-vertex) | 텍스처 매핑 |
| v3 | + material_names, face_material_ids | 다중 material 지원 |

UE 액터가 v1/v2/v3 모두 로드 가능 (backward compat). `dump_smpl_blob.py` 는 v1, `convert_fbx_to_blob.py` 는 v3 생성.

## 구조

```
python/
  main.py                # 파이프라인 진입점
  detectors/             # SAT-HMR wrapper
  tracking/              # person_id 트래킹
  transform/             # 좌표변환 (SMPL Y-up ↔ UE Z-up)
  depth/                 # RealSense depth 유틸 (verify + root-lock)
  smoothing/             # 시간축 EMA (pose + root 분리 alpha)
  network/               # UDP 패킷 직렬화
  capture/               # RealSense / 웹캠 / 파일 캡처
  debug/                 # 디버그 오버레이 (프로덕션 분리)
  tools/                 # 유틸 스크립트 (blob 생성, 캘리브 등)
  config/                # 매핑 테이블

unreal/
  MotionRetarget.uproject
  Source/MotionRetarget/     # 런타임 모듈
    Public/
      MotionReceiverSubsystem.h   # UDP 수신
      SMPLModel.h                 # SMPL LBS
      SMPLProceduralActor.h       # 액터 (mesh 렌더)
    Private/
  Source/MotionRetargetEditor/    # 에디터 모듈

docs/
  protocol.md            # UDP 패킷 스펙
  bone_mapping.md        # SMPL 24 joint ↔ UE bone
  sat_hmr_integration.md # SAT-HMR 통합 노트
  unreal_setup.md        # UE 프로젝트 설정
  ue_ik_retargeter.md    # 대안 방식 (IK Retargeter) 기록
```

## 주요 도구

| 파일 | 용도 |
|---|---|
| `main.py` | 파이프라인 진입점 |
| `tools/dump_smpl_blob.py` | SMPL PKL → 기본 SMPL blob (v1) |
| `tools/convert_fbx_to_blob.py` | 임의 T-pose FBX → blob (v3, wrapper) |
| `tools/blender_fbx_to_blob.py` | Blender 헤드리스 스크립트 (wrapper 가 호출) |
| `tools/calibrate_pelvis_rest.py` | Pelvis baseline 캘리브레이션 |
| `tools/verify_smpl_blob.py` | Blob LBS ↔ smplx 레퍼런스 비교 검증 |

## 라이선스

- 프로젝트 코드: MIT (또는 원하는 라이선스로 명시)
- **SMPL 모델**: [research-only 라이선스](https://smpl.is.tue.mpg.de/modellicense.html). 상업적 사용 금지.
- **SAT-HMR**: [해당 저장소 라이선스](https://github.com/ChiSu001/SAT-HMR) 확인.
- **Unreal Engine**: Epic Games EULA.

## 참고

- 상세 세션 기록 및 축 정합 설계 이력은 [CLAUDE.md](CLAUDE.md) 참고
- UDP 패킷 포맷: [docs/protocol.md](docs/protocol.md)
- SMPL ↔ UE 본 매핑: [docs/bone_mapping.md](docs/bone_mapping.md)
