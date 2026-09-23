# AGENTS.md — 실시간 다인 모션 리타게팅 프로젝트

이 파일은 Codex가 이 저장소에서 작업할 때 따라야 할 프로젝트 지침이다.
작업 전 반드시 이 문서를 읽고, 아래 규칙과 도메인 제약을 준수한다.

---

## 1. 프로젝트 개요 (Project Overview)

웹캠/영상 속 **최대 5~6명**의 인물 동작을 실시간으로 추정하여, **언리얼 엔진 아바타 여러 개**에
동시에 적용하는 모션 리타게팅 파이프라인.

- **입력**: 단일 RGB 카메라(웹캠) 영상
- **추정 모델**: **SAT-HMR** (CVPR 2025) — 단일샷(one-stage) 다인 3D 메시 복원 모델
- **핵심**: 메시(정점)는 **전송에 사용하지 않고**, SMPL **관절 회전값(θ)만** 추출하여 언리얼로 전송
- **메시 용도**: 검출 결과 검증을 위한 **디버그 2D 오버레이 전용** (프로덕션 경로와 분리, 8절 참고)
- **출력**: 언리얼 엔진에서 person_id별 아바타 본(Bone) 실시간 구동

### 데이터 흐름 (Data Flow)
```
웹캠(RGB)
   │
   ▼
[Python] SAT-HMR 추론 (다인 단일샷)
   │  θ(24 axis-angle) + global_orient + 카메라 기준 3D 위치
   ├───────────────────────────────────────────────┐
   ▼                                                ▼
[Python] 실시간 경로:                        [Python] 디버그 경로 (플래그 ON일 때만):
  person_id 트래킹 → axis-angle→quat →          SMPL forward(LBS) → 정점 →
  좌표 스위즐링 → 스무딩 → UDP 송신              카메라 투영 → 원본 프레임에 2D 오버레이
   │
   ▼
[Unreal C++] UDP 수신 Subsystem → person_id ↔ 아바타 인스턴스 매핑
   ▼
[Unreal C++] FAnimNode에서 각 아바타 본에 로컬 회전 적용
```

---

## 2. 기술 스택 & 환경 (Tech Stack)

| 구분 | 내용 |
|---|---|
| OS | Windows 11 |
| IDE | VSCode + Codex |
| 추정 모델 | SAT-HMR (https://github.com/ChiSu001/SAT-HMR) |
| Python 측 | Python 3.10+, PyTorch(nightly), OpenCV, NumPy, SciPy, (디버그) pyrender/trimesh |
| 통신 | **UDP 소켓** (저지연 우선, 프레임 손실 허용) |
| 엔진 | **Unreal Engine 5.3 이하** (C++ 네이티브) |
| GPU | **RTX 5070 12GB (Blackwell, sm_120)** |

### ⚠️ GPU 환경 필수 주의 (가장 먼저 해결)
RTX 5070은 Blackwell(sm_120)이라 **안정판 PyTorch에서 인식되지 않는다.**
- **CUDA 12.8 이상** 설치
- **PyTorch nightly 빌드** 사용 (안정판 금지)
- 설치 후 `torch.cuda.is_available()` 와 `torch.cuda.get_device_capability()` 가
  `(12, 0)`(sm_120)을 정상 인식하는지 반드시 검증 코드로 확인할 것
- 인식 실패 시 모델 코드부터 손대지 말고 **환경 문제로 우선 진단**한다.

---

## 3. 저장소 구조 (Repository Structure)

```
/python
  /models          # SAT-HMR 로드 및 추론 래퍼
  /tracking        # person_id 트래킹 (프레임 간 ID 매칭)
  /transform       # axis-angle→quat, 좌표계 스위즐링, rest-pose offset
  /network         # UDP 송신 (패킷 직렬화)
  /debug           # 디버그 시각화 (SMPL 메시 생성 + 2D 오버레이) — 프로덕션과 분리
  /config          # 관절 매핑 테이블, 네트워크/디버그 설정
  main.py          # 캡처→추론→(실시간 송신 | 디버그 오버레이) 루프
/unreal
  /Source
    /MotionRetarget
      MotionReceiverSubsystem.h/.cpp   # UDP 수신, person_id별 최신 회전 저장
      RTMotionAnimInstance.h/.cpp      # 회전값 보관 (FAnimNode가 읽음)
      AnimNode_RTMotion.h/.cpp         # 본 회전 적용 (핵심)
/docs
  protocol.md      # UDP 패킷 스펙 (Python↔Unreal 계약)
  bone_mapping.md  # SMPL 24관절 ↔ UE 본 이름 매핑
```

---

## 4. 핵심 도메인 규칙 (반드시 준수 / Critical Domain Rules)

이 프로젝트에서 반복적으로 발생하는 실수들이다. **아래를 위반하는 코드는 작성하지 않는다.**

### 4.1 전송에는 메시를 쓰지 않는다 — θ만 추출
- 실시간 전송 경로에서 SAT-HMR의 **정점(vertices)·렌더링은 사용하지 않는다.**
- `pose`(24×3 axis-angle)와 `global_orient`, 카메라 기준 3D 위치만 뽑아 전송한다.
- 단, **검증용 2D 메시 오버레이는 별도 디버그 경로로 허용**한다 (8절 규칙 준수).

### 4.2 회전은 항상 쿼터니언 — 오일러 각 금지
- axis-angle → **quaternion**으로 변환하여 전송·적용한다.
- 짐벌락 방지를 위해 파이프라인 **어디에서도 오일러(Rotator)로 중간 변환하지 않는다.**
- 변환: `scipy.spatial.transform.Rotation.from_rotvec(...).as_quat()` (xyzw 순서 주의).

### 4.3 좌표계 변환(스위즐링) 필수
- SMPL: **오른손 좌표계, Y-up**
- 언리얼: **왼손 좌표계, Z-up, X-forward**
- Python 측 `/transform`에서 쿼터니언·위치를 **언리얼 축으로 변환**한 뒤 전송한다.
- 축 변환 규칙은 한 곳(`/transform`)에만 두고, 매직넘버를 여러 곳에 흩뿌리지 않는다.

### 4.4 person_id 트래킹은 필수 (다인의 핵심 난제)
- 단일샷 모델은 프레임마다 인물 순서를 보장하지 않는다.
- 프레임 간 **위치 기반 매칭 등으로 person_id를 고정**한다.
- 언리얼로 보내는 모든 인물 데이터에 **안정적인 person_id를 태깅**한다.
- 아바타 깜빡임(엉뚱한 사람 모션 적용)의 원인 1순위이므로 초기 설계부터 반영한다.

### 4.5 Rest Pose(T-Pose) 정합 — retarget offset
- SMPL 기준 포즈와 언리얼 마네킹 기준 포즈 차이를 **1회 보정 오프셋**으로 처리한다.
- 이 보정 없이는 본이 꺾인다. 메시 유무와 무관한 필수 단계.

### 4.6 언리얼 본 적용은 FAnimNode로 (덮어쓰기 문제)
- **`USkeletalMeshComponent::SetBoneRotationByName` 은 존재하지 않는 API다. 사용 금지.**
- `NativeUpdateAnimation`에서 본을 직접 세팅하지 않는다 (AnimGraph가 덮어씀).
- `FAnimNode_SkeletalControlBase`를 상속한 노드의 `EvaluateSkeletalControl_AnyThread`에서
  `OutBoneTransforms`에 넣어 적용한다.
- SMPL θ는 **로컬(부모 기준) 회전**이므로 로컬 트랜스폼 회전을 교체하는 방식으로 다룬다.
- 회전은 `FQuat` 그대로 적용하고 `.Rotator()` 변환을 거치지 않는다.

### 4.7 [현재 구현] Option F: SMPL LBS Direct in UE (2026-07-21~)
**FAnimNode 방식 대신 채택된 현재 아키텍처**:
- `ASMPLProceduralActor` + `FSMPLModel` (Public/Private in `unreal/Source/MotionRetarget`)
- UE Skeletal system 우회. `FSMPLModel::ComputeVertices` 가 CPU 에서 SMPL LBS 직접 실행 → `UProceduralMeshComponent` 로 매 프레임 vertex 갱신
- Python 은 `--smpl-native` 로 raw SMPL Y-up quat + meters root 를 전송, UE 액터가 basis change 수행
- 5개 축 정합 (depth / 위치 L-R / yaw / tilt / 팔·다리 L-R) 실측 완료 — README 의 "5개 축 정합" 표 참조
- Foot IK: two-bone IK 로 planted foot 위치 lock (`SolveLegIK` in SMPLProceduralActor.cpp)
- Multi-material: blob v3 포맷 + `TArray<UMaterialInterface*> Materials` UPROPERTY

**FAnimNode 접근은 archived** — `AnimNode_RTMotion*` 파일은 남아있지만 실사용 안함.

### 4.8 [현재 지원] 커스텀 캐릭터 (Mixamo / 임의 T-pose FBX)
- `tools/convert_fbx_to_blob.py` — 헤드리스 Blender 4.5+ 로 자동 변환
- Mixamo bone → SMPL 24-joint 매핑 (`MIXAMO_TO_SMPL` in `blender_fbx_to_blob.py`)
- 원본 Mixamo weight 재활용 (fingers/twist bones 는 부모 SMPL joint 로 병합)
- UV, embedded texture PNG, 다중 material 자동 추출
- **주의**: Mixamo 캐릭터는 Blender 임포트 시 -Y 방향 (backward) 을 보므로 `blender_to_smpl` 에서 Z 부호 flip 필요 (이미 반영됨)

### 4.9 [현재 지원] SMPL-X (body + hands) — SMPLest-X + `--smpl-x` blob
- Detector: `--detector smplest-x` — [MotrixLab/SMPLest-X](https://github.com/MotrixLab/SMPLest-X) 통합 (별도 conda env 권장)
- Wrapper: `python/detectors/smplest_x_wrapper.py` — YOLOv8x 로 person detection → per-crop SMPLest-X 추론 → 55-joint output (body 22 + face-identity 3 + hands 30)
- `--joint-format smplx`: 55 quats 를 UDP 로 전송 (기존 24 quats 와 wire format 호환 — packet header 에 `joint_count` 필드)
- Blob v3 55-joint 캐릭터: `--smpl-x` 로 변환 (`MIXAMO_TO_SMPLX` 매핑에 finger bones 30개 추가). 없는 finger 는 wrist 위치 fallback (Ch14 같은 4손가락 캐릭터도 처리).
- bf16 autocast 적용 (Blackwell GPU, ~1.5-2x 속도 개선)

### 4.10 [현재 지원] 다중 인원 (SAT-HMR)
- `--max-persons N` (1-3): 카메라 Z 기준 가까운 순으로 필터
- UE 에 `SMPLProceduralActor` N 인스턴스 (`Ctrl+D` 로 복제) + Person Id = 1, 2, 3
- Blob path 모두 동일 (같은 마네킹) 가능, material 별도 지정도 가능
- SMPLest-X 는 crop-based 단일 인물 유리 → launcher 에서 자동 1 로 고정

### 4.11 [현재 지원] Launcher UI (`python/launcher.py`)
- Tkinter 기반. Detector / 인원 수 / 디버그 오버레이 / smoothing α 선택 후 실행 버튼.
- 백그라운드 스레드로 stdout 폴링 → UI 안 얼음.
- SAT-HMR ↔ SMPLest-X env 자동 라우팅 (base anaconda ↔ smplestx conda env).

---

## 5. UDP 패킷 프로토콜 (Python ↔ Unreal 계약)

`/docs/protocol.md`를 단일 진실 소스(source of truth)로 유지한다. 양측 코드를 바꿀 때 이 문서를 먼저 갱신한다.

- 전송 단위: **1 프레임 = 1 패킷** (UDP, 순서/손실 허용)
- 권장 페이로드(예시, 실제 스펙은 protocol.md 확정):
```
frame_id : uint32
person_count : uint8
per person:
    person_id       : uint16      # 트래킹으로 고정된 안정 ID
    root_translation: float32 x3  # 언리얼 좌표계로 변환된 위치
    quats           : float32 x (24*4)  # 본별 쿼터니언 (xyzw), 언리얼 축
```
- UDP이므로 **오래된 frame_id 패킷은 언리얼에서 버린다**(최신만 반영).
- 바이트 순서(엔디안)와 쿼터니언 성분 순서(xyzw vs wxyz)를 양측이 **명시적으로 일치**시킨다.

---

## 6. SMPL 24관절 ↔ 언리얼 본 매핑

`/docs/bone_mapping.md` 및 `/python/config`에 매핑 테이블을 둔다. 코드에 하드코딩하지 말고 설정으로 뺀다.

SMPL 관절 인덱스(0~23): pelvis, L_hip, R_hip, spine1, L_knee, R_knee, spine2,
L_ankle, R_ankle, spine3, L_foot, R_foot, neck, L_collar, R_collar, head,
L_shoulder, R_shoulder, L_elbow, R_elbow, L_wrist, R_wrist, L_hand, R_hand.

→ 언리얼 마네킹 본(pelvis, spine_01~03, clavicle_l/r, upperarm_l/r, lowerarm_l/r,
thigh_l/r, calf_l/r, foot_l/r, neck_01, head 등)에 1:1 매핑한다.
(정확한 본 이름은 사용하는 스켈레톤에 맞춰 bone_mapping.md에서 확정)

---

## 7. 코딩 컨벤션 (Coding Conventions)

### Python
- 타입 힌트 사용, 함수는 단일 책임. 좌표변환/트래킹/네트워크/디버그를 모듈로 분리.
- 추론 루프에서 불필요한 CPU-GPU 왕복·numpy 변환을 최소화 (실시간 성능).
- 매직넘버(축 부호, 관절 인덱스)는 config로 분리.
- 성능 크리티컬 경로에 로깅/시각화를 넣지 않는다 (디버그 플래그로 격리).

### Unreal C++ (UE 5.3 이하 API 기준)
- UE 5.3 이하에서 유효한 API만 사용한다. (신버전 전용 함수 도입 금지)
- `FAnimNode`는 워커 스레드에서 평가되므로 `_AnyThread` 함수 내 스레드 안전성 유지.
- UDP 수신은 게임스레드를 막지 않도록 별도 스레드/러너에서 처리하고,
  최신 회전값만 원자적으로 교체한다.
- UPROPERTY/UFUNCTION 매크로, 언리얼 코딩 표준(PascalCase, 접두어 U/A/F/E) 준수.

---

## 8. 디버그 시각화 (2D 메시 오버레이 / Debug Visualization)

검출 결과를 눈으로 검증하기 위한 **디버그 전용** 기능이다. 프로덕션 실시간 경로와 **반드시 분리**한다.

### 8.1 목적
- SAT-HMR이 인물을 제대로 잡았는지 즉시 확인
- **person_id 트래킹** 안정성 확인 (인물마다 색을 다르게)
- 좌표 스위즐링·rest-pose 정합 **전/후 자세 비교**
- 이상 발생 시 **파이썬 문제 vs 언리얼 문제**를 분리 진단 (파이썬 메시가 정상이면 언리얼 매핑 문제로 범위 축소)

### 8.2 구현 규칙 (엄격히 준수)
- **경로 분리**: 실시간 송신 루프와 디버그 렌더를 같은 함수/스레드에서 섞지 않는다.
  이미 추출한 θ를 분기해서 디버그 경로로만 넘긴다.
- **플래그 제어**: `--debug` (또는 config `debug.overlay=true`)로 **완전히 on/off** 가능해야 한다.
  기본값은 **OFF**.
- **성능 측정 시 OFF**: FPS 실측·성능 튜닝은 **반드시 디버그 OFF 상태**에서 한다.
  (렌더가 켜지면 FPS가 왜곡되므로 측정값을 신뢰하지 않는다.)
- **의존성 격리**: pyrender/trimesh 등 시각화 라이브러리는 `/python/debug`에서만 import 한다.
  프로덕션 모듈이 디버그 라이브러리에 의존하게 만들지 않는다.
- **디버그 실패가 파이프라인을 죽이지 않게**: 렌더 예외는 잡아서 경고만 남기고,
  실시간 송신은 계속 동작하도록 한다.

### 8.3 2D 메시 오버레이 방식 (권장 시작점)
1. 이미 뽑은 θ·global_orient·β로 **SMPL forward(LBS)** 실행 → 정점(vertices) 생성.
   (β는 전송엔 안 쓰지만 메시 형상 확인용으로 디버그에서만 사용 가능)
2. SAT-HMR이 예측한 **카메라 파라미터로 정점을 이미지 평면에 투영**.
3. 원본 프레임 위에 반투명 메시를 오버레이하여 표시(OpenCV `imshow`).
4. 각 인물은 **person_id별 고정 색상**으로 그려 트래킹 스왑을 눈으로 감시.
- SAT-HMR 저장소에 **자체 시각화 유틸이 포함**되어 있으면 우선 재활용한다(중복 구현 금지).
- 3D 뷰어(Open3D 등)는 선택 사항이며, 깊이/다인 배치 확인이 필요할 때만 별도 플래그로 추가한다.

### 8.4 디버그 HUD 권장 표시 항목
- 현재 검출 인원 수, 각 person_id, 프레임 FPS(디버그 표시용), 좌표 변환 적용 여부.

---

## 9. 개발 순서 (권장 마일스톤)

1. **환경 검증**: RTX 5070 + CUDA 12.8 + PyTorch nightly에서 SAT-HMR 추론 성공 (단일 프레임).
2. **디버그 오버레이 우선 구축**: θ → SMPL 메시 → 2D 오버레이로 **검출이 맞는지 먼저 확인**.
   (이후 모든 단계의 검증 도구가 된다.)
3. **단일 인물 E2E**: 1명 → θ 추출 → quat 변환 → UDP → 언리얼 아바타 1개 구동.
4. **좌표/포즈 정합**: 스위즐링 + rest-pose offset으로 꺾임 제거 (오버레이로 전/후 비교).
5. **다인 확장**: person_id 트래킹 + 다중 아바타 매핑 (오버레이로 ID 스왑 감시).
6. **안정화**: 시간축 스무딩, 가림(occlusion) 대응, 패킷 최신화 로직.
7. **성능 튜닝**: 디버그 OFF 상태에서 5~6명 실시간 FPS 확보.

작은 단계로 나눠 각 마일스톤을 실제 실행/검증한 뒤 다음으로 넘어간다.

---

## 10. 검증 규칙 (Verification)

- 새 변환 로직(좌표/쿼터니언)은 **단위 테스트**로 왕복 변환·기지값을 검증한다.
- 언리얼 적용 전, **디버그 2D 메시 오버레이**로 사람 자세와 일치하는지 눈으로 확인한다.
- 다인 트래킹은 **ID가 프레임 간 유지되는지** 오버레이 색상·로그로 확인 (스왑/깜빡임 감시).
- 성능은 추정치로 단정하지 말고 **디버그 OFF 상태의 실측 FPS**로 판단한다.

---

## 11. 하지 말 것 (Do NOT)

- ❌ `SetBoneRotationByName` 등 존재하지 않는 언리얼 API 사용
- ❌ `NativeUpdateAnimation`에서 본 직접 수정
- ❌ 오일러 각(Rotator)으로 중간 변환
- ❌ 좌표계 스위즐링 생략
- ❌ person_id 없이 인물 데이터 전송
- ❌ 메시/렌더링을 **실시간 송신 경로**에 포함 (디버그 경로는 별개로 허용)
- ❌ 디버그 오버레이를 켠 채로 성능(FPS) 측정
- ❌ 프로덕션 모듈이 pyrender 등 디버그 시각화 라이브러리에 의존
- ❌ 안정판 PyTorch로 5070 세팅 시도
- ❌ UE 5.4+ 전용 API 사용 (본 프로젝트는 5.3 이하)
- ❌ SAT-HMR 라이선스 확인 없이 상업적 배포 (저장소 라이선스 반드시 검토)