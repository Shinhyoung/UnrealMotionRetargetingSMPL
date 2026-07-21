# UDP Packet Protocol (Python → Unreal)

이 문서는 파이썬 파이프라인이 언리얼로 보내는 UDP 패킷의 **단일 진실 소스**다.
Python 측 (`/python/network/packet.py`) 과 Unreal 측 (`MotionReceiverSubsystem`) 코드를 바꾸기 전에 반드시 이 문서를 먼저 갱신한다.

## 1. 전송 규칙

- 프로토콜: **UDP** (unicast)
- 전송 단위: **1 프레임 = 1 패킷**
- 손실/재정렬 허용 — 오래된 `frame_id` 는 언리얼에서 **버림**
- 엔디안: **little-endian** (`<` in Python `struct`, Windows/UE 네이티브)
- 좌표계는 이미 **언리얼 축(Z-up, X-forward, 왼손)** 으로 변환된 값을 담는다 (Python `/transform` 담당)
- 회전은 **쿼터니언 xyzw 순서**, 유닛 쿼터니언

## 2. 패킷 레이아웃

```
[Header]
  frame_id      : uint32   # 4 bytes, monotonically increasing
  person_count  : uint8    # 1 byte

[Per person, repeated person_count times]
  person_id        : uint16                # 2 bytes, tracker가 부여한 안정 ID
  root_translation : float32 x 3           # 12 bytes, (x, y, z) in UE cm
  quats            : float32 x (24 * 4)    # 384 bytes, joint 0..23, xyzw each
```

크기: 헤더 5바이트 + 인당 398바이트. 6명 기준 2393바이트 (MTU 초과 가능성 있음, IP fragmentation 감안).

## 3. 관절 순서

`quats[i]` 는 SMPL 24관절 인덱스 `i` 의 **부모 기준 로컬 회전** (단, `i == 0` 은 pelvis = 글로벌 오리엔테이션).

인덱스와 UE 마네킹 본 대응은 `bone_mapping.md` 참조.

```
0 pelvis         1 L_hip         2 R_hip
3 spine1         4 L_knee        5 R_knee
6 spine2         7 L_ankle       8 R_ankle
9 spine3        10 L_foot       11 R_foot
12 neck         13 L_collar     14 R_collar
15 head         16 L_shoulder   17 R_shoulder
18 L_elbow      19 R_elbow      20 L_wrist
21 R_wrist      22 L_hand       23 R_hand
```

## 4. 좌표계

Python 송신 전에 다음 변환이 완료된 상태여야 한다:

- SMPL (오른손, Y-up) → UE (왼손, Z-up, X-forward)
- 위치: 미터 → 센티미터 (기본 스케일 100, config로 변경 가능)
- 쿼터니언: 로컬 회전, xyzw, 유닛
- 짐벌락 방지: 어디에서도 오일러(Rotator)로 중간 변환 금지

## 5. 수신 측 처리 요약 (참고)

Unreal은 다음을 지켜야 한다:

1. UDP 수신은 별도 스레드/러너에서 처리, 게임스레드 블록 금지
2. `frame_id` 가 마지막 반영본보다 작거나 같으면 **버림**
3. `person_id` 별 최신 회전을 원자적으로 교체
4. `FAnimNode_SkeletalControlBase::EvaluateSkeletalControl_AnyThread` 에서 `OutBoneTransforms` 로 반영
5. `FQuat` 그대로 사용, `.Rotator()` 변환 금지

## 6. 버전

- v0.1 (2026-07-13) 초안. Python 스켈레톤 커밋 기준.
