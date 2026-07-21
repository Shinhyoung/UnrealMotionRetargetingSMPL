# SMPL 24-Joint ↔ UE Mannequin Bone Mapping

Python (`/python/config/bone_mapping.py`) 와 Unreal 측 매핑 테이블이 동기화되어야 한다.
실제 사용 스켈레톤이 UE 마네킹이 아니라면 이 문서만 갱신해서 대응.

## 매핑 테이블 (기본: UE5 Mannequin)

| SMPL idx | SMPL joint    | UE bone         | 비고 |
|---------:|---------------|-----------------|------|
|  0       | pelvis        | pelvis          | 글로벌 오리엔테이션 겸함 |
|  1       | L_hip         | thigh_l         |      |
|  2       | R_hip         | thigh_r         |      |
|  3       | spine1        | spine_01        |      |
|  4       | L_knee        | calf_l          |      |
|  5       | R_knee        | calf_r          |      |
|  6       | spine2        | spine_02        |      |
|  7       | L_ankle       | foot_l          |      |
|  8       | R_ankle       | foot_r          |      |
|  9       | spine3        | spine_03        |      |
| 10       | L_foot        | ball_l          | 발가락 밑, 없으면 skip |
| 11       | R_foot        | ball_r          | 발가락 밑, 없으면 skip |
| 12       | neck          | neck_01         |      |
| 13       | L_collar      | clavicle_l      |      |
| 14       | R_collar      | clavicle_r      |      |
| 15       | head          | head            |      |
| 16       | L_shoulder    | upperarm_l      |      |
| 17       | R_shoulder    | upperarm_r      |      |
| 18       | L_elbow       | lowerarm_l      |      |
| 19       | R_elbow       | lowerarm_r      |      |
| 20       | L_wrist       | hand_l          |      |
| 21       | R_wrist       | hand_r          |      |
| 22       | L_hand        | (skip)          | 손가락 없음, 대응 없으면 무시 |
| 23       | R_hand        | (skip)          | 손가락 없음, 대응 없으면 무시 |

## 주의사항

- SMPL의 조인트 회전은 **부모 기준 로컬 회전(axis-angle → 쿼터니언 변환 후)**
- pelvis(0) 은 **글로벌 오리엔테이션** 을 함께 담는다 — UE `pelvis` 로컬 회전에 들어감
- SMPL 22/23 (L_hand, R_hand) 는 실제로 손목 하위 관절 하나뿐이므로 UE 마네킹에는 대응이 없다. 무시하거나 손목에 병합.
- **Rest-pose offset** (T-pose 정합) 은 별도 오프셋 쿼터니언을 곱해서 처리 — `/python/transform/rest_pose.py`
- 커스텀 스켈레톤 쓰는 경우 위 표를 갈아엎고, `bone_mapping.py` 및 Unreal 측 매핑을 함께 수정
