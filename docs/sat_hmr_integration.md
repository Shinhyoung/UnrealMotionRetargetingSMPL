# SAT-HMR 통합 가이드 (RTX 5070 / Blackwell 검증)

이 문서는 `python/detectors/sat_hmr_wrapper.py::SATHMRDetector` 를 실행하기 위한 실사용 절차다. 2026-07-13 실제 D455 로 검증 완료.

## 검증된 스택

| 항목 | 버전/값 |
|---|---|
| GPU | RTX 5070 (sm_120, Blackwell) |
| PyTorch | 2.10.0 + cu128 (stable, sm_120 지원) |
| CUDA runtime | 12.8 |
| SAT-HMR commit | main 브랜치, 2026-07-13 clone |
| SMPL 파일 | 사용자 보유본 (HybrIK 프로젝트에서 재활용) |
| xformers | 0.0.35 (memory_efficient_attention 은 native SDPA 로 monkey-patch) |
| 실 D455 추론 | 1280×720 @ ~13.5 FPS (bf16, 미스무딩) |

## 0. 사전 확인

```powershell
cd python
python verify_env.py
```

## 1. SAT-HMR 리포 배치

```powershell
cd C:\0.shinhyoung\Project
git clone --depth 1 https://github.com/ChiSu001/SAT-HMR.git
```

## 2. 파이썬 의존성 (torch 는 절대 재설치 금지 — 5070은 cu128 nightly/stable 유지)

```powershell
pip install pyyaml smplx chumpy xformers accelerate huggingface_hub --no-deps
# 나머지는 python/requirements.txt 참조
```

## 3. chumpy 호환 패치 (Py3.11+ / numpy 2.x)

`docs/fix_chumpy.md` 지침대로 두 곳을 편집. 우리 환경(C:/Users/ADMIN/anaconda3)에는 이미 반영됨:

- `.../chumpy/ch.py` 상단 `import inspect` 뒤에 `inspect.getargspec = inspect.getfullargspec` shim 추가
- `.../chumpy/__init__.py` 11번째 줄 `from numpy import bool, int, float, ...` 주석 처리

## 4. 체크포인트

```powershell
cd python
python tools/download_sat_hmr.py --sat-hmr-root C:\0.shinhyoung\Project\SAT-HMR --variant sat_644.pth
```

파일: `${SAT_HMR_ROOT}/weights/sat_hmr/sat_644.pth` (~875 MB, HuggingFace `ChiSu001/SAT-HMR`).

## 5. SMPL 파일 배치

경로: `${SAT_HMR_ROOT}/weights/smpl_data/smpl/`

필요 파일:
- `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl`, `SMPL_FEMALE.pkl` (모두 SMPL 공식, EULA)
- `smpl_mean_params.npz` — SAT-HMR 저자 배포 (`weights/smpl_data/README_ACTION_REQUIRED.txt` 참조). **없어도 로드는 되지만 shape 만 맞으면 됨**: state_dict 로 덮어씌워짐. 임시 zeros(144)/zeros(10) 로 대체 가능.
- `body_verts_smpl.npy`, `J_regressor_h36m_correct.npy` — eval 전용, inference 에서는 로드 후 미사용. 더미 파일 OK.

우리 환경에서는 `c:/0.shinhyoung/Project/1.Retargeting/Retargeting_noEngine/external/HybrIK/model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` 를 `SMPL_NEUTRAL/MALE/FEMALE.pkl` 세 이름으로 재활용, `J_regressor_h36m.npy` 를 `J_regressor_h36m_correct.npy` 로 재활용, 나머지 두 파일은 더미로 만들어 통과했다.

## 6. 실 파이프라인 실행

```powershell
cd python
python main.py --source realsense --conf-thresh 0.3 --smooth-alpha 0.5
python main.py --source realsense --debug --conf-thresh 0.3   # SAT-HMR vis_meshes_img 재활용 오버레이
```

## 7. 모델 출력 규격 (2026-07-13 clone)

`models/sat_model.py::Model.forward` 반환 dict — 우리 wrapper 매핑:

| 키 | 실제 shape | 우리 파이프라인 |
|---|---|---|
| `pred_poses`      | (1, 50, 72) fp32           | `.reshape(50, 24, 3)` → `PoseDetection.global_orient` (0) + `body_pose` (1..23) |
| `pred_betas`      | (1, 50, 10)                | `beta` (디버그) |
| `pred_transl`     | (1, 50, 3)                 | `root_position` (OpenCV Y-down → Y-up 변환 후) |
| `pred_boxes`      | (1, 50, 4) bf16            | cxcywh 정규화 → 원본 픽셀 xyxy `bbox` |
| `pred_confs`      | (1, 50, 1) bf16            | squeeze → threshold |
| `pred_verts`      | (1, 50, 6890, 3)           | **디버그 전용** (`keep_debug_output=True` 시만 저장) |
| `pred_intrinsics` | (1, 1, 3, 3)               | 디버그 2D 투영 |

## 8. 좌표계

SAT-HMR 출력의 `global_orient` + `pred_transl` 은 **OpenCV 카메라 좌표계** (X-right, Y-DOWN, Z-forward). 우리 wrapper 가 `opencv_cam_yflip_*` 로 Y-flip 을 적용해 SMPL Y-up 로 정합. `body_pose` (23관절 로컬) 는 그대로 유지 — 이후 `transform/coordinate.py` 가 SMPL→UE 스위즐링 담당.

## 9. Blackwell (sm_120) 대응 xformers monkey-patch

xformers Windows 0.0.35 wheel 은 fa2/fa3 미포함, `cutlassF-pt` 는 cc≤9.0 요구 → sm_120 에서 fp32/bf16 모두 실패.
Wrapper 가 `xformers.ops.memory_efficient_attention` 을 `F.scaled_dot_product_attention` 으로 교체한다. 이유:
- 단일 프레임 (batch=1) 추론에서 `BlockDiagonalMask` 는 단일 블록 → 무마스크와 동일
- SDPA 는 sm_120 에서 정상 지원 (fp16/bf16/fp32 모두)

이후 nightly wheel 이 sm_120 지원하면 monkey-patch 제거 가능.

## 10. 실 인퍼런스 결과 예시 (D455, 카메라 앞 사용자 1명)

```
avg latency: 74.1 ms  (13.5 FPS)  last detections: 1
  root=[-0.68 -0.78  2.29]  (SMPL Y-up, meters)
  bbox=[128, 205, 488, 715]  (원본 픽셀)
  global_orient=[0.79 -0.58 -2.92]
```

## 11. 남은 리스크 / 검증 필요

- **좌표 정합**: `opencv_cam_yflip_axis_angle(global_orient)` 는 이론적으로 옳지만 UE 마네킹에 실 붙였을 때 축이 뒤집혀 보일 수 있음. 마일스톤 4 (rest-pose offset 튜닝) 에서 함께 조정한다.
- **SMPL_MALE/FEMALE** 을 neutral 사본으로 대체하면 젠더 편향이 약간 감소하나 실사용에는 영향 미미. 실 데이터 확보되면 교체.
- **`smpl_mean_params.npz`** 우리는 zeros 로 채워넣고 state_dict 로 덮어씀. SAT-HMR 저자 배포본을 얻으면 교체.
- **CLAUDE.md §11**: SAT-HMR / SMPL 라이선스 확인 없이 상업 배포 금지.
