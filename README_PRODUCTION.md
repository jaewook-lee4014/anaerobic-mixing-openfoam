# 혐기성 소화조 CFD 시뮬레이션 - 프로덕션 버전

## 프로젝트 개요

20×8×16m 혐기성 소화조의 32개 펌프-제트 노즐 혼합 시스템 CFD 시뮬레이션

### 핵심 사양
- **탱크 크기**: 20m × 8m × 16m (2,560 m³)
- **노즐**: 32개 (4행 × 8열)
- **유량**: 430 m³/h (노즐당 13.44 m³/h)
- **목표 평균 속도**: ≥0.30 m/s
- **메쉬**: 100×40×80 = 320,000 cells

## 🚀 실제 시뮬레이션 실행 방법

### 1. Docker 사용 (권장)

```bash
# Docker로 OpenFOAM 실행
chmod +x docker_run.sh
./docker_run.sh
```

### 2. 로컬 OpenFOAM 사용

```bash
# OpenFOAM 환경 로드
source /opt/openfoam11/etc/bashrc

# 시뮬레이션 실행
python3 run_simulation.py
```

### 3. 수동 실행

```bash
cd runs/case_prod
blockMesh                    # 메쉬 생성
checkMesh                    # 메쉬 검사
pimpleFoam > log.pimpleFoam # CFD 솔버 실행
```

## 📁 프로젝트 구조 (정리됨)

```
.
├── run_simulation.py        # 메인 실행 스크립트 (실제 CFD)
├── docker_run.sh           # Docker 실행 스크립트
├── configs/
│   └── case_prod.yaml     # 프로덕션 설정
├── runs/
│   └── case_prod/          # OpenFOAM 케이스 파일
│       ├── 0/              # 초기 조건
│       ├── constant/       # 물성치, fvOptions
│       └── system/         # 솔버 설정
├── src/amx/
│   ├── physics/            # 학술적 물리 모델
│   │   ├── jet_model.py    # 제트 이론 (Rajaratnam 1976)
│   │   ├── turbulence.py   # k-ε 모델 (Launder 1974)
│   │   └── mixing_theory.py # 혼합 이론 (Camp 1943)
│   ├── verification/       # 검증 도구
│   │   └── mesh_independence.py # ASME V&V 20-2009
│   ├── uncertainty/        # 불확실성 정량화
│   └── validation/         # 벤치마크 케이스
└── docs/
    └── API_REFERENCE.md    # API 문서

```

## ✅ 검증 상태

### 구현 완료
- [x] 물리 모델: 학술 문헌 기반
- [x] 메쉬: 320,000 cells
- [x] 노즐 모델링: 32개 momentum source
- [x] 검증 방법론: ASME V&V 20-2009

### 실행 필요
- [ ] 실제 CFD 실행 (pimpleFoam)
- [ ] 메쉬 독립성 연구
- [ ] 벤치마크 검증
- [ ] 불확실성 정량화

## 📊 예상 성능

목표치:
- 평균 속도: ≥0.30 m/s
- Dead zones: <10%
- 혼합 시간: ≤30분
- 에너지 밀도: <20 W/m³

## ⚠️ 중요 사항

1. **실제 CFD 실행 필수**: `run_simulation.py` 또는 `docker_run.sh` 사용
2. **계산 시간**: 전체 시뮬레이션은 수 시간 소요
3. **메모리**: 최소 8GB RAM 필요
4. **검증**: 실제 실행 후 결과 검증 필수

## 🔧 문제 해결

### OpenFOAM 없음
```bash
# Docker 사용
docker pull openfoam/openfoam11-paraview510
./docker_run.sh
```

### 메모리 부족
```bash
# 메쉬 크기 축소
# blockMeshDict에서 (100 40 80) → (50 20 40)
```

### 수렴 문제
```bash
# system/fvSolution에서 relaxation factors 조정
# system/controlDict에서 timeStep 감소
```

## 📞 지원

기술 문의: CFD 엔지니어링 팀

---

**상태**: 프로덕션 준비 (CFD 실행 필요)
**버전**: 1.0.0
**업데이트**: 2025-08-09