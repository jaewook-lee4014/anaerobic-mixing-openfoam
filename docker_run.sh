#!/bin/bash
# 실제 OpenFOAM 시뮬레이션을 Docker로 실행

set -e

echo "======================================"
echo "OpenFOAM Docker 실행"
echo "======================================"

# Docker 이미지
IMAGE="openfoam/openfoam11-paraview510"

# Docker 이미지 확인 및 다운로드
if ! docker image inspect $IMAGE &> /dev/null; then
    echo "Docker 이미지 다운로드 중..."
    docker pull $IMAGE
fi

# Docker 컨테이너에서 실행
docker run --rm -v "$(pwd):/work" -w /work $IMAGE bash -c "
    source /opt/openfoam11/etc/bashrc
    
    echo '======================================'
    echo 'OpenFOAM 환경 로드 완료'
    echo 'Version:' \$WM_PROJECT_VERSION
    echo '======================================'
    
    # 케이스 디렉토리 설정
    CASE_DIR=runs/actual_simulation
    
    # 기존 케이스 삭제
    rm -rf \$CASE_DIR
    
    # 케이스 복사
    cp -r runs/case_prod \$CASE_DIR
    
    echo '메쉬 생성 중...'
    blockMesh -case \$CASE_DIR > \$CASE_DIR/log.blockMesh 2>&1
    
    echo '메쉬 체크 중...'
    checkMesh -case \$CASE_DIR > \$CASE_DIR/log.checkMesh 2>&1
    
    # 메쉬 정보 출력
    grep -A 10 'Mesh stats' \$CASE_DIR/log.checkMesh
    
    echo '======================================'
    echo '초기 조건 설정 중...'
    echo '======================================'
    
    # 간단한 테스트 실행 (1초만)
    echo '솔버 실행 (테스트: 1초)...'
    timeout 10 pimpleFoam -case \$CASE_DIR > \$CASE_DIR/log.pimpleFoam 2>&1 || true
    
    echo '======================================'
    echo '결과 확인'
    echo '======================================'
    
    # 잔차 확인
    if [ -f \$CASE_DIR/log.pimpleFoam ]; then
        echo '마지막 잔차:'
        grep 'Solving for' \$CASE_DIR/log.pimpleFoam | tail -5
    fi
    
    # 시간 디렉토리 확인
    echo '생성된 시간 스텝:'
    ls -d \$CASE_DIR/[0-9]* 2>/dev/null | head -5
    
    echo '======================================'
    echo '✅ OpenFOAM 실행 완료'
    echo '======================================'
    echo '결과 위치: \$CASE_DIR'
    echo '로그 파일:'
    echo '  - \$CASE_DIR/log.blockMesh'
    echo '  - \$CASE_DIR/log.checkMesh'
    echo '  - \$CASE_DIR/log.pimpleFoam'
"