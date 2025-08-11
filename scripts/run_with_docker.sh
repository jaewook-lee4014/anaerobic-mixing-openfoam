#!/bin/bash
# Docker를 사용한 OpenFOAM 실행 스크립트

set -e

# Configuration
DOCKER_IMAGE="openfoam/openfoam11-paraview510"
WORK_DIR="/work"
CASE_NAME="${1:-case_prod}"
CONFIG_FILE="${2:-configs/case_prod.yaml}"

echo "======================================"
echo "OpenFOAM Docker Runner"
echo "======================================"
echo "Case: $CASE_NAME"
echo "Config: $CONFIG_FILE"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    echo "Please install Docker from https://www.docker.com/get-started"
    exit 1
fi

# Check if Docker image exists
echo "Checking Docker image..."
if ! docker image inspect $DOCKER_IMAGE &> /dev/null; then
    echo "Docker image not found. Pulling $DOCKER_IMAGE..."
    docker pull $DOCKER_IMAGE
fi

# Create case directory if it doesn't exist
CASE_DIR="runs/$CASE_NAME"
mkdir -p $CASE_DIR

# Create a Docker entrypoint script
cat > $CASE_DIR/run_simulation.sh << 'EOF'
#!/bin/bash
set -e

cd /work

# Source OpenFOAM environment
source /opt/openfoam11/etc/bashrc

echo "OpenFOAM environment loaded"
echo "WM_PROJECT_VERSION = $WM_PROJECT_VERSION"

# Run Python setup script to generate case files
echo "Generating case files..."
python3 -m amx.cli setup-case --config $1 --output $2

CASE_PATH="$2/case"
cd $CASE_PATH

# Check mesh
if [ ! -d "constant/polyMesh" ]; then
    echo "Running blockMesh..."
    blockMesh > log.blockMesh 2>&1
    
    if [ $? -ne 0 ]; then
        echo "❌ Mesh generation failed"
        tail -20 log.blockMesh
        exit 1
    fi
    echo "✓ Mesh generated successfully"
else
    echo "✓ Mesh already exists"
fi

# Check mesh quality
echo "Checking mesh quality..."
checkMesh > log.checkMesh 2>&1
grep -A 5 "Mesh stats" log.checkMesh

# Initialize fields if needed
if [ ! -f "0/U" ]; then
    echo "❌ Initial conditions not found"
    exit 1
fi

# Run solver
echo "Running pimpleFoam..."
pimpleFoam > log.pimpleFoam 2>&1 &

# Monitor progress
SOLVER_PID=$!
COUNTER=0
while kill -0 $SOLVER_PID 2>/dev/null; do
    if [ -f "log.pimpleFoam" ]; then
        CURRENT_TIME=$(grep "^Time = " log.pimpleFoam | tail -1 | awk '{print $3}')
        if [ ! -z "$CURRENT_TIME" ]; then
            echo "Progress: Time = $CURRENT_TIME"
        fi
    fi
    sleep 5
    COUNTER=$((COUNTER + 1))
    
    # Timeout after 30 minutes
    if [ $COUNTER -gt 360 ]; then
        echo "⚠️ Simulation timeout (30 minutes)"
        kill $SOLVER_PID
        break
    fi
done

# Check if solver completed successfully
if grep -q "End" log.pimpleFoam 2>/dev/null; then
    echo "✓ Simulation completed successfully"
    
    # Post-processing
    echo "Running post-processing..."
    
    # Calculate vorticity
    pimpleFoam -postProcess -func vorticity -latestTime > /dev/null 2>&1
    
    # Sample data
    if [ -f "system/sample" ]; then
        sample -latestTime > /dev/null 2>&1
    fi
    
    # Convert to VTK for visualization
    foamToVTK -latestTime > /dev/null 2>&1
    
    echo "✓ Post-processing completed"
else
    echo "❌ Simulation failed or incomplete"
    tail -50 log.pimpleFoam
    exit 1
fi

echo "======================================"
echo "Simulation Complete"
echo "======================================"
echo "Results saved in: $CASE_PATH"
echo "VTK files in: $CASE_PATH/VTK"
EOF

chmod +x $CASE_DIR/run_simulation.sh

# Run Docker container
echo "Starting Docker container..."
docker run --rm \
    -v "$(pwd):$WORK_DIR" \
    -w $WORK_DIR \
    $DOCKER_IMAGE \
    bash $CASE_DIR/run_simulation.sh $CONFIG_FILE $CASE_DIR

echo ""
echo "======================================"
echo "Docker execution completed"
echo "======================================"

# Check results
if [ -d "$CASE_DIR/case/VTK" ]; then
    echo "✓ VTK files generated:"
    ls -la $CASE_DIR/case/VTK/ | head -5
    
    # Run Python analysis on the results
    echo ""
    echo "Running analysis..."
    python3 -m amx.cli analyze-mix --input $CASE_DIR --output data/processed/docker_results
else
    echo "❌ No VTK output found"
fi