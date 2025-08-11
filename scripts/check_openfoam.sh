#!/bin/bash
# OpenFOAM 환경 확인 스크립트

echo "======================================"
echo "OpenFOAM Environment Check"
echo "======================================"

# Check if OpenFOAM is installed
check_openfoam() {
    echo -n "Checking OpenFOAM installation... "
    
    # Try different OpenFOAM versions
    for version in 11 10 9 8 v2312 v2306 v2212; do
        if [ -d "/opt/openfoam${version}" ] || [ -d "/usr/lib/openfoam/openfoam${version}" ] || [ -d "$HOME/OpenFOAM/OpenFOAM-${version}" ]; then
            echo "Found OpenFOAM-${version}"
            FOAM_VERSION=${version}
            return 0
        fi
    done
    
    # Check if OpenFOAM command exists
    if command -v simpleFoam &> /dev/null; then
        echo "OpenFOAM commands found in PATH"
        return 0
    fi
    
    echo "NOT FOUND"
    return 1
}

# Check environment variables
check_env_vars() {
    echo -e "\nChecking environment variables:"
    
    vars=("WM_PROJECT" "WM_PROJECT_VERSION" "WM_PROJECT_DIR" "FOAM_USER_APPBIN" "FOAM_USER_LIBBIN")
    
    for var in "${vars[@]}"; do
        if [ -z "${!var}" ]; then
            echo "  ❌ $var is not set"
        else
            echo "  ✓ $var = ${!var}"
        fi
    done
}

# Check required solvers
check_solvers() {
    echo -e "\nChecking required solvers:"
    
    solvers=("pimpleFoam" "simpleFoam" "blockMesh" "decomposePar" "reconstructPar")
    
    for solver in "${solvers[@]}"; do
        if command -v $solver &> /dev/null; then
            echo "  ✓ $solver found"
        else
            echo "  ❌ $solver NOT found"
        fi
    done
}

# Check Python OpenFOAM tools
check_python_tools() {
    echo -e "\nChecking Python environment:"
    
    python3 -c "import numpy; print(f'  ✓ NumPy {numpy.__version__}')" 2>/dev/null || echo "  ❌ NumPy not installed"
    python3 -c "import pyvista; print(f'  ✓ PyVista {pyvista.__version__}')" 2>/dev/null || echo "  ❌ PyVista not installed"
    python3 -c "import pandas; print(f'  ✓ Pandas {pandas.__version__}')" 2>/dev/null || echo "  ❌ Pandas not installed"
}

# Main execution
echo "1. OpenFOAM Installation:"
if check_openfoam; then
    check_env_vars
    check_solvers
else
    echo -e "\n⚠️  OpenFOAM is not installed or not properly configured"
    echo -e "\nTo install OpenFOAM on Ubuntu/Debian:"
    echo "  sudo sh -c \"wget -O - https://dl.openfoam.org/gpg.key | apt-key add -\""
    echo "  sudo add-apt-repository http://dl.openfoam.org/ubuntu"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install openfoam11"
    echo -e "\nTo install on macOS:"
    echo "  Use Docker: docker pull openfoam/openfoam11-paraview510"
fi

check_python_tools

echo -e "\n======================================"
echo "Recommendations:"
echo "======================================"

if ! check_openfoam &> /dev/null; then
    echo "1. Install OpenFOAM or use Docker container"
    echo "2. Source OpenFOAM environment:"
    echo "   source /opt/openfoam11/etc/bashrc"
else
    echo "✓ OpenFOAM environment appears to be configured"
fi

echo -e "\nTo use Docker instead:"
echo "  docker run -it -v \$(pwd):/work openfoam/openfoam11-paraview510"