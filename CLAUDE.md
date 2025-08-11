# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an industrial-grade OpenFOAM-based CFD simulation framework for anaerobic digester mixing. It features:
- OpenFOAM v11 for high-fidelity CFD simulations (pimpleFoam solver)
- Refactored Python framework with clean architecture patterns
- Non-Newtonian rheology models for accurate sludge simulation
- Advanced mixing metrics and performance analysis
- PIV (Particle Image Velocimetry) validation capabilities
- Comprehensive energy optimization tools

## Common Development Commands

### Installation and Setup
```bash
# Install package with development dependencies
pip install -e .[dev]

# Quick setup with all dev tools
make setup
```

### Testing
```bash
# Run unit tests
pytest
make test

# Run tests with coverage
pytest --cov=amx
make test-cov
```

### Code Quality
```bash
# Run all linting checks
make lint
# Or individually:
ruff check .
black --check .
isort --check-only .

# Auto-format code
make format
# Or individually:
ruff check --fix .
black .
isort .
```

### Running Simulations
```bash
# Run production case via CLI
amx run-case --config configs/case_prod.yaml --out runs/prod

# Or via make
make run

# Analyze results
amx analyze-mix --in runs/prod --out data/processed/prod_metrics
make analyze

# PIV validation
amx analyze-piv --config configs/piv_lab.yaml --cfd runs/prod --piv data/raw/piv
```

### Docker Operations
```bash
# Build Docker image
make docker-build

# Run interactively
make docker-run

# Execute simulation in Docker
docker run -v $(pwd):/app amx-openfoam amx run-case --config configs/case_prod.yaml --out runs/prod
```

### OpenFOAM Environment Check
```bash
# Verify OpenFOAM installation
./scripts/check_openfoam.sh
```

## High-Level Architecture (Refactored)

### Core Framework (`src/amx/core/`)
- **base.py**: Abstract base classes for simulations and analyzers
- **interfaces.py**: Clean interfaces for physics models (FluidModel, TurbulenceModel, etc.)
- **exceptions.py**: Structured exception hierarchy for better error handling

### Factory Pattern (`src/amx/factories.py`)
- **ModelFactory**: Main factory for creating all models from configuration
- **FluidFactory**: Creates Newtonian/Non-Newtonian fluid models
- **TurbulenceFactory**: Creates turbulence models (k-ε, k-ω, etc.)
- **MixingFactory**: Creates mixing analysis models

### Physics Modeling (`src/amx/physics/`)
- **models.py**: Unified physics models with clean interfaces
  - `NewtonianFluid`: Standard fluid properties
  - `NonNewtonianFluid`: Sludge rheology (Herschel-Bulkley, Power-law, etc.)
  - `StandardKEpsilon`: Turbulence modeling
  - `ComprehensiveMixingModel`: Mixing analysis
- **sludge_rheology.py**: Advanced rheological models for anaerobic sludge
  - Temperature-dependent viscosity
  - Total solids correlations
  - Multiple rheology models (validated against literature)
- **jet_model.py**: Jet dynamics with configurable tank geometry
- **mixing_theory.py**: Theoretical mixing correlations

### OpenFOAM Interface (`src/amx/openfoam/`)
- **fvoptions.py**: Enhanced momentum source with Gaussian distribution
- **writer.py**: Template-based OpenFOAM dictionary generation
- **meshing.py**: Industrial-grade mesh (5-7M cells)
- **runner.py**: Robust solver execution with error handling

### Advanced Post-Processing (`src/amx/post/`)
- **advanced_metrics.py**: Comprehensive industrial metrics
  - Dead zone quantification with connectivity analysis
  - Residence time distribution (RTD)
  - Energy uniformity assessment
  - Automatic performance grading
- **metrics.py**: Standard mixing performance metrics
- **viz.py**: 3D visualization with PyVista

### Workflow Orchestration (`src/amx/workflow.py`)
Main simulation pipeline with improved error handling:
1. Model creation via factories
2. Validation checks
3. OpenFOAM case setup
4. Solver execution with monitoring
5. Comprehensive post-processing
6. Report generation with recommendations

**Configuration System**
- YAML-based configuration in `configs/`
- Pydantic models in `src/amx/config.py` for validation
- Case templates in `case_templates/` for OpenFOAM setup

### Key Improvements in Refactored Code

1. **Clean Architecture**
   - Abstract base classes for extensibility
   - Interface segregation for physics models
   - Factory pattern for flexible model creation
   - Proper exception hierarchy

2. **Industrial-Grade Physics**
   - Non-Newtonian sludge rheology (3-4% TS)
   - Temperature-dependent properties
   - Validated correlations from literature
   - Multiple rheology models (Herschel-Bulkley, Power-law, Casson)

3. **Enhanced Simulation Parameters**
   - Solver: pimpleFoam (unsteady, incompressible)
   - Turbulence: k-ε with proper boundary conditions
   - Mesh: 5-7M cells with refinement zones
   - Fluid: Anaerobic sludge (ρ=1015 kg/m³, μ=0.0035 Pa·s at 3.5% TS)
   - Momentum sources: 32 jets with Gaussian distribution
   - Performance targets:
     - Mean velocity ≥0.30 m/s
     - Dead zones <10% volume
     - Mixing time ≤30 minutes
     - Energy uniformity >0.7

4. **Code Quality**
   - Type hints throughout
   - Comprehensive docstrings
   - Validation at every level
   - Proper error handling
   - Factory pattern for model selection

### CLI Tool (`amx`)
All functionality exposed through the `amx` CLI with commands:
- `run-case`: Execute full simulation
- `analyze-mix`: Compute mixing metrics
- `analyze-piv`: PIV validation
- `make-figures`: Generate visualizations
- `build-report`: Create analysis report

Entry point is `src/amx/cli.py` using Typer framework.