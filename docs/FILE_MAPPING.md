# File-to-Process Mapping

This document provides a comprehensive mapping of which files are used in each process of the simulation pipeline.

## 📁 Complete File Inventory

### Core Framework Files
```
src/amx/core/
├── base.py           → [All Processes] Base classes for entire framework
├── interfaces.py     → [Model Creation] Defines model contracts
└── exceptions.py     → [Error Handling] Used throughout pipeline
```

### Configuration Files
```
configs/
├── case_prod.yaml    → [Process 1] Production configuration
├── case_sensitivity.yaml → [Sensitivity Analysis] Parameter studies
└── piv_lab.yaml      → [PIV Validation] Experimental comparison
```

### Physics Model Files
```
src/amx/physics/
├── models.py         → [Process 2] Fluid, turbulence, mixing models
├── sludge_rheology.py → [Process 2] Non-Newtonian sludge properties
├── jet_model.py      → [Process 3] Jet momentum calculations
├── mixing_theory.py  → [Process 6] Mixing time correlations
└── turbulence.py     → [Legacy] Refactored into models.py
```

---

## 🔄 Process-Specific File Usage

### Process 1: Configuration & Initialization

| File | Purpose | When Used |
|------|---------|-----------|
| `src/amx/cli.py` | Entry point | Start of simulation |
| `src/amx/config.py` | Configuration validation | After YAML loading |
| `configs/*.yaml` | User parameters | Initial input |
| `src/amx/utils/paths.py` | Path management | Directory setup |
| `src/amx/utils/logging.py` | Logging setup | Throughout process |

**Data Flow:**
```
CLI Arguments → config.yaml → Config Object → Validated Parameters
```

---

### Process 2: Model Factory & Physics Setup

| File | Purpose | Output |
|------|---------|--------|
| `src/amx/factories.py` | Model creation | Model instances |
| `src/amx/physics/models.py` | Model implementations | Physics objects |
| `src/amx/physics/sludge_rheology.py` | Sludge properties | Rheology parameters |
| `src/amx/core/interfaces.py` | Model contracts | Interface compliance |

**Model Creation Chain:**
```python
ModelFactory.create()
├── FluidFactory → NonNewtonianFluid (sludge_rheology.py)
├── TurbulenceFactory → StandardKEpsilon (models.py)
└── MixingFactory → ComprehensiveMixingModel (models.py)
```

---

### Process 3: OpenFOAM Case Setup

| File | Purpose | Generated Output |
|------|---------|------------------|
| `src/amx/workflow.py` | Orchestration | Controls flow |
| `src/amx/openfoam/writer.py` | Dictionary writing | OpenFOAM dicts |
| `src/amx/openfoam/fvoptions.py` | Momentum sources | `fvOptions` file |
| `src/amx/geometry.py` | Geometry calculations | Nozzle positions |
| `case_templates/0/*` | Initial conditions | `0/` directory |
| `case_templates/constant/*` | Properties | `constant/` directory |
| `case_templates/system/*` | Solver settings | `system/` directory |

**File Generation Map:**
```
Template Files + Configuration → OpenFOAM Case Structure
├── 0/U, p_rgh, k, epsilon    (from templates + models.py)
├── constant/fvOptions         (from fvoptions.py)
├── constant/transportProperties (from fluid model)
└── system/controlDict         (from solver config)
```

---

### Process 4: Mesh Generation

| File | Purpose | Commands Executed |
|------|---------|-------------------|
| `src/amx/openfoam/meshing.py` | Mesh control | `blockMesh`, `snappyHexMesh` |
| `src/amx/geometry.py` | Domain geometry | Tank dimensions |
| Generated: `system/blockMeshDict` | Block structure | Base mesh |
| Generated: `system/snappyHexMeshDict` | Refinements | Local refinement |

**Mesh Pipeline:**
```bash
meshing.py → blockMeshDict → blockMesh → base mesh
           → snappyHexMeshDict → snappyHexMesh → refined mesh
           → checkMesh → quality report
```

---

### Process 5: Solver Execution

| File | Purpose | Role in Execution |
|------|---------|-------------------|
| `src/amx/openfoam/runner.py` | Solver management | Process control |
| `system/controlDict` | Time control | Simulation parameters |
| `system/fvSchemes` | Discretization | Numerical schemes |
| `system/fvSolution` | Linear solvers | Solution methods |
| `constant/fvOptions` | Momentum sources | Jet modeling |

**Execution Flow:**
```
runner.py → pimpleFoam
         → Monitor: residuals, Courant, continuity
         → Write: time directories (100, 200, ..., 1800)
         → Log: solver.log
```

---

### Process 6: Post-Processing & Analysis

| File | Purpose | Analysis Type |
|------|---------|---------------|
| `src/amx/post/advanced_metrics.py` | Comprehensive metrics | Dead zones, RTD, energy |
| `src/amx/post/metrics.py` | Basic analysis | Velocity, mixing |
| `src/amx/post/fields.py` | Field extraction | VTK conversion |
| `src/amx/post/io.py` | Data I/O | File reading |
| `src/amx/post/viz.py` | Visualization | 3D rendering |
| `src/amx/utils/paths.py` | Result paths | File locations |

**Analysis Pipeline:**
```
Time Directories → VTK Files → Field Arrays → Metrics
                              ↓
                        Visualization
                              ↓
                          Reports
```

---

### Process 7: Energy Analysis

| File | Purpose | Calculations |
|------|---------|--------------|
| `src/amx/energy/power.py` | Power calculations | Pump power, G-value |
| `src/amx/physics/mixing_theory.py` | Theory correlations | Camp number |
| Configuration data | Operating parameters | Flow rate, head |

**Energy Flow:**
```
Operating Parameters → Power Calculation → G-value → Mixing Efficiency
                    → Energy Consumption → Optimization
```

---

### Process 8: PIV Validation (Optional)

| File | Purpose | Function |
|------|---------|----------|
| `src/amx/piv/processing.py` | PIV data processing | Image analysis |
| `src/amx/piv/compare.py` | CFD-PIV comparison | Correlation |
| `configs/piv_lab.yaml` | PIV configuration | Experimental setup |
| `data/raw/piv/*` | PIV images | Input data |

---

## 📊 File Dependencies Matrix

| Module | Depends On | Used By |
|--------|-----------|---------|
| `workflow.py` | config, factories, openfoam, post | CLI |
| `factories.py` | physics/models, core/interfaces | workflow |
| `models.py` | core/interfaces, sludge_rheology | factories |
| `fvoptions.py` | config, geometry, jet_model | workflow |
| `runner.py` | config, core/base | workflow |
| `advanced_metrics.py` | numpy, scipy, core/base | workflow |
| `power.py` | config, mixing_theory | workflow |

---

## 🗂️ File Categories

### 1. **Configuration Files** (User Input)
- `configs/*.yaml`
- Define simulation parameters

### 2. **Template Files** (Static Resources)
- `case_templates/**/*`
- OpenFOAM dictionary templates

### 3. **Core Library** (Framework)
- `src/amx/core/*`
- Base classes and interfaces

### 4. **Domain Models** (Physics)
- `src/amx/physics/*`
- Physical model implementations

### 5. **Infrastructure** (OpenFOAM Interface)
- `src/amx/openfoam/*`
- OpenFOAM interaction layer

### 6. **Analysis Tools** (Post-Processing)
- `src/amx/post/*`
- Result analysis and visualization

### 7. **Utilities** (Support)
- `src/amx/utils/*`
- Helper functions and tools

### 8. **Generated Files** (Runtime)
- `runs/*/case/**/*`
- OpenFOAM case files
- `data/processed/*`
- Analysis results

---

## 🔍 File Search Guide

### To modify fluid properties:
→ `src/amx/physics/models.py` (NewtonianFluid, NonNewtonianFluid)
→ `src/amx/physics/sludge_rheology.py` (SludgeProperties)

### To change mesh resolution:
→ `configs/case_prod.yaml` (mesh section)
→ `src/amx/openfoam/meshing.py` (MeshGenerator)

### To adjust jet configuration:
→ `configs/case_prod.yaml` (nozzle section)
→ `src/amx/openfoam/fvoptions.py` (momentum sources)
→ `src/amx/physics/jet_model.py` (jet physics)

### To modify solver settings:
→ `configs/case_prod.yaml` (solver section)
→ `case_templates/system/controlDict`
→ `case_templates/system/fvSolution`

### To add new metrics:
→ `src/amx/post/advanced_metrics.py` (MixingMetricsAdvanced)
→ Implement new method in class

### To change turbulence model:
→ `src/amx/physics/models.py` (TurbulenceModel implementations)
→ `src/amx/factories.py` (TurbulenceFactory)

---

## 📝 File Modification Impact

### High Impact Files (Affect Multiple Processes):
- `src/amx/config.py` - Changes affect entire pipeline
- `src/amx/workflow.py` - Central orchestration
- `src/amx/factories.py` - Model creation
- `configs/*.yaml` - User parameters

### Medium Impact Files (Affect Specific Domain):
- `src/amx/physics/models.py` - Physics calculations
- `src/amx/openfoam/fvoptions.py` - Momentum sources
- `src/amx/post/advanced_metrics.py` - Analysis results

### Low Impact Files (Localized Changes):
- `src/amx/post/viz.py` - Only visualization
- `src/amx/utils/logging.py` - Only logging
- `case_templates/*` - Template modifications

---

## 🚀 Quick Reference: Common Tasks

| Task | Primary Files |
|------|--------------|
| Change fluid from water to sludge | `configs/case_prod.yaml`, `factories.py` |
| Increase mesh resolution | `configs/case_prod.yaml`, `meshing.py` |
| Add more nozzles | `configs/case_prod.yaml`, `fvoptions.py` |
| Modify jet angle | `configs/case_prod.yaml`, `jet_model.py` |
| Change simulation time | `configs/case_prod.yaml`, `controlDict` |
| Add new analysis metric | `advanced_metrics.py` |
| Optimize energy consumption | `power.py`, `mixing_theory.py` |
| Compare with experiments | `piv/compare.py` |